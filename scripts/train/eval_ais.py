"""Test « AIS » : vérité terrain indépendante, l'AIS ouvert de l'Autorité maritime danoise.

Pour chaque zone : passe Sentinel-2 lue comme le pipeline (fenêtre du TCI L2A,
masque d'eau SCL), détection par tuiles 640 / chevauchement 96, fusion, filtre
eau 0,25 comme `geo.to_features`. Puis appariement optimal détection ↔ position
AIS interpolée à l'heure de la passe, seuil 500 m.

Lecture des chiffres :
  - le rappel AIS est fiable pour les grands navires (> 50 m, AIS obligatoire) ;
  - la « précision AIS » (détections appariées / détections) est une borne basse :
    un bateau sans AIS (plaisance, pêche) compte comme faux positif.

Le code AIS est repris de `src/vigie/ais.py` (branche main, lead), sans la partie scan.

Usage : uv run python scripts/train/eval_ais.py --set val --weights A.pt B.pt --out runs/eval/ais
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from vigie import sentinel
from vigie.ais import AIS_DIR
from vigie.geo import to_features
from vigie.thermal import wait_if_hot
from vigie.tiling import Box, make_tiles, merge

WINDOW_MIN = 20
GATE_M = 500
MARGIN_DEG = 0.03
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

# Zones : passe claire, trafic dense, couverture AIS danoise.
# « test » reprend exactement les cinq cas de scripts/validate_ais.py (main, 34afc84).
# « val » sert à choisir modèle et seuil ; aucune de ses zones n'est dans le test.
ZONE_SETS = {
    "test": [
        {"id": "skagen", "scene": "S2B_32VNJ_20261018_0_L2A", "bbox": [10.50, 57.55, 10.80, 57.72]},
        {"id": "aarhus", "scene": "S2B_32VNH_20261018_0_L2A", "bbox": [10.18, 56.08, 10.40, 56.20]},
        {"id": "fredericia", "scene": "S2B_32UNG_20261018_0_L2A", "bbox": [9.65, 55.50, 9.85, 55.62]},
        {"id": "oresund", "scene": "S2C_33UUB_20260927_0_L2A", "bbox": [12.55, 55.58, 12.80, 55.75]},
        {"id": "grand_belt", "scene": "S2C_32UPG_20260930_0_L2A", "bbox": [10.85, 55.25, 11.10, 55.40]},
    ],
    "val": [
        {"id": "fehmarnbelt", "scene": "S2C_32UPF_20260927_0_L2A", "bbox": [11.00, 54.45, 11.35, 54.65]},
        {"id": "langeland", "scene": "S2C_32UPF_20260930_0_L2A", "bbox": [10.85, 54.75, 11.10, 54.95]},
        {"id": "frederikshavn", "scene": "S2B_32VPJ_20261018_0_L2A", "bbox": [10.55, 57.38, 10.75, 57.50]},
    ],
}


# --- AIS, repris de vigie.ais -------------------------------------------------------------

def load_window(csv: Path, bbox, t0: datetime) -> pl.DataFrame:
    w, s, e, n = bbox
    lo, hi = t0 - timedelta(minutes=WINDOW_MIN), t0 + timedelta(minutes=WINDOW_MIN)
    f = pl.col
    return (
        pl.scan_csv(csv, infer_schema_length=0)
        .rename({"# Timestamp": "ts"})
        .filter(f("Type of mobile").is_in(["Class A", "Class B"]))
        .with_columns(f("ts").str.strptime(pl.Datetime("us", "UTC"), "%d/%m/%Y %H:%M:%S", strict=False),
                      *(f(c).cast(pl.Float64, strict=False) for c in ("Latitude", "Longitude", "SOG", "COG",
                                                                        "Length")))
        .filter(f("ts").is_between(lo, hi)
                & f("Latitude").is_between(s - MARGIN_DEG, n + MARGIN_DEG)
                & f("Longitude").is_between(w - MARGIN_DEG, e + MARGIN_DEG))
        .unique(subset=["MMSI", "ts"])
        .sort("ts")
        .collect()
    )


def position_at(track: pl.DataFrame, t0: datetime) -> tuple[float, float] | None:
    ts = track["ts"].to_list()
    lat, lon = track["Latitude"].to_numpy(), track["Longitude"].to_numpy()
    secs = np.array([(t - t0).total_seconds() for t in ts])
    before, after = np.where(secs <= 0)[0], np.where(secs >= 0)[0]
    if len(before) and len(after):
        i, j = before[-1], after[0]
        if i == j or secs[j] == secs[i]:
            return float(lat[i]), float(lon[i])
        a = -secs[i] / (secs[j] - secs[i])
        return float(lat[i] + a * (lat[j] - lat[i])), float(lon[i] + a * (lon[j] - lon[i]))
    k = before[-1] if len(before) else after[0]
    dt = -secs[k]
    sog, cog = track["SOG"][int(k)], track["COG"][int(k)]
    if abs(dt) > 600:
        return None
    if sog is None or cog is None or sog < 0.5 or sog > 60:
        return float(lat[k]), float(lon[k])
    d = sog * 0.5144 * dt
    return (float(lat[k] + d * math.cos(math.radians(cog)) / 111320),
            float(lon[k] + d * math.sin(math.radians(cog)) / (111320 * math.cos(math.radians(lat[k])))))


def vessels_at(df: pl.DataFrame, bbox, t0: datetime) -> list[dict]:
    w, s, e, n = bbox
    out = []
    for (mmsi,), track in df.group_by(["MMSI"]):
        pos = position_at(track, t0)
        if pos is None or not (s <= pos[0] <= n and w <= pos[1] <= e):
            continue
        lengths = [v for v in track["Length"].to_list() if v]
        sogs = [v for v in track["SOG"].to_list() if v is not None]
        out.append({"mmsi": mmsi, "lat": pos[0], "lon": pos[1], "length_m": lengths[-1] if lengths else None,
                    "sog_kn": sogs[-1] if sogs else None})
    return out


def dist_m(lat1, lon1, lat2, lon2) -> float:
    dy = (lat2 - lat1) * 111320
    dx = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def match(dets: list[dict], vessels: list[dict], gate: float = GATE_M) -> list[tuple[int, int, float]]:
    if not dets or not vessels:
        return []
    cost = np.array([[dist_m(d["lat"], d["lon"], v["lat"], v["lon"]) for v in vessels] for d in dets])
    big = cost.max() + gate * 10
    rows, cols = linear_sum_assignment(np.where(cost <= gate, cost, big))
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] <= gate]


# --- Détection, comme vigie.detector mais avec un seuil bas pour balayer les seuils --------

def detect(model, rgb: np.ndarray, device, conf: float = 0.05) -> list[Box]:
    h, w = rgb.shape[:2]
    tiles = make_tiles(w, h, 640, 96)
    found = []
    for i in range(0, len(tiles), 16):
        wait_if_hot()
        chunk = tiles[i:i + 16]
        crops = [np.ascontiguousarray(rgb[t.y:t.y + t.height, t.x:t.x + t.width, ::-1]) for t in chunk]
        for res, t in zip(model.predict(crops, conf=conf, iou=0.45, imgsz=640, max_det=500, device=device,
                                        verbose=False), chunk):
            for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                found.append(Box(x1 + t.x, y1 + t.y, x2 + t.x, y2 + t.y, float(c)))
    return merge(found, 0.45)


def get_scene(scene_id: str) -> sentinel.Scene:
    r = httpx.get(f"{sentinel.STAC}/collections/{sentinel.COLLECTION}/items/{scene_id}", timeout=30)
    r.raise_for_status()
    return sentinel._feature_to_scene(r.json())


def score(dets: list[dict], vessels: list[dict]) -> dict:
    pairs = match(dets, vessels)
    mv = {c for _, c, _ in pairs}

    def rec(min_len):
        sel = [j for j, v in enumerate(vessels) if (v["length_m"] or 0) >= min_len]
        return sum(j in mv for j in sel), len(sel)

    out = {"detections": len(dets), "matched": len(pairs), "ais": len(vessels)}
    for k, lim in (("all", 0), ("ge50m", 50), ("ge100m", 100)):
        hit, n = rec(lim)
        out[f"hit_{k}"], out[f"n_{k}"] = hit, n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/eval/ais"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--set", default="val", choices=list(ZONE_SETS))
    args = ap.parse_args()
    zone_list = ZONE_SETS[args.set]
    args.out.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    zones = []
    for z in zone_list:
        sc = get_scene(z["scene"])
        t0 = datetime.fromisoformat(sc.datetime).astimezone(UTC)
        csv = AIS_DIR / f"aisdk-{t0:%Y-%m-%d}.csv"
        started = time.perf_counter()
        vessels = vessels_at(load_window(csv, z["bbox"], t0), z["bbox"], t0)
        raster = sentinel.load(sc, tuple(z["bbox"]))
        print(f"{z['id']} : {raster.rgb.shape[1]}x{raster.rgb.shape[0]} px, {len(vessels)} AIS, "
              f"{time.perf_counter() - started:.0f} s", flush=True)
        zones.append((z, sc, raster, vessels))
    results = {}
    for wpath in args.weights:
        model = YOLO(str(wpath))
        per_zone = {}
        for z, _sc, raster, vessels in zones:
            boxes = detect(model, raster.rgb, args.device)
            feats = to_features(boxes, raster.transform, raster.crs, raster.gsd, raster.water)
            dets = [{"lat": f["properties"]["lat"], "lon": f["properties"]["lon"],
                     "conf": f["properties"]["confidence"]} for f in feats]
            per_zone[z["id"]] = {str(t): score([d for d in dets if d["conf"] >= t], vessels) for t in THRESHOLDS}
        # Agrégat sur toutes les zones, par seuil.
        agg = {}
        for t in map(str, THRESHOLDS):
            keys = per_zone[next(iter(per_zone))][t].keys()
            agg[t] = {k: sum(per_zone[zid][t][k] for zid in per_zone) for k in keys}
        results[wpath.stem] = {"zones": per_zone, "total": agg}
        a = agg["0.2"]
        print(f"{wpath.stem} @0,20 : {a['detections']} dét., {a['matched']} appariées, "
              f"rappel {a['hit_all']}/{a['n_all']}, >50 m {a['hit_ge50m']}/{a['n_ge50m']}", flush=True)
    meta = {"set": args.set, "zones": zone_list, "gate_m": GATE_M, "window_min": WINDOW_MIN, "water_filter": 0.25}
    (args.out / f"metrics_ais_{args.set}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    lines = [("| Modèle | seuil | détections | appariées AIS | préc. AIS (borne basse) | rappel AIS tous "
              "| rappel > 50 m | rappel > 100 m |"), "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        for t in ("0.2", "0.3", "0.5"):
            a = r["total"][t]
            lines.append(
                f"| {name} | {t} | {a['detections']} | {a['matched']} | {a['matched'] / max(a['detections'], 1):.2f} "
                f"| {a['hit_all']}/{a['n_all']} ({a['hit_all'] / max(a['n_all'], 1):.2f}) "
                f"| {a['hit_ge50m']}/{a['n_ge50m']} ({a['hit_ge50m'] / max(a['n_ge50m'], 1):.2f}) "
                f"| {a['hit_ge100m']}/{a['n_ge100m']} ({a['hit_ge100m'] / max(a['n_ge100m'], 1):.2f}) |")
    lines += ["", "Par zone, seuil 0,20 (détections / appariées / AIS > 50 m trouvés) :", "",
              "| Modèle | " + " | ".join(z["id"] for z in zone_list) + " |", "|---|" + "---|" * len(zone_list)]
    for name, r in results.items():
        cells = []
        for z in zone_list:
            a = r["zones"][z["id"]]["0.2"]
            cells.append(f"{a['detections']} / {a['matched']} / {a['hit_ge50m']}/{a['n_ge50m']}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (args.out / f"metrics_ais_{args.set}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
