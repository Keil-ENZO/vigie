"""Croisement avec l'AIS : qui émet, qui se tait.

Chaque navire de plus de 300 tonneaux doit émettre sa position par AIS. Un navire
vu par le satellite mais absent de l'AIS est un « navire sombre » : balise coupée,
trop petit pour l'obligation, ou fausse détection. L'inverse, l'AIS sans détection,
mesure le rappel du détecteur.

Source : les données AIS ouvertes de l'Autorité maritime danoise, un CSV par jour,
https://aisdata.ais.dk. On ne garde que la zone et une fenêtre autour de la passe.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from . import scan

AIS_DIR = Path(os.environ.get("VIGIE_AIS_DIR",
                            Path(os.environ.get("VIGIE_DATA", Path(__file__).resolve().parents[2] / "data")) / "ais"))
DMA_URL = "http://aisdata.ais.dk/aisdk-{day}.zip"
WINDOW_MIN = 20  # minutes de part et d'autre de la passe
GATE_M = 500  # distance maximale d'appariement : antenne décalée, pixel de 10 m, horloge
MARGIN_DEG = 0.03

TYPE_MAP = {
    "Cargo": "cargo", "Tanker": "tanker", "Passenger": "passenger", "Fishing": "fishing",
    "Pleasure": "pleasure", "Sailing": "pleasure", "Military": "military", "Law enforcement": "service",
    "Tug": "service", "Towing": "service", "Towing long/wide": "service", "Pilot": "service",
    "SAR": "service", "Port tender": "service", "Dredging": "service", "Diving": "service",
    "HSC": "passenger", "Anti-pollution": "service", "Medical": "service",
}


# Emprise approximative de la couverture des stations côtières danoises.
COVERAGE = (3.0, 53.5, 17.0, 59.0)


def covers(bbox) -> bool:
    w, s, e, n = bbox
    cw, cs, ce, cn = COVERAGE
    return cw <= w and cs <= s and e <= ce and n <= cn


def csv_path(day: str) -> Path:
    return AIS_DIR / f"aisdk-{day}.csv"


def ensure_day(day: str, log=print) -> Path:
    """Télécharge et décompresse le fichier du jour s'il manque. Environ 600 Mo zippé."""
    csv = csv_path(day)
    if csv.exists():
        return csv
    AIS_DIR.mkdir(parents=True, exist_ok=True)
    zpath = AIS_DIR / f"aisdk-{day}.zip"
    if not zpath.exists():
        log(f"téléchargement de l'AIS danois du {day}")
        tmp = zpath.with_suffix(".part")
        with httpx.stream("GET", DMA_URL.format(day=day), timeout=600) as r:
            if r.status_code == 404:
                raise LookupError(f"pas encore d'AIS danois publié pour le {day}")
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.rename(zpath)
    log("décompression")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(AIS_DIR)
    return csv


def load_window(csv: Path, bbox, t0: datetime) -> pl.DataFrame:
    w, s, e, n = bbox
    lo, hi = t0 - timedelta(minutes=WINDOW_MIN), t0 + timedelta(minutes=WINDOW_MIN)
    f = pl.col
    return (
        pl.scan_csv(csv, infer_schema_length=0)
        .rename({"# Timestamp": "ts"})
        .filter(f("Type of mobile").is_in(["Class A", "Class B"]))
        .with_columns(f("ts").str.strptime(pl.Datetime("us", "UTC"), "%d/%m/%Y %H:%M:%S", strict=False),
                      *(f(c).cast(pl.Float64, strict=False) for c in ("Latitude", "Longitude", "SOG", "COG", "Length",
                                                                        "Width")))
        .filter(f("ts").is_between(lo, hi)
                & f("Latitude").is_between(s - MARGIN_DEG, n + MARGIN_DEG)
                & f("Longitude").is_between(w - MARGIN_DEG, e + MARGIN_DEG))
        .unique(subset=["MMSI", "ts"])
        .sort("ts")
        .collect()
    )


def position_at(track: pl.DataFrame, t0: datetime) -> tuple[float, float] | None:
    """Position interpolée à l'instant de la passe, ou extrapolée par vitesse et route."""
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


def _last(track: pl.DataFrame, col: str):
    vals = [v for v in track[col].to_list() if v not in (None, "", "Unknown", "Undefined")]
    return vals[-1] if vals else None


def vessels_at(df: pl.DataFrame, bbox, t0: datetime) -> list[dict]:
    w, s, e, n = bbox
    out = []
    for (mmsi,), track in df.group_by(["MMSI"]):
        pos = position_at(track, t0)
        if pos is None or not (s <= pos[0] <= n and w <= pos[1] <= e):
            continue
        ship_type = _last(track, "Ship type")
        sog = _last(track, "SOG")
        out.append({
            "mmsi": mmsi, "name": _last(track, "Name"), "ship_type": ship_type,
            "category": TYPE_MAP.get(ship_type or "", "unknown"),
            "length_m": _last(track, "Length"), "width_m": _last(track, "Width"),
            "sog_kn": sog, "cog_deg": _last(track, "COG"), "nav_status": _last(track, "Navigational status"),
            "destination": _last(track, "Destination"), "class": _last(track, "Type of mobile"),
            "lat": round(pos[0], 6), "lon": round(pos[1], 6), "fixes": track.height,
        })
    return out


def _dist_m(lat1, lon1, lat2, lon2) -> float:
    dy = (lat2 - lat1) * 111320
    dx = (lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def match(features: list[dict], vessels: list[dict], gate: float = GATE_M) -> list[tuple[int, int, float]]:
    """Appariement optimal détection ↔ AIS, distance minimale, au-delà du seuil rien."""
    if not features or not vessels:
        return []
    cost = np.array([[_dist_m(f["properties"]["lat"], f["properties"]["lon"], v["lat"], v["lon"])
                      for v in vessels] for f in features])
    big = cost.max() + gate * 10
    rows, cols = linear_sum_assignment(np.where(cost <= gate, cost, big))
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] <= gate]


def crosscheck(scan_id: str, log=print) -> dict:
    meta, det = scan.load_scan(scan_id)
    t0 = datetime.fromisoformat(meta["scene"]["datetime"].replace("Z", "+00:00")).astimezone(timezone.utc)
    day = t0.strftime("%Y-%m-%d")
    csv = ensure_day(day, log)
    log("lecture de l'AIS autour de la passe")
    df = load_window(csv, meta["bbox"], t0)
    # L'image est en UTM : ses coins débordent de la zone en lon/lat. On apparie donc
    # contre l'AIS d'une zone élargie, sinon un navire du coin passe pour « sombre ».
    w, so, e, n = meta["bbox"]
    vessels = vessels_at(df, (w - MARGIN_DEG, so - MARGIN_DEG, e + MARGIN_DEG, n + MARGIN_DEG), t0)
    inside = [w <= v["lon"] <= e and so <= v["lat"] <= n for v in vessels]
    fixed = [f for f in det["features"] if f["properties"].get("fixed")]
    feats = [f for f in det["features"] if not f["properties"].get("fixed")]
    pairs = match(feats, vessels)
    matched_f = {r: (c, d) for r, c, d in pairs}
    matched_v = {c for _, c, _ in pairs}

    by_det = {}
    for i, f in enumerate(feats):
        p = f["properties"]
        if i in matched_f:
            c, d = matched_f[i]
            by_det[p["id"]] = {"status": "matched", "distance_m": round(d), **vessels[c]}
        else:
            by_det[p["id"]] = {"status": "dark"}
    for f in fixed:
        by_det[f["properties"]["id"]] = {"status": "fixed", "kind": f["properties"]["fixed"]}
    ais_only = [v | {"status": "missed"} for j, v in enumerate(vessels) if j not in matched_v and inside[j]]

    # Validation de nos mesures contre ce que déclarent les navires.
    errs = [feats[r]["properties"]["length_m"] - vessels[c]["length_m"] for r, c, _ in pairs
            if vessels[c].get("length_m") and feats[r]["properties"].get("measured")]
    # Le rappel ne compte que les navires AIS dans la zone demandée.
    big = [v for j, v in enumerate(vessels) if inside[j] and (v.get("length_m") or 0) >= 50]
    big_found = sum(1 for j, v in enumerate(vessels)
                    if inside[j] and j in matched_v and (v.get("length_m") or 0) >= 50)
    n_inside = sum(inside)
    matched_inside = sum(1 for j in matched_v if inside[j])
    summary = {
        "source": "Autorité maritime danoise, AIS ouvert", "day": day, "t0": t0.isoformat(),
        "window_min": WINDOW_MIN, "gate_m": GATE_M,
        "ais_vessels": n_inside, "detections": len(feats), "fixed": len(fixed), "matched": len(pairs),
        "dark": len(feats) - len(pairs), "missed": len(ais_only),
        "recall_ais": round(matched_inside / n_inside, 3) if n_inside else None,
        "recall_ais_50m": round(big_found / len(big), 3) if big else None,
        "ais_50m": len(big),
        "length_error_m": {"n": len(errs), "median_abs": round(float(np.median(np.abs(errs))), 1) if errs else None,
                           "bias": round(float(np.median(errs)), 1) if errs else None},
        "median_distance_m": round(float(np.median([d for _, _, d in pairs])), 0) if pairs else None,
    }
    result = {"summary": summary, "detections": by_det, "ais_only": ais_only}
    (scan.scan_dir(scan_id) / "ais.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    log(f"AIS : {len(vessels)} navires, {len(pairs)} appariés, {summary['dark']} sombres, {len(ais_only)} manqués")
    return result
