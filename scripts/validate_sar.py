"""Validation AIS du radar Sentinel-1 sur des passes danoises, comparée à l'optique.

    uv run python scripts/validate_sar.py [--figure]

Chaque ligne : zone, boîte, passe. Le script lance `scan_sar.run` puis
`ais.crosscheck`, le chemin réel de l'application, et écrit docs/validation-sar.md
et docs/validation-sar.json. Les cinq premières passes tombent le jour même des
cinq scènes optiques de docs/validation-ais.md ; les deux dernières ont servi à
régler les seuils du CFAR et sont comptées à part.

En plus des chiffres de l'optique, on sépare le rappel en eau libre, AIS à plus de
50 % d'eau autour de lui selon le masque, du rappel brut : le masque écarte les
bassins portuaires exprès. On mesure aussi l'écart de position des navires en route,
décalés en azimut par l'effet Doppler.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

from vigie import ais, landmask, sar, scan, scan_sar

# zone, boîte, jour de la passe, préfixe de l'identifiant, réglage
CASES = [
    ("Skagen, mouillage", (10.50, 57.55, 10.80, 57.72), "S1C_IW_GRDH_1SDV_20261018T1701", False),
    ("Aarhus, baie", (10.18, 56.08, 10.40, 56.20), "S1C_IW_GRDH_1SDV_20261018T1700", False),
    ("Fredericia, Petit Belt", (9.65, 55.50, 9.85, 55.62), "S1C_IW_GRDH_1SDV_20261018T1700", False),
    ("Øresund, Copenhague", (12.55, 55.58, 12.80, 55.75), "S1D_IW_GRDH_1SDV_20260927T0523", False),
    ("Grand Belt, pont", (10.85, 55.25, 11.10, 55.40), "S1D_IW_GRDH_1SDV_20260930T1700", False),
    ("Skagen, mouillage", (10.50, 57.55, 10.80, 57.72), "S1D_IW_GRDH_1SDV_20260930T1701", True),
    ("Øresund, Copenhague", (12.55, 55.58, 12.80, 55.75), "S1D_IW_GRDH_1SDV_20260930T1700", True),
]
OPTICAL = Path("docs/validation-ais.json")


def find_scene(bbox, prefix: str) -> sar.SarScene:
    day = date(int(prefix[17:21]), int(prefix[21:23]), int(prefix[23:25]))
    for s in sar.search(bbox, days=1, until=day, limit=50):
        if s.id.startswith(prefix):
            return s
    raise LookupError(f"passe {prefix} introuvable")


def open_water(meta: dict, vessels: list[dict]) -> list[bool]:
    """Pour chaque AIS : la mer domine-t-elle autour de lui, selon le masque du scan ?"""
    transform, (w, h) = rasterio.Affine(*meta["transform"]), meta["size_px"]
    water = landmask.water_mask(meta["bbox"], transform, meta["crs"], (h, w), log=lambda m: None)
    to = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    inv = ~transform
    out = []
    for v in vessels:
        c, r = inv @ to.transform(v["lon"], v["lat"])
        c, r = int(c), int(r)
        patch = water[max(0, r - 3) : r + 4, max(0, c - 3) : c + 4]
        out.append(bool(patch.size) and float(patch.mean()) >= 0.5)
    return out


def row_for(name, bbox, prefix, tuning) -> dict:
    sc = find_scene(bbox, prefix)
    ais.ensure_day(sc.datetime[:10])
    meta = scan_sar.run(bbox, scene=sc, name=name)
    res = ais.crosscheck(meta["id"])
    s = res["summary"]
    _, det = scan.load_scan(meta["id"])
    feats = [f["properties"] for f in det["features"] if not f["properties"].get("fixed")]
    matched = [p for p in feats if p.get("ais", {}).get("status") == "matched"]
    errs = [p["length_m"] - p["ais"]["length_m"] for p in matched if p["ais"].get("length_m")]
    # Tous les AIS de la zone : appariés, plus manqués.
    vessels = [p["ais"] for p in matched] + res["ais_only"]
    ow = open_water(meta, vessels)
    big = [(v, o) for v, o in zip(vessels, ow) if (v.get("length_m") or 0) >= 50]
    big_ow = [v for v, o in big if o]
    moving = [p["ais"]["distance_m"] for p in matched if (p["ais"].get("sog_kn") or 0) >= 3]
    still = [p["ais"]["distance_m"] for p in matched if (p["ais"].get("sog_kn") or 0) < 0.5]
    return {
        "zone": name, "scene": sc.id, "datetime": sc.datetime, "orbit": sc.orbit_state, "tuning": tuning,
        "scan_id": meta["id"], "ais": s["ais_vessels"], "ais_50m": s["ais_50m"], "detections": s["detections"],
        "fixed": s["fixed"], "matched": s["matched"], "dark": s["dark"],
        "precision": round(s["matched"] / s["detections"], 3) if s["detections"] else None,
        "recall_50m": s["recall_ais_50m"], "matched_50m": sum(1 for v, _ in big if v["status"] == "matched"),
        "ais_50m_open": len(big_ow), "matched_50m_open": sum(1 for v in big_ow if v["status"] == "matched"),
        "len_mae": s["length_error_m"]["median_abs"], "len_bias": s["length_error_m"]["bias"], "len_errs": errs,
        "dist_moving_m": moving, "dist_still_m": still,
        "timing_s": meta["timing_s"],
    }


def total(rows: list[dict]) -> dict:
    keys = ("ais", "ais_50m", "detections", "fixed", "matched", "dark", "matched_50m", "ais_50m_open",
            "matched_50m_open")
    t = {k: sum(r[k] or 0 for r in rows) for k in keys}
    errs = [e for r in rows for e in r["len_errs"]]
    t["precision"] = round(t["matched"] / t["detections"], 3) if t["detections"] else None
    t["recall_50m"] = round(t["matched_50m"] / t["ais_50m"], 3) if t["ais_50m"] else None
    t["recall_50m_open"] = round(t["matched_50m_open"] / t["ais_50m_open"], 3) if t["ais_50m_open"] else None
    t["len_mae"] = round(float(np.median(np.abs(errs))), 1) if errs else None
    t["len_bias"] = round(float(np.median(errs)), 1) if errs else None
    mv = [d for r in rows for d in r["dist_moving_m"]]
    st = [d for r in rows for d in r["dist_still_m"]]
    t["dist_moving_median_m"] = round(float(np.median(mv))) if mv else None
    t["dist_still_median_m"] = round(float(np.median(st))) if st else None
    t["n_moving"], t["n_still"] = len(mv), len(st)
    return t


def pct(v):
    return "–" if v is None else f"{v * 100:.0f} %"


def table(rows, tot, optical) -> list[str]:
    lines = [("| Zone | Passe S1, UTC | AIS | AIS > 50 m | Détections | Appariées | Sombres | Précision | "
             "Rappel > 50 m | Rappel > 50 m eau libre | Longueur, err. méd. |"),
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['zone']} | {r['datetime'][:16].replace('T', ' ')} | {r['ais']} | {r['ais_50m']} | "
                     f"{r['detections']} | {r['matched']} | {r['dark']} | {pct(r['precision'])} | {pct(r['recall_50m'])}"
                     f" | {r['matched_50m_open']}/{r['ais_50m_open']} | {r['len_mae']} m |")
    lines.append(f"| **Total** | | {tot['ais']} | {tot['ais_50m']} | {tot['detections']} | {tot['matched']} | "
                 f"{tot['dark']} | **{pct(tot['precision'])}** | **{pct(tot['recall_50m'])}** | "
                 f"**{pct(tot['recall_50m_open'])}** | {tot['len_mae']} m, biais {tot['len_bias']} m |")
    if optical:
        lines.append(f"| *Optique S2, mêmes zones et jours* | | {optical['ais']} | {optical['ais_50m']} | "
                     f"{optical['detections']} | {optical['matched']} | {optical['dark']} | {pct(optical['precision'])}"
                     f" | {pct(optical['recall_50m'])} | – | {optical['len_mae']} m |")
    return lines


def figure(row: dict, out: Path, side: int = 640) -> None:
    """Le mouillage de Skagen en VV, niveaux de gris. Détections en rouge, AIS > 50 m en jaune,
    autres AIS en bleu."""
    from PIL import Image, ImageDraw

    d = scan.scan_dir(row["scan_id"])
    meta, det = scan.load_scan(row["scan_id"])
    a = json.loads((d / "ais.json").read_text())
    vv = Image.open(d / "image.jpg").getchannel("R")
    inv = ~rasterio.Affine(*meta["transform"])
    to = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    px = lambda lon, lat: inv @ to.transform(lon, lat)
    feats = [f["properties"] for f in det["features"]]
    pts = [px(p["lon"], p["lat"]) for p in feats]
    # Cadre carré autour des détections, 60 px de marge.
    x0, y0 = min(x for x, _ in pts) - 60, min(y for _, y in pts) - 60
    size = max(max(x for x, _ in pts) + 60 - x0, max(y for _, y in pts) + 60 - y0)
    x0, y0 = max(0, x0), max(0, y0)
    scale = side / size
    im = vv.crop((int(x0), int(y0), int(x0 + size), int(y0 + size))).resize((side, side)).convert("RGB")
    draw = ImageDraw.Draw(im)

    def at(x, y):
        return (x - x0) * scale, (y - y0) * scale

    vessels = [p["ais"] for p in feats if p.get("ais", {}).get("lat")]
    for v in vessels + a["ais_only"]:
        x, y = at(*px(v["lon"], v["lat"]))
        c = (255, 220, 0) if (v.get("length_m") or 0) >= 50 else (0, 170, 255)
        draw.rectangle([x - 5, y - 5, x + 5, y + 5], outline=c, width=2)
    for x, y in pts:
        x, y = at(x, y)
        draw.ellipse([x - 11, y - 11, x + 11, y + 11], outline=(255, 40, 40), width=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.quantize(64).save(out, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/validation-sar"))
    ap.add_argument("--figure", action="store_true", help="écrit docs/img/sar-skagen.png")
    args = ap.parse_args()
    rows = []
    for name, bbox, prefix, tuning in CASES:
        try:
            rows.append(row_for(name, bbox, prefix, tuning))
        except LookupError as exc:
            print(f"{name} : {exc}")
            continue
        print({k: v for k, v in rows[-1].items() if not isinstance(v, list)})
    test = [r for r in rows if not r["tuning"]]
    tune = [r for r in rows if r["tuning"]]
    optical = None
    if OPTICAL.exists():
        optical = json.loads(OPTICAL.read_text())["total"]
    t_test, t_tune, t_all = total(test), total(tune), total(rows)
    args.out.parent.mkdir(exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"params": scan.load_scan(rows[0]["scan_id"])[0]["params"], "rows": rows,
         "total_test": t_test, "total_tuning": t_tune, "total_all": t_all}, indent=2, ensure_ascii=False))
    lines = ["# Validation AIS du radar Sentinel-1, CFAR log-normal VV et VH", "",
             "## Passes du même jour que l'optique, non vues au réglage", "", *table(test, t_test, optical), "",
             "## Passes de réglage des seuils", "", *table(tune, t_tune, None), "",
             (f"Écart médian à l'AIS : {t_all['dist_still_median_m']} m à l'arrêt ({t_all['n_still']} navires), "
             f"{t_all['dist_moving_median_m']} m en route à 3 nœuds ou plus ({t_all['n_moving']} navires)."), ""]
    args.out.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))
    if args.figure and rows:
        figure(rows[0], Path("docs/img/sar-skagen.png"))


if __name__ == "__main__":
    main()
