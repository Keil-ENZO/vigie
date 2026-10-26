"""Tuiles négatives « dures » : mer ouverte danoise, sans navire AIS à proximité.

Le modèle entraîné sur la Finlande déclenche sur le moindre pixel clair en mer
ouverte (écume, bruit, éoliennes). On lui montre donc des tuiles d'eau L2A, lues
comme le pipeline, où l'AIS ne signale aucun navire à moins de 500 m de la tuile.

Biais assumé : un petit bateau sans AIS dans une de ces tuiles devient un exemple
négatif. On le paie en rappel sur la petite plaisance, qui n'est pas la cible.

Zones et dates disjointes des zones AIS de val et de test (scripts/train/eval_ais.py).

Usage : uv run python scripts/train/prepare_ais_negatives.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ais import AIS_DIR, ZONE_SETS, get_scene, load_window, vessels_at

from vigie import sentinel

TILE = 640
MARGIN_PX = 50  # 500 m autour de la tuile
MIN_WATER = 0.6

ZONES = [
    {"id": "mer_du_nord_ouest", "scene": "S2B_32VMH_20261018_0_L2A", "bbox": [7.60, 56.00, 8.10, 56.40]},
    {"id": "skagerrak_sud", "scene": "S2B_32VMJ_20261018_0_L2A", "bbox": [7.60, 57.00, 8.20, 57.40]},
    {"id": "kattegat_anholt", "scene": "S2B_32VPH_20261018_0_L2A", "bbox": [11.20, 56.20, 11.70, 56.60]},
    {"id": "kadetrinne", "scene": "S2C_33UUA_20260927_0_L2A", "bbox": [12.20, 54.45, 12.70, 54.80]},
    {"id": "skagerrak_est", "scene": "S2C_32VPK_20260930_0_L2A", "bbox": [11.00, 57.90, 11.50, 58.25]},
]


def overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/datasets/ais_negatives"))
    args = ap.parse_args()
    for z in ZONES:  # garde-fou : aucune zone de val ou de test
        for zz in ZONE_SETS["val"] + ZONE_SETS["test"]:
            if overlaps(z["bbox"], zz["bbox"]):
                raise SystemExit(f"{z['id']} recoupe {zz['id']}")
    img_dir, lbl_dir = args.out / "images" / "train", args.out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for z in ZONES:
        sc = get_scene(z["scene"])
        t0 = datetime.fromisoformat(sc.datetime).astimezone(UTC)
        w, s, e, n = z["bbox"]
        pad = 0.02
        big = (w - pad, s - pad, e + pad, n + pad)
        vessels = vessels_at(load_window(AIS_DIR / f"aisdk-{t0:%Y-%m-%d}.csv", big, t0), big, t0)
        r = sentinel.load(sc, tuple(z["bbox"]))
        to_px = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True)
        inv = ~r.transform
        pts = np.array([inv * to_px.transform(v["lon"], v["lat"]) for v in vessels]).reshape(-1, 2)
        h, wd = r.rgb.shape[:2]
        kept = 0
        for y0 in range(0, h - TILE + 1, TILE):
            for x0 in range(0, wd - TILE + 1, TILE):
                crop = r.rgb[y0:y0 + TILE, x0:x0 + TILE]
                if (crop.max(axis=2) == 0).mean() > 0.01:
                    continue
                if r.water is None or r.water[y0:y0 + TILE, x0:x0 + TILE].mean() < MIN_WATER:
                    continue
                near = ((pts[:, 0] > x0 - MARGIN_PX) & (pts[:, 0] < x0 + TILE + MARGIN_PX)
                        & (pts[:, 1] > y0 - MARGIN_PX) & (pts[:, 1] < y0 + TILE + MARGIN_PX))
                if near.any():
                    continue
                name = f"neg_{z['id']}_{y0:05d}_{x0:05d}"
                cv2.imwrite(str(img_dir / f"{name}.png"), np.ascontiguousarray(crop[:, :, ::-1]))
                (lbl_dir / f"{name}.txt").write_text("")
                kept += 1
        stats.append({"zone": z["id"], "scene": z["scene"], "ais": len(vessels), "size_px": [wd, h],
                      "tiles": kept})
        print(stats[-1], flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"zones": ZONES, "stats": stats,
                                                         "margin_px": MARGIN_PX, "min_water": MIN_WATER}, indent=2))


if __name__ == "__main__":
    main()
