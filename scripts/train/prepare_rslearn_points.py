"""Test hors domaine : fenêtres « sentinel2_vessels » d'Ai2, étiquettes ponctuelles.

Les étiquettes sont des centres de navires, pas des boîtes. On les écrit comme des
carrés de 15 px (150 m) centrés sur le point : seules les métriques « centre »
d'evaluate.py ont un sens sur ce jeu, pas le mAP par IoU.

Images : RGB L1C harmonisé (rslearn), pas le TCI L2A du pipeline. Fenêtres de
300 à 1000 px découpées en tuiles 640 sans chevauchement, complétées de noir.
Annotations Ai2 sous CC BY 4.0.

Usage : uv run python scripts/train/prepare_rslearn_points.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TILE = 640
HALF = 7.5  # demi-côté du carré autour du point, en pixels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/raw/rslearn_s2_vessels"))
    ap.add_argument("--out", type=Path, default=Path("data/datasets/rslearn_points"))
    args = ap.parse_args()
    img_dir, lbl_dir = args.out / "images" / "test", args.out / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_tiles = n_pts = n_win = 0
    for meta_p in sorted(args.src.glob("windows/*/*/metadata.json")):
        win = meta_p.parent
        img_p = win / "layers/sentinel2/R_G_B/image.png"
        lbl_p = win / "layers/label/data.geojson"
        if not img_p.exists() or not lbl_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        x0b, y0b = meta["bounds"][:2]
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        pts = np.array([f["geometry"]["coordinates"][:2] for f in json.loads(lbl_p.read_text())["features"]
                        if f["geometry"]["type"] == "Point"]).reshape(-1, 2) - [x0b, y0b]
        n_win += 1
        for ty in range(0, h, TILE):
            for tx in range(0, w, TILE):
                crop = np.zeros((TILE, TILE, 3), np.uint8)
                sub = img[ty:ty + TILE, tx:tx + TILE]
                if (sub.max(axis=2) == 0).mean() > 0.5:
                    continue
                crop[:sub.shape[0], :sub.shape[1]] = sub
                p = pts - [tx, ty]
                p = p[(p[:, 0] >= 0) & (p[:, 0] < sub.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < sub.shape[0])]
                name = f"{win.parent.name}_{win.name}_{ty}_{tx}"
                cv2.imwrite(str(img_dir / f"{name}.png"), crop)
                (lbl_dir / f"{name}.txt").write_text("".join(
                    f"0 {x / TILE:.6f} {y / TILE:.6f} {2 * HALF / TILE:.6f} {2 * HALF / TILE:.6f}\n" for x, y in p))
                n_tiles += 1
                n_pts += len(p)
    (args.out / "data.yaml").write_text(f"path: {args.out}\ntest: images/test\nnames:\n  0: boat\n")
    summary = {"windows": n_win, "tiles": n_tiles, "points": n_pts,
               "source": "gs://ai2-rslearn-projects-data/sentinel2_vessels/sentinel2_vessels.tar (extrait)",
               "license_annotations": "CC-BY-4.0"}
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
