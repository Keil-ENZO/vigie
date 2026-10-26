"""Figure AIS : détections de deux modèles (rouge, seuil 0,20, filtre eau) et positions
AIS à l'heure de la passe (jaune, croix large si > 50 m), sur des recadrages de zones.

Usage : uv run python scripts/train/figure_ais.py --weights A.pt B.pt --labels ref v1 \
            --zone oresund:400:700:500 --zone fehmarnbelt:900:900:500 --out docs/img/ais.png
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_images import render
from eval_ais import AIS_DIR, ZONE_SETS, detect, get_scene, load_window, vessels_at
from PIL import Image

from vigie import sentinel
from vigie.geo import to_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--zone", action="append", required=True, help="id:x0:y0:côté en pixels")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    models = [YOLO(str(w)) for w in args.weights]
    zones = {z["id"]: z for zs in ZONE_SETS.values() for z in zs}
    rows = []
    for spec in args.zone:
        zid, x0, y0, side = spec.split(":")
        x0, y0, side = int(x0), int(y0), int(side)
        z = zones[zid]
        sc = get_scene(z["scene"])
        t0 = datetime.fromisoformat(sc.datetime).astimezone(UTC)
        vessels = vessels_at(load_window(AIS_DIR / f"aisdk-{t0:%Y-%m-%d}.csv", z["bbox"], t0), z["bbox"], t0)
        r = sentinel.load(sc, tuple(z["bbox"]))
        to_px = Transformer.from_crs("EPSG:4326", r.crs, always_xy=True)
        inv = ~r.transform
        ais_px = [(inv * to_px.transform(v["lon"], v["lat"]), (v["length_m"] or 0) >= 50) for v in vessels]
        sub = np.ascontiguousarray(r.rgb[y0:y0 + side, x0:x0 + side, ::-1])
        base = cv2.resize(sub, (side * args.scale, side * args.scale), interpolation=cv2.INTER_NEAREST)
        cells = [(base, [("title", f"{zid} : {len(vessels)} AIS")], ais_px, (x0, y0))]
        for m, lab in zip(models, args.labels):
            feats = to_features(detect(m, r.rgb, 0, conf=args.conf), r.transform, r.crs, r.gsd, r.water)
            boxes = np.array([f["properties"]["box_px"] for f in feats]).reshape(-1, 4)
            inside = ((boxes[:, 2] > x0) & (boxes[:, 0] < x0 + side) & (boxes[:, 3] > y0) & (boxes[:, 1] < y0 + side))
            marks = [("box", (boxes - [x0, y0, x0, y0]) * args.scale + [-2, -2, 2, 2], (0, 0, 255), 1),
                     ("title", f"{lab} : {int(inside.sum())} det.")]
            cells.append((base.copy(), marks, ais_px, (x0, y0)))
        rows.append(cells)
    ph, pw = rows[0][0][0].shape[:2]
    gap = 6
    canvas = np.full((len(rows) * (ph + gap) - gap, len(rows[0]) * (pw + gap) - gap, 3), 255, np.uint8)
    for ri, cells in enumerate(rows):
        for ci, (base, _, _, _) in enumerate(cells):
            canvas[ri * (ph + gap):ri * (ph + gap) + ph, ci * (pw + gap):ci * (pw + gap) + pw] = base
    q = Image.fromarray(canvas[:, :, ::-1]).quantize(colors=248, method=Image.Quantize.MEDIANCUT)
    canvas = np.ascontiguousarray(np.array(q.convert("RGB"))[:, :, ::-1])
    for ri, cells in enumerate(rows):
        for ci, (_, marks, ais_px, (x0, y0)) in enumerate(cells):
            ox, oy = ci * (pw + gap), ri * (ph + gap)
            for (px, py), big in ais_px:
                cx, cy = int((px - x0) * args.scale), int((py - y0) * args.scale)
                if 0 <= cx < pw and 0 <= cy < ph:
                    cv2.drawMarker(canvas, (ox + cx, oy + cy), (0, 255, 255), cv2.MARKER_CROSS,
                                   14 if big else 7, 1, cv2.LINE_8)
            render(canvas, ox, oy, pw, ph, marks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas[:, :, ::-1]).quantize(colors=256, dither=Image.Dither.NONE).save(args.out, optimize=True)
    print(args.out, args.out.stat().st_size // 1024, "Ko")


if __name__ == "__main__":
    main()
