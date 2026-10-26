"""Figures côte à côte : vérité terrain et prédictions de deux modèles sur les mêmes tuiles.

Lit les npz écrits par evaluate.py. Vert : vérité. Rouge : prédiction au seuil donné.
Les tuiles sont agrandies x2 au plus proche voisin pour que les petits navires se voient.

Usage :
  uv run python scripts/train/compare_images.py --data .../data.yaml --split test \
      --preds runs/eval/final/preds_A_test.npz runs/eval/final/preds_B_test.npz \
      --labels "référence" "vigie-s2-v1" --tiles NOM1.png NOM2.png --out docs/img/cmp.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_preds(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {str(n): z[f"p{i}"] for i, n in enumerate(z["names"])}


def panel(img_path: Path, gt: np.ndarray, pred: np.ndarray | None, conf: float, crop: tuple | None,
          scale: int, title: str) -> tuple[np.ndarray, list]:
    """Rend l'image agrandie et la liste des tracés à faire après quantification."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    x0, y0, cw, ch = crop or (0, 0, w, h)
    big = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (cw * scale, ch * scale), interpolation=cv2.INTER_NEAREST)
    off = np.array([x0, y0, x0, y0])
    marks = [("box", (gt - off) * scale, (0, 220, 0), 1)]
    if pred is not None:
        p = pred[pred[:, 4] >= conf]
        marks.append(("box", (p[:, :4] - off) * scale, (0, 0, 255), 2))
        title = f"{title} : {len(p)} det."
    marks.append(("title", title))
    return big, marks


def render(canvas: np.ndarray, x: int, y: int, w: int, h: int, marks: list) -> None:
    """Trace sur la figure entière, décalée de (x, y), découpé au panneau.
    Pas d'anticrénelage : les couleurs restent exactes dans la palette."""
    for m in marks:
        if m[0] == "box":
            for b in m[1]:
                x1, y1, x2, y2 = (round(float(v)) for v in b[:4])
                if x2 < 0 or y2 < 0 or x1 >= w or y1 >= h:
                    continue
                x1, x2 = np.clip([x1, x2], 0, w - 1)
                y1, y2 = np.clip([y1, y2], 0, h - 1)
                cv2.rectangle(canvas, (x + int(x1), y + int(y1)), (x + int(x2), y + int(y2)), m[2], m[3])
        else:
            cv2.rectangle(canvas, (x, y), (x + w - 1, y + 26), (0, 0, 0), -1)
            cv2.putText(canvas, m[1], (x + 6, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--preds", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--crop", type=int, default=320, help="côté du recadrage centré sur les navires, 0 = tuile")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.data.read_text())
    img_dir = Path(cfg["path"]) / cfg[args.split]
    preds = [load_preds(p) for p in args.preds]
    rows = []
    for name in args.tiles:
        ip = img_dir / name
        lbl = Path(str(img_dir).replace("/images/", "/labels/")) / f"{Path(name).stem}.txt"
        g = np.loadtxt(lbl, ndmin=2) if lbl.stat().st_size else np.zeros((0, 5))
        w = h = 640
        gt = np.stack([(g[:, 1] - g[:, 3] / 2) * w, (g[:, 2] - g[:, 4] / 2) * h,
                       (g[:, 1] + g[:, 3] / 2) * w, (g[:, 2] + g[:, 4] / 2) * h], axis=1) if len(g) else np.zeros((0, 4))
        crop = None
        if args.crop:
            # Centre du recadrage : le navire qui a le plus de voisins proches.
            if len(gt):
                ctr = np.stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2], axis=1)
                d = np.hypot(*(ctr[:, None, :] - ctr[None, :, :]).transpose(2, 0, 1))
                cx, cy = ctr[int(np.argmax((d < args.crop / 2).sum(axis=1)))]
            else:
                cx, cy = w / 2, h / 2
            x0 = int(np.clip(cx - args.crop / 2, 0, w - args.crop))
            y0 = int(np.clip(cy - args.crop / 2, 0, h - args.crop))
            crop = (x0, y0, args.crop, args.crop)
        cells = [panel(ip, gt, None, args.conf, crop, args.scale, f"verite : {len(gt)}")]
        cells += [panel(ip, gt, p[name], args.conf, crop, args.scale, lab) for p, lab in zip(preds, args.labels)]
        rows.append(cells)
    from PIL import Image

    # Une seule palette pour toute la figure : on quantifie les fonds ensemble,
    # puis on trace les boîtes en couleurs pures, qui restent exactes dans le PNG.
    ph, pw = rows[0][0][0].shape[:2]
    gap = 6
    canvas = np.full((len(rows) * (ph + gap) - gap, len(rows[0]) * (pw + gap) - gap, 3), 255, np.uint8)
    for r, cells in enumerate(rows):
        for c, (base, _) in enumerate(cells):
            canvas[r * (ph + gap):r * (ph + gap) + ph, c * (pw + gap):c * (pw + gap) + pw] = base
    q = Image.fromarray(canvas[:, :, ::-1]).quantize(colors=250, method=Image.Quantize.MEDIANCUT)
    canvas = np.ascontiguousarray(np.array(q.convert("RGB"))[:, :, ::-1])
    for r, cells in enumerate(rows):
        for c, (_, marks) in enumerate(cells):
            render(canvas, c * (pw + gap), r * (ph + gap), pw, ph, marks)
    out = canvas
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # PNG en palette pour rester sous 500 Ko.
    rgb = out[:, :, ::-1]
    colors = np.unique(rgb.reshape(-1, 3), axis=0)
    if len(colors) <= 256:
        pal = Image.new("P", (1, 1))
        pal.putpalette(colors.flatten().tolist() + [0] * (768 - colors.size))
        img = Image.fromarray(rgb).quantize(palette=pal, dither=Image.Dither.NONE)
    else:
        img = Image.fromarray(rgb).quantize(colors=256, dither=Image.Dither.NONE)
    img.save(args.out, optimize=True)
    print(args.out, args.out.stat().st_size // 1024, "Ko")


if __name__ == "__main__":
    main()
