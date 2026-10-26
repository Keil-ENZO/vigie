"""Fine-tune YOLO sur le jeu préparé, avec la garde thermique et un plafond VRAM.

Usage :
  nohup nice -n 10 uv run python scripts/train/train.py --name y11s_coco --model yolo11s.pt \
      > runs/train_y11s_coco.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vigie.training import ThermalGuard

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/datasets/finnish/data.yaml"))
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--freeze", type=int, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    project = ROOT / "runs" / "train"
    guard = ThermalGuard(log_path=project / f"{args.name}_thermal.log")
    model = YOLO(args.model)
    guard.attach(model)
    started = time.time()
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=640, batch=args.batch, workers=min(args.workers, 6),
        device=0, project=str(project), name=args.name, exist_ok=True, seed=0, deterministic=False,
        optimizer="AdamW", lr0=args.lr0, lrf=0.05, cos_lr=True, warmup_epochs=2, patience=args.patience,
        single_cls=True, cache="ram", amp=True, plots=True, freeze=args.freeze,
        # Augmentations : la vue est zénithale, toutes les orientations se valent.
        # Les rotations à 90° sont faites hors ligne (tuiles tournées dans le train).
        flipud=0.5, fliplr=0.5, degrees=0.0, translate=0.1, scale=args.scale, shear=0.0, perspective=0.0,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3, mosaic=args.mosaic, close_mosaic=args.close_mosaic,
        mixup=0.0, copy_paste=0.0, erasing=0.0,
    )
    summary = {"name": args.name, "args": vars(args) | {"data": str(args.data)},
               "wall_s": round(time.time() - started), "thermal": guard.summary()}
    (project / args.name / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
