"""Détecteur de navires par tuiles, autour d'un modèle YOLO."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .thermal import wait_if_hot
from .tiling import Box, make_tiles, merge

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path(os.environ.get("VIGIE_MODEL", ROOT / "models" / "defendintel-yolov8s-s2.pt"))


@dataclass
class DetectParams:
    confidence: float = 0.20
    iou: float = 0.45
    tile: int = 640
    overlap: int = 96
    batch: int = 16
    max_det: int = 500


@dataclass
class DetectResult:
    boxes: list[Box]
    tiles: int
    seconds: float
    model: str


@lru_cache(maxsize=4)
def load_model(path: str):
    from ultralytics import YOLO

    if not Path(path).exists():
        raise FileNotFoundError(f"modèle introuvable : {path}. Lance `vigie fetch-model`.")
    return YOLO(path)


def device() -> str | int:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def detect(image: np.ndarray, params: DetectParams | None = None, model_path: Path | None = None) -> DetectResult:
    """`image` en RGB, HxWx3, uint8. Rend les boîtes en pixels de l'image entière."""
    p = params or DetectParams()
    # Lu à l'appel, pas à l'import : `validate_ais.py --model` change DEFAULT_MODEL.
    model_path = model_path or DEFAULT_MODEL
    model = load_model(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    h, w = image.shape[:2]
    tiles = make_tiles(w, h, p.tile, p.overlap)
    started = time.perf_counter()
    found: list[Box] = []
    for i in range(0, len(tiles), p.batch):
        wait_if_hot()
        chunk = tiles[i : i + p.batch]
        crops = [np.ascontiguousarray(image[t.y : t.y + t.height, t.x : t.x + t.width, ::-1]) for t in chunk]
        results = model.predict(crops, conf=p.confidence, iou=p.iou, imgsz=p.tile,
                                max_det=p.max_det, device=device(), verbose=False)
        for res, t in zip(results, chunk):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for (x1, y1, x2, y2), conf, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                                   res.boxes.cls.cpu().numpy().astype(int)):
                found.append(Box(x1 + t.x, y1 + t.y, x2 + t.x, y2 + t.y, float(conf), int(cls),
                                 str(names.get(int(cls), cls))))
    boxes = merge(found, p.iou)[: p.max_det]
    return DetectResult(boxes, len(tiles), time.perf_counter() - started, Path(model_path).name)
