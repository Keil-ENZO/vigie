"""Découpage en tuiles et fusion des détections.

Sentinel-2 est à 10 m par pixel : un cargo fait 10 à 30 pixels. On découpe donc
l'image en tuiles qui se chevauchent, pour que le modèle voie les navires à leur
taille d'entraînement, puis on fusionne les boîtes en double aux jonctions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    label: str = "vessel"
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def shifted(self, dx: float, dy: float) -> "Box":
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy,
                   self.confidence, self.class_id, self.label, dict(self.extra))


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Positions de départ qui couvrent `length`, la dernière tuile collée au bord."""
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def make_tiles(width: int, height: int, tile: int = 640, overlap: int = 96) -> list[Tile]:
    if overlap >= tile:
        raise ValueError("le chevauchement doit être plus petit que la tuile")
    tiles = []
    for y in tile_starts(height, tile, overlap):
        for x in tile_starts(width, tile, overlap):
            tiles.append(Tile(x, y, min(tile, width - x), min(tile, height - y)))
    return tiles


def iou(a: Box, b: Box) -> float:
    iw = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    ih = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def ios(a: Box, b: Box) -> float:
    """Intersection sur la plus petite boîte : attrape un navire coupé en deux par une jonction."""
    iw = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    ih = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    smallest = min(a.area, b.area)
    return (iw * ih) / smallest if smallest > 0 else 0.0


def merge(boxes: list[Box], iou_threshold: float = 0.45, ios_threshold: float = 0.8) -> list[Box]:
    """NMS par classe, plus suppression des boîtes presque incluses dans une meilleure."""
    remaining = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[Box] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            b for b in remaining
            if b.class_id != best.class_id
            or (iou(b, best) < iou_threshold and ios(b, best) < ios_threshold)
        ]
    return kept
