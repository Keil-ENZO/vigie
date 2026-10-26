"""Passage des boîtes en pixels vers des navires géoréférencés."""

from __future__ import annotations

import math

import numpy as np
from pyproj import Transformer

from .morphology import measure
from .tiling import Box


def water_fraction(water: np.ndarray, box: Box, margin: int = 6) -> float:
    """Part d'eau autour de la boîte. Le navire lui-même n'est pas classé eau par SCL,
    on regarde donc un anneau élargi."""
    h, w = water.shape
    x1, y1 = max(0, int(box.x1) - margin), max(0, int(box.y1) - margin)
    x2, y2 = min(w, int(math.ceil(box.x2)) + margin), min(h, int(math.ceil(box.y2)) + margin)
    patch = water[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def to_features(boxes: list[Box], transform, crs: str, gsd: float, water: np.ndarray | None = None,
                min_water: float = 0.25, rgb: np.ndarray | None = None) -> list[dict]:
    """GeoJSON en WGS84. Chaque navire garde sa boîte pixel pour le recadrage."""
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    feats = []
    for i, b in enumerate(boxes, start=1):
        wf = water_fraction(water, b) if water is not None else None
        if wf is not None and wf < min_water:
            continue
        cx, cy = b.center
        corners = [transform * (b.x1, b.y1), transform * (b.x2, b.y1), transform * (b.x2, b.y2),
                   transform * (b.x1, b.y2)]
        ring = [list(to_wgs.transform(x, y)) for x, y in corners]
        ring.append(ring[0])
        lon, lat = to_wgs.transform(*(transform * (cx, cy)))
        shape = measure(rgb, b, gsd) if rgb is not None else None
        if shape is not None:
            length, width, heading = shape.length_m, shape.width_m, shape.heading_deg
            course, wake = shape.course_deg, shape.wake_m
        else:
            # Repli : la diagonale de la boîte, borne haute de la longueur.
            length, width, heading = math.hypot(b.width, b.height) * gsd, None, None
            course = wake = None
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "id": f"V{i:03d}", "confidence": round(float(b.confidence), 3), "label": b.label,
                "lat": round(lat, 6), "lon": round(lon, 6),
                "length_m": round(float(length), 0), "width_m": width, "heading_deg": heading,
                "course_deg": course, "wake_m": wake, "underway": wake is not None,
                "measured": shape is not None, "box_px": [round(float(v), 1) for v in (b.x1, b.y1, b.x2, b.y2)],
                "water_fraction": None if wf is None else round(wf, 2),
            },
        })
    return feats


def renumber(features: list[dict]) -> list[dict]:
    for i, f in enumerate(features, start=1):
        f["properties"]["id"] = f"V{i:03d}"
    return features
