"""Longueur et cap d'un navire, mesurés sur ses pixels.

La boîte YOLO n'est pas orientée : sa diagonale surestime la longueur d'un navire
de biais. On isole donc les pixels plus clairs que l'eau autour, puis on prend
l'axe principal de leur nuage (ACP). Pour un rectangle plein, la longueur vaut
sqrt(12 * λ1), λ1 étant la plus grande valeur propre de la covariance.

Un navire en route traîne un sillage clair, souvent trois fois plus long que lui.
On sépare donc la signature entière, coque et sillage, de la coque seule : les
pixels proches du pic de luminosité. La longueur vient de la coque. Le sillage,
lui, donne le sens de marche : la coque est à l'avant de la signature.

Limite mesurée contre l'AIS à Skagen, le 2026-10-18 : sur 23 navires, 1 vrai
sillage trouvé et 6 faux. Un cargo au mouillage avec sa passerelle blanche à
l'arrière ressemble, à 10 m par pixel, à une coque suivie d'un sillage. On parle
donc de « sillage probable ». Quand l'AIS est là, sa vitesse fait foi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .tiling import Box


@dataclass
class Shape:
    length_m: float
    width_m: float
    heading_deg: float  # axe du navire, 0 = nord, sens horaire, ambigu à 180° près
    pixels: int
    course_deg: float | None = None  # sens de marche, levé par le sillage
    wake_m: float | None = None


def _axis(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float, float]:
    """Longueur et largeur en pixels, angle de l'axe depuis le nord, centre x, centre y."""
    pts = np.stack([xs, -ys], axis=1).astype(np.float64)  # y vers le nord
    vals, vecs = np.linalg.eigh(np.cov(pts, rowvar=False))
    vx, vy = vecs[:, 1]
    angle = (math.degrees(math.atan2(vx, vy)) + 360) % 180
    # +1 pixel : la moitié de pixel perdue à chaque extrémité par le seuillage.
    return (math.sqrt(12 * vals[1]) + 1, math.sqrt(12 * max(vals[0], 0.0)) + 1, angle,
            float(xs.mean()), float(ys.mean()))


def measure(rgb: np.ndarray, box: Box, gsd: float, pad: int = 4, k: float = 2.5) -> Shape | None:
    h, w = rgb.shape[:2]
    x1, y1 = max(0, int(box.x1) - pad), max(0, int(box.y1) - pad)
    x2, y2 = min(w, int(math.ceil(box.x2)) + pad), min(h, int(math.ceil(box.y2)) + pad)
    crop = rgb[y1:y2, x1:x2].astype(np.float32)
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return None
    lum = crop.mean(axis=2)
    # Le fond de référence : le bord de la fenêtre, presque toujours de l'eau.
    border = np.concatenate([lum[0], lum[-1], lum[:, 0], lum[:, -1]])
    med = float(np.median(border))
    mad = float(np.median(np.abs(border - med))) * 1.4826 + 1.0
    mask = np.abs(lum - med) > k * mad
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    cy, cx = (lum.shape[0] - 1) / 2, (lum.shape[1] - 1) / 2
    # On garde la composante la plus grande, à égalité la plus proche du centre.
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    centers = ndimage.center_of_mass(mask, labels, range(1, n + 1))
    best = max(range(n), key=lambda i: (sizes[i], -math.hypot(centers[i][0] - cy, centers[i][1] - cx)))
    comp = labels == best + 1
    ys, xs = np.nonzero(comp)
    if len(xs) < 3:
        return None
    sig_len, _, sig_angle, sx, sy = _axis(xs, ys)
    peak = float(lum[comp].max())
    hull = comp & (np.abs(lum - med) >= 0.5 * abs(peak - med))
    hy, hx = np.nonzero(hull)
    if len(hx) < 2:
        hx, hy = xs, ys
    if len(hx) >= 3:
        length, width, angle, hcx, hcy = _axis(hx, hy)
    else:
        length, width, angle, hcx, hcy = 2.0, 1.0, sig_angle, float(hx.mean()), float(hy.mean())
    course = wake = None
    if sig_len >= 1.8 * length and sig_len * gsd >= 60:
        # Sillage : le sens de marche va du centre de la signature vers la coque.
        dx, dy = hcx - sx, -(hcy - sy)
        if math.hypot(dx, dy) >= 0.5:
            course = round((math.degrees(math.atan2(dx, dy)) + 360) % 360, 0)
            wake = round((sig_len - length) * gsd, 0)
            angle = sig_angle  # l'axe du sillage est plus sûr que celui d'une coque de 3 pixels
    return Shape(round(length * gsd, 0), round(width * gsd, 0), round(angle, 0), int(len(xs)), course, wake)
