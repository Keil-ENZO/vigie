"""Détection de navires sur radar : CFAR, sans apprentissage.

Un navire en acier renvoie l'onde radar comme un coin de réflecteur : il ressort de
10 à 40 dB au-dessus de la mer. Le fond de mer, lui, varie avec le vent et l'angle
d'incidence. On compare donc chaque pixel à son voisinage, pas à un seuil fixe.

Fenêtres, centrées sur le pixel testé :

    +-------------------------+
    |   fond : la mer locale  |
    |    +---------------+    |
    |    | garde : le    |    |
    |    | navire entier |    |
    |    +---------------+    |
    +-------------------------+

Modèle : log-normal. En dB, le fouillis de mer suit à peu près une gaussienne, de
moyenne μ et d'écart-type σ mesurés dans l'anneau de fond. Un pixel est candidat si
x_dB > μ + k·σ et x_dB − μ > un contraste minimal. Le modèle K colle mieux aux
queues de distribution par mer forte, mais demande d'estimer un paramètre de forme
par fenêtre ; le log-normal se calcule en trois filtres moyenneurs, et l'écart en
dB garde-fou rattrape ses fausses alarmes. Le fond est estimé deux fois : la seconde
passe exclut les cibles de la première, pour qu'un navire voisin ne gonfle pas σ.

On teste VV et VH. En VH, la mer tombe presque au bruit du capteur et un navire
garde un fort contraste ; en VV, la mer est plus claire mais les petits navires y
ressortent mieux par mer calme. Un pixel est candidat s'il l'est dans l'une des deux.

Les composantes connexes donnent les navires. La longueur et l'axe viennent d'une
ACP, comme `morphology.measure`, mais sur les seuls pixels à moins de `hull_db` du
pic : un réflecteur fort dessine une croix de lobes secondaires le long des axes de
l'image, qui allongerait la coque de plusieurs centaines de mètres.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .morphology import _axis
from .tiling import Box

MODEL = "cfar-lognormal-vv-vh"


@dataclass
class CfarParams:
    guard: int = 15  # demi-côté de la fenêtre de garde, en pixels : 310 m de côté
    background: int = 40  # demi-côté de la fenêtre de fond : 810 m de côté
    k: float = 4.5  # seuil en écarts-types du fond, en dB
    min_contrast_db: float = 8.0  # écart minimal au fond, en dB
    min_peak_contrast_db: float = 13.0  # contraste du pixel le plus brillant d'un navire
    min_pixels: int = 2
    max_length_m: float = 450.0  # le plus long navire en service fait 400 m
    hull_db: float = 12.0  # pixels gardés pour la mesure : à moins de 12 dB du pic
    # La réponse impulsionnelle du GRD IW fait 20 m sur 22 : une coque paraît plus
    # longue d'environ une cellule de résolution. Biais mesuré : +18 à +32 m.
    psf_m: float = 20.0
    use_vh: bool = True
    min_valid: float = 0.3  # part minimale de mer dans l'anneau de fond
    merge_px: int = 1  # dilatation avant regroupement, recolle une coque morcelée
    big_length_m: float = 150.0  # au-delà, un vrai navire est une montagne d'acier :
    big_min_contrast_db: float = 20.0  # il doit ressortir d'au moins 20 dB
    # Même règle en VH, plus stricte : la mer y est au plancher de bruit du capteur, et
    # les cibles VH seules sont surtout du bruit (Grand Belt, Øresund).
    vh_min_peak_contrast_db: float = 18.0
    vh_measure_db: float = 15.0  # contraste VH au-delà duquel la coque se mesure en VH

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Target:
    box: Box
    lon_px: float  # centre pondéré, en pixels
    lat_px: float
    length_m: float
    width_m: float
    heading_deg: float
    pixels: int
    peak_db: float
    contrast_db: float
    vh_contrast_db: float | None = None


@dataclass
class CfarResult:
    targets: list[Target]
    candidates: int
    seconds: float
    model: str = MODEL
    stats: dict = field(default_factory=dict)


def _box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Somme sur un carré de demi-côté r, bords à zéro."""
    size = 2 * r + 1
    return ndimage.uniform_filter(a, size=size, mode="constant", cval=0.0) * (size * size)


def background(db: np.ndarray, valid: np.ndarray, guard: int, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moyenne, écart-type et part valide du fond, sur l'anneau garde-fond."""
    w = valid.astype(np.float64)
    x = np.where(valid, db, 0.0).astype(np.float64)
    n = _box_sum(w, bg) - _box_sum(w, guard)
    s1 = _box_sum(x, bg) - _box_sum(x, guard)
    s2 = _box_sum(x * x, bg) - _box_sum(x * x, guard)
    ring = (2 * bg + 1) ** 2 - (2 * guard + 1) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean * mean, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32), (n / ring).astype(np.float32)


def detect_band(db: np.ndarray, valid: np.ndarray, p: CfarParams) -> tuple[np.ndarray, np.ndarray]:
    """Carte des pixels candidats et contraste au fond, en dB, pour une polarisation."""
    mean, std, frac = background(db, valid, p.guard, p.background)
    hit = valid & (frac >= p.min_valid) & (db > mean + p.k * std) & (db - mean > p.min_contrast_db)
    # Seconde passe : les cibles de la première, élargies, sortent du fond.
    clean = valid & ~ndimage.binary_dilation(hit, iterations=2)
    mean, std, frac = background(db, clean, p.guard, p.background)
    contrast = db - mean
    hit = valid & (frac >= p.min_valid) & (db > mean + p.k * std) & (contrast > p.min_contrast_db)
    return hit, np.where(valid, contrast, np.nan)


def detect(vv: np.ndarray, vh: np.ndarray | None, water: np.ndarray | None, gsd: float,
           params: CfarParams | None = None) -> CfarResult:
    """`vv` et `vh` en intensité linéaire, NaN hors fauchée. `water` : True sur la mer."""
    p = params or CfarParams()
    t0 = time.perf_counter()
    finite = np.isfinite(vv)
    valid = finite if water is None else finite & water
    vv_db = 10 * np.log10(np.where(finite, np.maximum(vv, 1e-6), 1e-6)).astype(np.float32)
    hit, contrast = detect_band(vv_db, valid, p)
    vh_contrast = vh_db = None
    if p.use_vh and vh is not None:
        vh_ok = valid & np.isfinite(vh)
        vh_db = 10 * np.log10(np.where(vh_ok, np.maximum(vh, 1e-6), 1e-6)).astype(np.float32)
        vh_hit, vh_contrast = detect_band(vh_db, vh_ok, p)
        hit |= vh_hit
    grown = ndimage.binary_dilation(hit, iterations=p.merge_px) & valid if p.merge_px else hit
    labels, n = ndimage.label(grown, structure=np.ones((3, 3)))
    targets = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        comp = (labels[sl] == i) & hit[sl]
        if comp.sum() < p.min_pixels:
            continue
        c = contrast[sl]
        peak_contrast = float(np.nanmax(np.where(comp, c, np.nan)))
        vhc = None
        if vh_contrast is not None:
            vhc = float(np.nanmax(np.where(comp, vh_contrast[sl], np.nan)))
        vh_ok = vhc is not None and vhc >= p.vh_min_peak_contrast_db
        if peak_contrast < p.min_peak_contrast_db and not vh_ok:
            continue
        # La coque se mesure en VH quand elle y ressort bien : les lobes secondaires
        # y restent sous le bruit, la mer étant 10 dB plus sombre qu'en VV.
        use_vh = vh_db is not None and vhc is not None and vhc >= p.vh_measure_db
        t = _measure(vh_db[sl] if use_vh else vv_db[sl], comp, sl, gsd, p)
        if t is None or t.length_m > p.max_length_m:
            continue
        # Une longue cible pâle : digue oubliée, nappe, front de houle, pas un cargo.
        if t.length_m >= p.big_length_m and max(peak_contrast, vhc or -99.0) < p.big_min_contrast_db:
            continue
        t.contrast_db, t.vh_contrast_db = round(peak_contrast, 1), None if vhc is None else round(vhc, 1)
        targets.append(t)
    targets.sort(key=lambda t: -t.contrast_db)
    stats = {"sea_pixels": int(valid.sum()), "candidate_pixels": int(hit.sum()),
             "vv_sea_median_db": round(float(np.median(vv_db[valid])), 1) if valid.any() else None}
    return CfarResult(targets=targets, candidates=n, seconds=time.perf_counter() - t0, stats=stats)


def _measure(db: np.ndarray, comp: np.ndarray, sl: tuple[slice, slice], gsd: float, p: CfarParams) -> Target | None:
    """Longueur, largeur et axe sur la coque : les pixels proches du pic."""
    ys, xs = np.nonzero(comp)
    vals = db[ys, xs]
    peak = float(vals.max())
    keep = vals >= peak - p.hull_db
    hx, hy, hv = xs[keep], ys[keep], vals[keep]
    if len(hx) >= 3 and (np.ptp(hx) > 0 or np.ptp(hy) > 0):
        try:
            length, width, angle, _, _ = _axis(hx, hy)
        except np.linalg.LinAlgError:
            length, width, angle = 1.0, 1.0, 0.0
    else:
        length, width, angle = float(max(np.ptp(hx), np.ptp(hy)) + 1), 1.0, 0.0
    # Centre pondéré par l'intensité linéaire : le pic, pas le milieu des lobes.
    wts = 10 ** ((hv - peak) / 10)
    cx = float((hx * wts).sum() / wts.sum()) + sl[1].start
    cy = float((hy * wts).sum() / wts.sum()) + sl[0].start
    x1, y1 = xs.min() + sl[1].start, ys.min() + sl[0].start
    x2, y2 = xs.max() + sl[1].start + 1, ys.max() + sl[0].start + 1
    length_m = max(gsd, length * gsd - p.psf_m)
    return Target(box=Box(float(x1), float(y1), float(x2), float(y2), 0.0, "vessel"), lon_px=cx + 0.5,
                  lat_px=cy + 0.5, length_m=round(length_m, 0), width_m=round(width * gsd, 0),
                  heading_deg=round(angle, 0), pixels=int(comp.sum()), peak_db=round(peak, 1), contrast_db=0.0)


def confidence(contrast_db: float, p: CfarParams | None = None) -> float:
    """Pseudo-confiance entre 0 et 1, croissante avec le contraste au fond.

    0.5 au contraste minimal d'un navire, 0.9 vers 30 dB. Pas une probabilité :
    seulement de quoi trier et de quoi remplir le champ commun avec YOLO.
    """
    p = p or CfarParams()
    z = (contrast_db - p.min_peak_contrast_db) / 8.0
    return round(1 / (1 + math.exp(-z)), 3)
