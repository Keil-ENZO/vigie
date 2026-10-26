"""Recherche et lecture des passes Sentinel-1, via le STAC Microsoft Planetary Computer.

Le radar voit la nuit et à travers les nuages. On lit le produit RTC, pas le GRD :

- le GRD est en géométrie radar, géoréférencé par une grille de GCP. Il faut le
  rééchantillonner soi-même, et ses valeurs sont des comptes numériques à calibrer ;
- le RTC est le même GRD, calibré en gamma0 et projeté en UTM à 10 m, en COG. La
  fenêtre se lit en HTTP Range exactement comme une tuile Sentinel-2, sur la même
  grille. La correction de relief ne sert à rien en mer, mais elle ne gêne pas.

Le GRD reste un repli, projeté à la volée par GDAL à partir de ses GCP, si le RTC
d'une passe manque. Les URL du Planetary Computer se signent anonymement par l'API SAS.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import httpx
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape

from . import landmask
from .sentinel import GDAL_ENV as S2_GDAL_ENV
from .sentinel import BBox

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
RTC = "sentinel-1-rtc"
GRD = "sentinel-1-grd"
GSD = 10.0
MAX_SIDE_PX = 6000
# Les COG du Planetary Computer finissent en .tiff, suivis d'une signature.
GDAL_ENV = dict(S2_GDAL_ENV, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff")
_signed: dict[str, tuple[str, float]] = {}


@dataclass
class SarScene:
    id: str
    datetime: str
    platform: str
    collection: str
    vv_href: str
    vh_href: str | None
    orbit_state: str | None
    relative_orbit: int | None
    epsg: int | None
    thumbnail: str | None
    coverage: float | None = None
    # Mêmes clés que la passe Sentinel-2 : l'interface et l'analyste lisent `cloud_cover`.
    # Le radar traverse les nuages, on écrit donc 0.
    cloud_cover: float = 0.0
    sensor: str = "sentinel-1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SarRaster:
    """Fenêtre lue : intensités linéaires VV et VH, NaN hors fauchée."""

    vv: np.ndarray
    vh: np.ndarray | None
    transform: rasterio.Affine
    crs: str
    gsd: float
    water: np.ndarray | None = None


def sign(href: str) -> str:
    """URL signée, gardée en cache tant qu'elle reste valable."""
    hit = _signed.get(href)
    if hit and hit[1] > time.time() + 600:
        return hit[0]
    r = httpx.get(SAS, params={"href": href}, timeout=30)
    r.raise_for_status()
    signed = r.json()["href"]
    _signed[href] = (signed, time.time() + 3 * 3600)
    return signed


def _feature_to_scene(f: dict) -> SarScene:
    a, p = f["assets"], f["properties"]
    return SarScene(
        id=f["id"], datetime=p["datetime"], platform=p.get("platform", ""), collection=f.get("collection", RTC),
        vv_href=a["vv"]["href"], vh_href=a.get("vh", {}).get("href"), orbit_state=p.get("sat:orbit_state"),
        relative_orbit=p.get("sat:relative_orbit"), epsg=p.get("proj:epsg"),
        thumbnail=a.get("thumbnail", a.get("rendered_preview", {})).get("href"),
    )


def search(bbox: BBox, days: int = 12, limit: int = 20, until: date | None = None,
           collection: str = RTC) -> list[SarScene]:
    """Passes IW en double polarisation VV et VH, de la plus récente à la plus ancienne."""
    end = until or date.today()
    body = {
        "collections": [collection], "bbox": list(bbox), "limit": limit,
        "datetime": f"{end - timedelta(days=days)}T00:00:00Z/{end}T23:59:59Z",
        "query": {"sar:instrument_mode": {"eq": "IW"}, "sar:polarizations": {"eq": ["VV", "VH"]}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    r = httpx.post(f"{STAC}/search", json=body, timeout=60)
    r.raise_for_status()
    feats = r.json()["features"]
    area = box(*bbox)
    for f in feats:
        try:
            f["_cover"] = shape(f["geometry"]).intersection(area).area / area.area
        except Exception:  # noqa: BLE001 - géométrie absente ou invalide
            f["_cover"] = 0.0
    # Deux tranches d'une même orbite se partagent parfois la zone : on ne garde que
    # les passes qui la couvrent, sinon la meilleure couverture.
    good = [f for f in feats if f["_cover"] >= 0.95] or sorted(feats, key=lambda f: -f["_cover"])
    scenes = []
    for f in good:
        sc = _feature_to_scene(f)
        sc.coverage = round(f["_cover"], 3)
        scenes.append(sc)
    return scenes


def get_scene(scene_id: str) -> SarScene:
    """Passe par identifiant, RTC d'abord, GRD en repli."""
    base = scene_id.removesuffix("_rtc")
    for coll, iid in ((RTC, f"{base}_rtc"), (GRD, base)):
        r = httpx.get(f"{STAC}/collections/{coll}/items/{iid}", timeout=30)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        return _feature_to_scene(r.json())
    raise LookupError(f"passe Sentinel-1 inconnue : {scene_id}")


def utm_crs(bbox: BBox) -> str:
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    zone = math.floor((lon + 180) / 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def grid(bbox: BBox, crs: str, gsd: float = GSD) -> tuple[rasterio.Affine, int, int]:
    """Grille cible alignée sur des multiples de `gsd`, comme les tuiles Sentinel."""
    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    left, top = math.floor(left / gsd) * gsd, math.ceil(top / gsd) * gsd
    width, height = math.ceil((right - left) / gsd), math.ceil((top - bottom) / gsd)
    return rasterio.Affine(gsd, 0, left, 0, -gsd, top), width, height


def _read_rtc(href: str, bbox: BBox) -> tuple[np.ndarray, rasterio.Affine, str]:
    with rasterio.Env(**GDAL_ENV), rasterio.open(sign(href)) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        win = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width * win.height == 0:
            raise ValueError("la zone ne recoupe pas cette passe")
        if max(win.width, win.height) > MAX_SIDE_PX:
            raise ValueError(f"zone trop grande : {win.width}x{win.height} px, max {MAX_SIDE_PX}")
        if abs(abs(src.transform.a) - GSD) > 0.01:
            raise ValueError(f"pas de {src.transform.a} m inattendu pour un RTC")
        data = src.read(1, window=win).astype(np.float32)
        nodata = src.nodata
        transform = src.window_transform(win)
        crs = src.crs.to_string()
    if nodata is not None:
        data[data == nodata] = np.nan
    data[data <= 0] = np.nan
    return data, transform, crs


def _grd_window(gcps, bbox: BBox, width: int, height: int, pad: int = 64) -> Window:
    """Fenêtre radar qui couvre la zone, par un plan ajusté sur les GCP voisins.

    La grille de GCP du GRD est lâche, une vingtaine de points par ligne : un modèle
    linéaire lon/lat vers ligne/colonne suffit à quelques kilomètres près, la marge
    `pad` absorbe l'écart.
    """
    pts = np.array([(g.x, g.y, g.row, g.col) for g in gcps])
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    near = pts[np.argsort(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))[:12]]
    design = np.column_stack([near[:, 0], near[:, 1], np.ones(len(near))])
    coef_row, *_ = np.linalg.lstsq(design, near[:, 2], rcond=None)
    coef_col, *_ = np.linalg.lstsq(design, near[:, 3], rcond=None)
    corners = np.array([(x, y, 1.0) for x in (bbox[0], bbox[2]) for y in (bbox[1], bbox[3])])
    rows, cols = corners @ coef_row, corners @ coef_col
    r0, r1 = max(0, int(rows.min()) - pad), min(height, math.ceil(rows.max()) + pad)
    c0, c1 = max(0, int(cols.min()) - pad), min(width, math.ceil(cols.max()) + pad)
    if r1 <= r0 or c1 <= c0:
        raise ValueError("la zone ne recoupe pas cette passe")
    return Window(c0, r0, c1 - c0, r1 - r0)


def _read_grd(href: str, bbox: BBox, transform: rasterio.Affine, crs: str, width: int, height: int) -> np.ndarray:
    """GRD en géométrie radar, projeté par ses GCP sur la grille UTM demandée.

    On lit la seule fenêtre radar utile, puis on la projette avec les GCP décalés
    d'autant. Les valeurs sont des amplitudes non calibrées : on rend leur carré, une
    intensité à un facteur près, ce qui suffit au CFAR qui ne regarde que des rapports.
    """
    with rasterio.Env(**GDAL_ENV), rasterio.open(sign(href)) as src:
        gcps, gcp_crs = src.gcps
        win = _grd_window(gcps, bbox, src.width, src.height)
        raw = src.read(1, window=win).astype(np.float32)
    shifted = [GroundControlPoint(row=g.row - win.row_off, col=g.col - win.col_off, x=g.x, y=g.y, z=g.z)
               for g in gcps]
    out = np.zeros((height, width), np.float32)
    reproject(raw, out, gcps=shifted, src_crs=gcp_crs, dst_transform=transform, dst_crs=CRS.from_string(crs),
              resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0)
    out[out <= 0] = np.nan
    return out * out


def load(scene: SarScene, bbox: BBox, with_water: bool = True, log=print) -> SarRaster:
    if scene.collection == RTC:
        vv, transform, crs = _read_rtc(scene.vv_href, bbox)
        vh = None
        if scene.vh_href:
            vh2, t2, _ = _read_rtc(scene.vh_href, bbox)
            if vh2.shape == vv.shape and t2 == transform:
                vh = vh2
    else:
        crs = utm_crs(bbox)
        transform, width, height = grid(bbox, crs)
        if max(width, height) > MAX_SIDE_PX:
            raise ValueError(f"zone trop grande : {width}x{height} px, max {MAX_SIDE_PX}")
        vv = _read_grd(scene.vv_href, bbox, transform, crs, width, height)
        vh = _read_grd(scene.vh_href, bbox, transform, crs, width, height) if scene.vh_href else None
    if np.isnan(vv).all():
        raise ValueError("la fenêtre est entièrement hors fauchée")
    water = None
    if with_water:
        try:
            water = landmask.water_mask(bbox, transform, crs, vv.shape, log=log)
        except Exception as exc:  # noqa: BLE001 - sans masque, le scan marche mais s'allume à terre
            log(f"masque terre indisponible : {exc}")
    return SarRaster(vv=vv, vh=vh, transform=transform, crs=crs, gsd=GSD, water=water)


def to_db(intensity: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10 * np.log10(np.maximum(intensity, floor))


def quicklook(raster: SarRaster) -> np.ndarray:
    """Fausse couleur classique : rouge VV, vert VH, bleu VV/VH. Noir hors fauchée."""
    vv_db = to_db(raster.vv)
    r = np.clip((vv_db + 25) / 25, 0, 1)
    if raster.vh is None:
        g = b = r
    else:
        vh_db = to_db(raster.vh)
        g = np.clip((vh_db + 32) / 25, 0, 1)
        b = np.clip((vv_db - vh_db - 2) / 14, 0, 1)
    rgb = (np.nan_to_num(np.stack([r, g, b], axis=-1)) * 255).astype(np.uint8)
    rgb[np.isnan(raster.vv)] = 0
    return rgb
