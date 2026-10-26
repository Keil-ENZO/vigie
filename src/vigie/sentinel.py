"""Recherche et lecture des passes Sentinel-2 L2A, via le STAC public d'Element 84.

On ne télécharge jamais une tuile entière de 110 km : on lit seulement la fenêtre
de la zone demandée dans le COG `visual`, en HTTP Range.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta

import httpx
import numpy as np
import rasterio
from shapely.geometry import box, shape
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

STAC = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
MAX_SIDE_PX = 6000  # 60 km de côté à 10 m : garde-fou mémoire et temps de calcul
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE")

BBox = tuple[float, float, float, float]  # ouest, sud, est, nord en WGS84


@dataclass
class Scene:
    id: str
    datetime: str
    cloud_cover: float
    platform: str
    visual_href: str
    scl_href: str | None
    epsg: int | None
    thumbnail: str | None
    coverage: float | None = None  # part de la zone couverte par la passe

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Raster:
    """Fenêtre lue : pixels RGB et passage pixel vers coordonnées."""

    rgb: np.ndarray
    transform: rasterio.Affine
    crs: str
    gsd: float
    water: np.ndarray | None = None  # masque booléen à la résolution de rgb


def _feature_to_scene(f: dict) -> Scene:
    a, p = f["assets"], f["properties"]
    return Scene(
        id=f["id"], datetime=p["datetime"], cloud_cover=float(p.get("eo:cloud_cover", -1)),
        platform=p.get("platform", ""), visual_href=a["visual"]["href"],
        scl_href=a.get("scl", {}).get("href"), epsg=p.get("proj:epsg"),
        thumbnail=a.get("thumbnail", {}).get("href"),
    )


def search(bbox: BBox, days: int = 30, max_cloud: float = 20.0, limit: int = 10,
           until: date | None = None) -> list[Scene]:
    """Passes qui couvrent la zone, de la plus récente à la plus ancienne."""
    end = until or date.today()
    body = {
        "collections": [COLLECTION], "bbox": list(bbox), "limit": limit,
        "datetime": f"{end - timedelta(days=days)}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    r = httpx.post(f"{STAC}/search", json=body, timeout=30)
    r.raise_for_status()
    feats = r.json()["features"]
    # Le STAC rend aussi les tuiles qui ne touchent qu'un coin de la zone. On classe
    # par part de la zone couverte de pixels valides, l'empreinte `geometry`, puis par date.
    area = box(*bbox)
    for f in feats:
        try:
            f["_cover"] = shape(f["geometry"]).intersection(area).area / area.area
        except Exception:  # noqa: BLE001 - géométrie absente ou invalide
            f["_cover"] = 0.0
    good = [f for f in feats if f["_cover"] >= 0.98] or sorted(feats, key=lambda f: -f["_cover"])
    scenes = []
    for f in good:
        sc = _feature_to_scene(f)
        sc.coverage = round(f["_cover"], 3)
        scenes.append(sc)
    return scenes


def read_window(href: str, bbox: BBox, out_shape: tuple[int, int] | None = None,
                resampling: Resampling = Resampling.bilinear):
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        win = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width * win.height == 0:
            raise ValueError("la zone ne recoupe pas cette passe")
        shape = out_shape or (int(win.height), int(win.width))
        data = src.read(window=win, out_shape=(src.count, *shape), resampling=resampling, boundless=False)
        transform = src.window_transform(win)
        if out_shape:
            transform = transform * transform.scale(win.width / shape[1], win.height / shape[0])
        return data, transform, src.crs.to_string(), abs(src.transform.a)


def load(scene: Scene, bbox: BBox, with_water: bool = True) -> Raster:
    data, transform, crs, gsd = read_window(scene.visual_href, bbox)
    if max(data.shape[1:]) > MAX_SIDE_PX:
        raise ValueError(f"zone trop grande : {data.shape[2]}x{data.shape[1]} px, max {MAX_SIDE_PX}")
    rgb = np.moveaxis(data[:3], 0, -1).copy()
    water = None
    if with_water and scene.scl_href:
        try:
            scl, *_ = read_window(scene.scl_href, bbox, out_shape=rgb.shape[:2], resampling=Resampling.nearest)
            water = scl[0] == 6  # classe SCL 6 : eau
        except Exception:  # noqa: BLE001 - le masque est un bonus, la détection marche sans
            water = None
    return Raster(rgb=rgb, transform=transform, crs=crs, gsd=gsd, water=water)


def get_scene(scene_id: str) -> Scene:
    r = httpx.get(f"{STAC}/collections/{COLLECTION}/items/{scene_id}", timeout=30)
    r.raise_for_status()
    return _feature_to_scene(r.json())
