"""Masque terre pour le radar, tiré d'ESA WorldCover 2021.

Le CFAR s'allume partout à terre : bâtiments, grues, forêts. Le masque SCL de
Sentinel-2 dépend d'une passe sans nuages, ce qu'on cherche justement à éviter. On
prend donc WorldCover : 10 m, mondial, classe 80 pour l'eau permanente, en COG sur
le Planetary Computer. On lit seulement la zone, reprojetée sur la grille du radar,
et on la garde en cache sur disque.

Les quais et bassins portuaires sont classés « bâti ». Un navire à quai tombe donc
souvent dans la marge de terre : c'est voulu, les grues et les conteneurs d'un quai
ressemblent trop à des navires pour le radar.

WorldCover rate en revanche les ouvrages fins : une digue de 15 m de large reste
« eau », et le radar la voit comme un long navire. On ajoute donc les digues, jetées,
épis et ponts d'OpenStreetMap, élargis de quelques pixels. Trouvé à Skagen : la
digue du port donnait un « navire sombre » de 230 m.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from scipy import ndimage
from shapely.geometry import LineString, Polygon

from .infra import OVERPASS

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "esa-worldcover"
WATER = 80
NODATA = 0
CACHE = Path(os.environ.get("VIGIE_DATA", Path(__file__).resolve().parents[2] / "data")) / "landmask-cache"
COAST_MARGIN_PX = 3  # 30 m : écart de géolocalisation entre WorldCover et le radar
STRUCTURE_BUFFER_M = 30.0  # demi-largeur masquée autour d'une digue
# Un tablier de pont à 50 m au-dessus de l'eau se replie vers le radar (repliement,
# « layover ») de h / tan(incidence), soit 70 à 90 m en IW. Trouvé sur l'Øresund.
BRIDGE_BUFFER_M = 120.0

STRUCTURES_QUERY = """[out:json][timeout:25];
(
  way["man_made"~"^(breakwater|groyne|pier|quay|dyke)$"]({s},{w},{n},{e});
  way["bridge"]["highway"]({s},{w},{n},{e});
  way["bridge"]["railway"]({s},{w},{n},{e});
  way["man_made"="bridge"]({s},{w},{n},{e});
);
out geom;"""


def _items(bbox) -> list[dict]:
    body = {"collections": [COLLECTION], "bbox": list(bbox), "limit": 10}
    r = httpx.post(f"{STAC}/search", json=body, timeout=60)
    r.raise_for_status()
    feats = r.json()["features"]
    # Une seule édition suffit : la plus récente, v200 de 2021.
    latest = max((f["properties"].get("start_datetime", "") for f in feats), default="")
    return [f for f in feats if f["properties"].get("start_datetime", "") == latest]


def classes(bbox, transform, crs: str, shape: tuple[int, int]) -> np.ndarray:
    """Classes WorldCover sur la grille demandée. 0 là où aucune tuile ne couvre."""
    from .sar import GDAL_ENV, sign  # import tardif : sar importe ce module

    key = hashlib.sha1(json.dumps([list(bbox), list(transform)[:6], crs, list(shape)]).encode()).hexdigest()[:16]
    cache = CACHE / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    out = np.zeros(shape, np.uint8)
    items = _items(bbox)
    if not items:
        raise LookupError("aucune tuile WorldCover sur la zone")
    with rasterio.Env(**GDAL_ENV):
        for item in items:
            with rasterio.open(sign(item["assets"]["map"]["href"])) as src, \
                    WarpedVRT(src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                              resampling=Resampling.nearest, nodata=NODATA) as vrt:
                part = vrt.read(1)
            out = np.where(out == NODATA, part, out)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def structures(bbox) -> list[dict]:
    """Lignes et contours des digues, jetées et ponts d'OpenStreetMap, en lon/lat, en cache."""
    w, s, e, n = bbox
    key = hashlib.sha1(json.dumps(["structures-v2", [round(v, 4) for v in bbox]]).encode()).hexdigest()[:16]
    cache = CACHE / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = httpx.post(OVERPASS, data={"data": STRUCTURES_QUERY.format(w=w, s=s, e=e, n=n)}, timeout=40,
                   headers={"User-Agent": "vigie/0.1 (projet perso, surveillance maritime locale)"})
    r.raise_for_status()
    lines = [{"bridge": "bridge" in el.get("tags", {}) or el.get("tags", {}).get("man_made") == "bridge",
              "coords": [(pt["lon"], pt["lat"]) for pt in el["geometry"]]}
             for el in r.json().get("elements", []) if len(el.get("geometry") or []) >= 2]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(lines))
    return lines


def structures_mask(lines: list[dict], transform, crs: str, shape: tuple[int, int],
                    buffer_m: float = STRUCTURE_BUFFER_M, bridge_m: float = BRIDGE_BUFFER_M) -> np.ndarray:
    """True sur les ouvrages, élargis de part et d'autre : `buffer_m`, ou `bridge_m` pour un pont."""
    if not lines:
        return np.zeros(shape, bool)
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    geoms = []
    for item in lines:
        line = item["coords"]
        g = Polygon(line) if len(line) >= 4 and line[0] == line[-1] else LineString(line)
        g = shapely.transform(g, lambda x, y: to_utm(x, y), interleaved=False)
        geoms.append(g.buffer(bridge_m if item.get("bridge") else buffer_m))
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform, fill=0,
                     dtype="uint8").astype(bool)


def water_from_classes(lc: np.ndarray, margin: int = COAST_MARGIN_PX) -> np.ndarray:
    """Eau sûre : classe 80, rognée d'une marge le long des côtes.

    Hors couverture WorldCover, au large, la tuile manque : on compte ces pixels
    comme de l'eau, WorldCover couvrant toutes les terres émergées.
    """
    water = (lc == WATER) | (lc == NODATA)
    if margin > 0:
        water = ndimage.binary_erosion(water, iterations=margin, border_value=1)
    return water


def water_mask(bbox, transform, crs: str, shape: tuple[int, int], margin: int = COAST_MARGIN_PX,
               with_structures: bool = True, log=print) -> np.ndarray:
    """Masque booléen, True sur la mer : WorldCover moins les ouvrages d'OpenStreetMap."""
    water = water_from_classes(classes(bbox, transform, crs, shape), margin)
    if with_structures:
        try:
            water &= ~structures_mask(structures(bbox), transform, crs, shape)
        except Exception as exc:  # noqa: BLE001 - Overpass est un bonus, WorldCover suffit
            log(f"digues et ponts non masqués : {exc}")
    return water
