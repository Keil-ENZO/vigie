"""Les objets fixes en mer qui ressemblent à des navires vus de l'espace.

Éoliennes, plateformes, phares : un point blanc sur l'eau, toujours au même endroit.
On les prend dans OpenStreetMap, par l'API Overpass, et on marque les détections
qui tombent dessus. Trouvé par la validation AIS de l'Øresund : le parc de
Middelgrunden donnait 20 « navires sombres ».
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import httpx

OVERPASS = os.environ.get("VIGIE_OVERPASS", "https://overpass-api.de/api/interpreter")
CACHE = Path(os.environ.get("VIGIE_DATA", Path(__file__).resolve().parents[2] / "data")) / "infra-cache"
RADIUS_M = 80

QUERY = """[out:json][timeout:25];
(
  node["generator:source"="wind"]({s},{w},{n},{e});
  node["power"="generator"]["generator:method"="wind_turbine"]({s},{w},{n},{e});
  node["man_made"~"^(offshore_platform|lighthouse|mast|beacon)$"]({s},{w},{n},{e});
  way["man_made"="offshore_platform"]({s},{w},{n},{e});
  node["seamark:type"~"^(platform|light_major|light_minor|beacon_lateral|beacon_cardinal)$"]({s},{w},{n},{e});
);
out center;"""


def _kind(tags: dict) -> str:
    if tags.get("generator:source") == "wind" or tags.get("generator:method") == "wind_turbine":
        return "wind_turbine"
    if "platform" in (tags.get("man_made", "") + tags.get("seamark:type", "")):
        return "platform"
    if "light" in (tags.get("man_made", "") + tags.get("seamark:type", "")):
        return "lighthouse"
    return "beacon"


def fetch(bbox) -> list[dict]:
    w, s, e, n = bbox
    key = hashlib.sha1(json.dumps([round(v, 4) for v in bbox]).encode()).hexdigest()[:16]
    cache = CACHE / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = httpx.post(OVERPASS, data={"data": QUERY.format(w=w, s=s, e=e, n=n)}, timeout=40,
                   headers={"User-Agent": "vigie/0.1 (projet perso, surveillance maritime locale)"})
    r.raise_for_status()
    objs = []
    for el in r.json().get("elements", []):
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if lat is not None:
            objs.append({"lat": lat, "lon": lon, "kind": _kind(el.get("tags", {})), "osm_id": el["id"]})
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(objs))
    return objs


def flag(features: list[dict], objects: list[dict], radius: float = RADIUS_M) -> int:
    """Ajoute `fixed` aux détections posées sur un objet fixe. Rend leur nombre."""
    n = 0
    for f in features:
        p = f["properties"]
        for o in objects:
            dy = (o["lat"] - p["lat"]) * 111320
            dx = (o["lon"] - p["lon"]) * 111320 * math.cos(math.radians(p["lat"]))
            if math.hypot(dx, dy) <= radius:
                p["fixed"] = o["kind"]
                n += 1
                break
    return n
