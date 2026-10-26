"""Un scan radar : une zone, une passe Sentinel-1, ses navires.

Même rangement sur disque que `scan.run`, pour que la carte, `ais.crosscheck` et
l'analyste marchent sans changement :

    data/scans/<id>/
        scan.json            métadonnées, avec "sensor": "sentinel-1"
        detections.geojson   un Feature par navire, mêmes propriétés que l'optique
        image.jpg            fausse couleur VV, VH, VV/VH, posée par ses 4 coins
        crops/V001.png       chaque navire agrandi
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime

import numpy as np
from PIL import Image
from pyproj import Transformer

from . import cfar, geo, infra, sar
from .scan import _corners, _save_crops, scan_dir
from .sentinel import BBox

SENSOR = "sentinel-1"


def to_features(result: cfar.CfarResult, transform, crs: str, water: np.ndarray | None,
                params: cfar.CfarParams, min_water: float = 0.25) -> list[dict]:
    """GeoJSON en WGS84, mêmes propriétés que `geo.to_features`, plus quelques mesures radar."""
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    feats = []
    for i, t in enumerate(result.targets, start=1):
        b = t.box
        wf = geo.water_fraction(water, b) if water is not None else None
        if wf is not None and wf < min_water:
            continue
        corners = [transform @ (b.x1, b.y1), transform @ (b.x2, b.y1), transform @ (b.x2, b.y2),
                   transform @ (b.x1, b.y2)]
        ring = [list(to_wgs.transform(x, y)) for x, y in corners]
        ring.append(ring[0])
        lon, lat = to_wgs.transform(*(transform @ (t.lon_px, t.lat_px)))
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "id": f"V{i:03d}", "confidence": cfar.confidence(max(t.contrast_db, t.vh_contrast_db or -99), params),
                "label": "vessel", "lat": round(lat, 6), "lon": round(lon, 6),
                "length_m": float(t.length_m), "width_m": float(t.width_m), "heading_deg": float(t.heading_deg),
                # Le radar ne voit pas de sillage exploitable à 10 m : pas de sens de marche.
                "course_deg": None, "wake_m": None, "underway": False, "measured": True,
                "box_px": [round(float(v), 1) for v in (b.x1, b.y1, b.x2, b.y2)],
                "water_fraction": None if wf is None else round(wf, 2),
                "sensor": SENSOR, "peak_db": t.peak_db, "contrast_db": t.contrast_db,
                "vh_contrast_db": t.vh_contrast_db, "pixels": t.pixels,
            },
        })
    return geo.renumber(feats)


def run(bbox: BBox, scene: sar.SarScene | None = None, params: cfar.CfarParams | None = None,
        days: int = 12, name: str | None = None, log=print, **_ignored) -> dict:
    """Scanne la zone sur la passe donnée, ou la plus récente. `max_cloud` et autres sont ignorés."""
    t0 = time.perf_counter()
    if scene is None:
        scenes = sar.search(bbox, days=days)
        if not scenes:
            raise LookupError(f"aucune passe Sentinel-1 sur {days} jours")
        scene = scenes[0]
    log(f"passe {scene.id}, {scene.datetime[:16]}, orbite {scene.orbit_state or '?'}")
    raster = sar.load(scene, bbox, log=log)
    t_read = time.perf_counter() - t0
    h, w = raster.vv.shape
    log(f"fenêtre {w}x{h} px lue en {t_read:.1f} s, masque terre {'oui' if raster.water is not None else 'non'}")
    params = params or cfar.CfarParams()
    result = cfar.detect(raster.vv, raster.vh, raster.water, raster.gsd, params)
    feats = to_features(result, raster.transform, raster.crs, raster.water, params)
    log(f"{len(result.targets)} cibles CFAR, {len(feats)} sur l'eau, en {result.seconds:.1f} s")
    try:
        fixed = infra.flag(feats, infra.fetch(bbox))
        if fixed:
            log(f"{fixed} posées sur un objet fixe d'OpenStreetMap : éolienne, phare, balise")
    except Exception as exc:  # noqa: BLE001 - Overpass est un bonus, le scan vaut sans
        fixed = None
        log(f"objets fixes non vérifiés : {exc}")

    scan_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    out = scan_dir(scan_id)
    out.mkdir(parents=True)
    rgb = sar.quicklook(raster)
    Image.fromarray(rgb).save(out / "image.jpg", quality=88)
    _save_crops(rgb, feats, out / "crops")
    (out / "detections.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    meta = {
        "id": scan_id, "name": name, "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "sensor": SENSOR, "bbox": list(bbox), "scene": scene.to_dict(), "params": params.to_dict(),
        "model": result.model, "size_px": [w, h], "gsd_m": raster.gsd, "crs": raster.crs,
        "transform": list(raster.transform)[:6],
        "corners": _corners(raster.transform, raster.crs, w, h),
        "water_mask": raster.water is not None,
        "counts": {"detections": len(result.targets), "vessels": len(feats) - (fixed or 0), "fixed": fixed},
        "timing_s": {"read": round(t_read, 2), "detect": round(result.seconds, 2),
                     "total": round(time.perf_counter() - t0, 2)},
        "cfar": result.stats,
    }
    (out / "scan.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta
