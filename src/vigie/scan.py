"""Un scan : une zone, une passe Sentinel-2, ses navires. Tout est rangé sur disque.

    data/scans/<id>/
        scan.json            métadonnées, paramètres, temps
        detections.geojson   un Feature par navire
        image.jpg            la fenêtre RGB, posée sur la carte par ses 4 coins
        crops/V001.png       chaque navire agrandi, pour l'interface et le VLM
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

from . import detector, geo, infra, sentinel
from .tiling import Box

DATA = Path(os.environ.get("VIGIE_DATA", Path(__file__).resolve().parents[2] / "data"))
SCANS = DATA / "scans"
CROP_SIDE = 48  # pixels Sentinel autour du navire, soit 480 m
CROP_SCALE = 4


def scan_dir(scan_id: str) -> Path:
    if not scan_id.replace("-", "").isalnum():
        raise ValueError("identifiant de scan invalide")
    return SCANS / scan_id


def _corners(transform, crs: str, width: int, height: int) -> list[list[float]]:
    """Coins haut-gauche, haut-droit, bas-droit, bas-gauche en lon/lat, l'ordre de MapLibre."""
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    pts = [(0, 0), (width, 0), (width, height), (0, height)]
    return [list(to_wgs.transform(*(transform * p))) for p in pts]


def _save_crops(rgb: np.ndarray, features: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    half = CROP_SIDE // 2
    for f in features:
        x1, y1, x2, y2 = f["properties"]["box_px"]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        l, t = min(max(0, cx - half), max(0, w - CROP_SIDE)), min(max(0, cy - half), max(0, h - CROP_SIDE))
        crop = Image.fromarray(rgb[t : t + CROP_SIDE, l : l + CROP_SIDE])
        crop = crop.resize((crop.width * CROP_SCALE, crop.height * CROP_SCALE), Image.Resampling.LANCZOS)
        crop.save(out / f"{f['properties']['id']}.png")


def run(bbox: sentinel.BBox, scene: sentinel.Scene | None = None, params: detector.DetectParams | None = None,
        days: int = 30, max_cloud: float = 20.0, name: str | None = None, log=print) -> dict:
    t0 = time.perf_counter()
    if scene is None:
        scenes = sentinel.search(bbox, days=days, max_cloud=max_cloud)
        if not scenes:
            raise LookupError(f"aucune passe Sentinel-2 sous {max_cloud} % de nuages sur {days} jours")
        scene = scenes[0]
    log(f"passe {scene.id}, {scene.datetime[:16]}, nuages {scene.cloud_cover:.1f} %")
    raster = sentinel.load(scene, bbox)
    t_read = time.perf_counter() - t0
    h, w = raster.rgb.shape[:2]
    log(f"fenêtre {w}x{h} px lue en {t_read:.1f} s")
    params = params or detector.DetectParams()
    result = detector.detect(raster.rgb, params)
    feats = geo.renumber(geo.to_features(result.boxes, raster.transform, raster.crs, raster.gsd,
                                         raster.water, rgb=raster.rgb))
    log(f"{len(result.boxes)} détections, {len(feats)} sur l'eau, {result.tiles} tuiles en {result.seconds:.1f} s")
    try:
        fixed = infra.flag(feats, infra.fetch(bbox))
        if fixed:
            log(f"{fixed} posées sur un objet fixe d'OpenStreetMap : éolienne, phare, balise")
    except Exception as exc:  # noqa: BLE001 - Overpass est un bonus, le scan vaut sans
        fixed = None
        log(f"objets fixes non vérifiés : {exc}")

    scan_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    out = scan_dir(scan_id)
    out.mkdir(parents=True)
    Image.fromarray(raster.rgb).save(out / "image.jpg", quality=88)
    _save_crops(raster.rgb, feats, out / "crops")
    (out / "detections.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    meta = {
        "id": scan_id, "name": name, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox": list(bbox), "scene": scene.to_dict(), "params": asdict(params), "model": result.model,
        "size_px": [w, h], "gsd_m": raster.gsd, "crs": raster.crs,
        "corners": _corners(raster.transform, raster.crs, w, h),
        "water_mask": raster.water is not None,
        "counts": {"detections": len(result.boxes), "vessels": len(feats) - (fixed or 0), "fixed": fixed},
        "timing_s": {"read": round(t_read, 2), "detect": round(result.seconds, 2),
                     "total": round(time.perf_counter() - t0, 2)},
    }
    (out / "scan.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def detect_upload(rgb: np.ndarray, params: detector.DetectParams | None = None) -> tuple[list[Box], detector.DetectResult]:
    """Image libre sans géoréférence, comme la démo d'origine."""
    result = detector.detect(rgb, params)
    return result.boxes, result


def list_scans() -> list[dict]:
    if not SCANS.exists():
        return []
    metas = []
    for p in sorted(SCANS.iterdir(), reverse=True):
        f = p / "scan.json"
        if f.exists():
            m = json.loads(f.read_text())
            m["analysed"] = (p / "analysis.json").exists()
            m["ais"] = (p / "ais.json").exists()
            metas.append(m)
    return metas


def load_scan(scan_id: str) -> tuple[dict, dict]:
    d = scan_dir(scan_id)
    meta = json.loads((d / "scan.json").read_text())
    det = json.loads((d / "detections.geojson").read_text())
    analysis = d / "analysis.json"
    if analysis.exists():
        by_id = json.loads(analysis.read_text()).get("vessels", {})
        for f in det["features"]:
            if (a := by_id.get(f["properties"]["id"])) is not None:
                f["properties"]["analysis"] = a
    ais = d / "ais.json"
    if ais.exists():
        a = json.loads(ais.read_text())
        for f in det["features"]:
            if (m := a["detections"].get(f["properties"]["id"])) is not None:
                f["properties"]["ais"] = m
        det["ais_summary"], det["ais_only"] = a["summary"], a["ais_only"]
    return meta, det
