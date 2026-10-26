"""L'application web : API JSON et front statique, sur le port 8740."""

from __future__ import annotations

import io
import math
import shutil
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from . import ais, aislive, analyst, detector, jobs, scan, sentinel, series, thermal
from .presets import PRESETS

WEB = Path(__file__).resolve().parents[2] / "web"
MAX_SIDE_DEG = 0.5
MAX_UPLOAD_PX = 8000

app = FastAPI(title="Vigie", version="0.1.0")


@app.middleware("http")
async def no_stale_front(request, call_next):
    """Le front change souvent : le navigateur revalide à chaque chargement, sans garder une vieille version."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

from .sar_api import router as sar_router  # noqa: E402

app.include_router(sar_router)


class ScanRequest(BaseModel):
    bbox: tuple[float, float, float, float]
    name: str | None = Field(default=None, max_length=80)
    scene_id: str | None = None
    days: int = Field(default=30, ge=1, le=365)
    max_cloud: float = Field(default=20, ge=0, le=100)
    confidence: float = Field(default=0.20, ge=0.01, le=0.95)
    analyse: bool = False

    @field_validator("bbox")
    @classmethod
    def check_bbox(cls, b):
        w, s, e, n = b
        if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
            raise ValueError("boîte invalide : ouest < est, sud < nord")
        if e - w > MAX_SIDE_DEG or n - s > MAX_SIDE_DEG:
            raise ValueError(f"zone trop grande, {MAX_SIDE_DEG}° de côté au plus")
        return b


def _job_progress(job: jobs.Job):
    def cb(done, total, msg):
        job.progress = done / total if total else 0
        job.say(msg)
    return cb


@app.get("/api/health")
def health():
    import torch

    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_celsius": thermal.vram_celsius(),
        "vram_pause_at": thermal.PAUSE_AT,
        "model": detector.DEFAULT_MODEL.name,
        "model_ready": detector.DEFAULT_MODEL.exists(),
        "vlm": analyst.MODEL,
        "vlm_ready": analyst.available(),
        "jobs_active": len(jobs.active()),
    }


@app.get("/api/presets")
def presets():
    return PRESETS


@app.get("/api/scenes")
def scenes(west: float, south: float, east: float, north: float, days: int = 60, max_cloud: float = 30):
    ScanRequest(bbox=(west, south, east, north))  # même validation
    return [s.to_dict() for s in sentinel.search((west, south, east, north), days=days, max_cloud=max_cloud,
                                                 limit=20)]


@app.post("/api/scans", status_code=202)
def create_scan(req: ScanRequest):
    def work(job: jobs.Job):
        scene = sentinel.get_scene(req.scene_id) if req.scene_id else None
        job.progress = 0.05
        meta = scan.run(req.bbox, scene=scene, params=detector.DetectParams(confidence=req.confidence),
                        days=req.days, max_cloud=req.max_cloud, name=req.name, log=job.say)
        job.progress = 0.5 if req.analyse else 1.0
        if req.analyse and analyst.available():
            cb = _job_progress(job)
            analyst.analyse(meta["id"], progress=lambda d, t, m: cb(t + d, 2 * t, m))
        return {"scan_id": meta["id"]}

    return jobs.submit("scan", work).to_dict()


@app.post("/api/scans/{scan_id}/analyse", status_code=202)
def analyse_scan(scan_id: str):
    _scan_or_404(scan_id)
    if not analyst.available():
        raise HTTPException(503, f"modèle {analyst.MODEL} introuvable sur {analyst.URL}")

    def work(job: jobs.Job):
        analyst.analyse(scan_id, progress=_job_progress(job))
        return {"scan_id": scan_id}

    return jobs.submit("analyse", work).to_dict()


@app.post("/api/scans/{scan_id}/ais", status_code=202)
def ais_scan(scan_id: str):
    d = _scan_or_404(scan_id)
    meta = scan.load_scan(scan_id)[0]
    if not ais.covers(meta["bbox"]):
        raise HTTPException(422, "AIS ouvert disponible seulement dans les eaux danoises")

    def work(job: jobs.Job):
        job.progress = 0.1
        res = ais.crosscheck(scan_id, log=job.say)
        return {"scan_id": scan_id, "summary": res["summary"], "dir": d.name}

    return jobs.submit("ais", work).to_dict()


class SeriesRequest(BaseModel):
    bbox: tuple[float, float, float, float]
    name: str | None = Field(default=None, max_length=80)
    n: int = Field(default=6, ge=2, le=10)
    days: int = Field(default=90, ge=7, le=365)
    max_cloud: float = Field(default=10, ge=0, le=50)

    _check = field_validator("bbox")(ScanRequest.check_bbox.__func__)


@app.post("/api/series", status_code=202)
def create_series(req: SeriesRequest):
    def work(job: jobs.Job):
        def prog(d, t):
            job.progress = d / t if t else 0
        res = series.run(req.bbox, name=req.name, n=req.n, days=req.days, max_cloud=req.max_cloud,
                         log=job.say, progress=prog)
        return {"series_id": res["id"]}

    return jobs.submit("series", work).to_dict()


@app.get("/api/series")
def get_series(west: float | None = None, south: float | None = None, east: float | None = None,
               north: float | None = None):
    items = series.list_series()
    if west is not None:
        box = (west, south, east, north)
        items = [s for s in items if all(abs(a - b) < 1e-4 for a, b in zip(s["bbox"], box))]
    return items


@app.get("/api/ais-live/status")
def ais_live_status():
    return aislive.collector.state()


@app.get("/api/ais-live")
def ais_live(west: float, south: float, east: float, north: float):
    """Abonne la zone au flux en direct et rend les navires qui y sont déjà connus."""
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90) or east - west > 5 or north - south > 5:
        raise HTTPException(422, "zone invalide, 5° de côté au plus")
    box = (west, south, east, north)
    aislive.collector.watch(box)
    return {"state": aislive.collector.state(), "vessels": aislive.collector.snapshot(box)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "tâche inconnue")
    return job.to_dict()


@app.get("/api/scans")
def list_scans():
    return scan.list_scans()


def _scan_or_404(scan_id: str) -> Path:
    try:
        d = scan.scan_dir(scan_id)
    except ValueError:
        raise HTTPException(400, "identifiant invalide") from None
    if not (d / "scan.json").exists():
        raise HTTPException(404, "scan inconnu")
    return d


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: str):
    d = _scan_or_404(scan_id)
    meta, det = scan.load_scan(scan_id)
    report = d / "report.md"
    return {"meta": meta, "detections": det, "report_md": report.read_text() if report.exists() else None,
            "ais_available": ais.covers(meta["bbox"])}


@app.delete("/api/scans/{scan_id}")
def delete_scan(scan_id: str):
    shutil.rmtree(_scan_or_404(scan_id))
    return {"deleted": scan_id}


@app.get("/api/scans/{scan_id}/image.jpg")
def scan_image(scan_id: str):
    return FileResponse(_scan_or_404(scan_id) / "image.jpg", media_type="image/jpeg")


@app.get("/api/scans/{scan_id}/crops/{vessel_id}.png")
def scan_crop(scan_id: str, vessel_id: str):
    if not vessel_id.isalnum():
        raise HTTPException(400, "identifiant invalide")
    f = _scan_or_404(scan_id) / "crops" / f"{vessel_id}.png"
    if not f.exists():
        raise HTTPException(404, "navire inconnu")
    return FileResponse(f, media_type="image/png")


@app.get("/api/scans/{scan_id}/report.md", response_class=PlainTextResponse)
def scan_report(scan_id: str):
    f = _scan_or_404(scan_id) / "report.md"
    if not f.exists():
        raise HTTPException(404, "pas encore de rapport")
    return f.read_text()


@app.get("/api/scans/{scan_id}/detections.geojson")
def scan_geojson(scan_id: str):
    return FileResponse(_scan_or_404(scan_id) / "detections.geojson", media_type="application/geo+json")


@app.post("/api/detect")
async def detect_upload(image: UploadFile = File(...), confidence: float = Form(0.20), tile: int = Form(640)):
    """Le mode de la démo d'origine : une image quelconque, sans géoréférence."""
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "image illisible") from None
    if max(img.size) > MAX_UPLOAD_PX:
        raise HTTPException(413, f"image trop grande, {MAX_UPLOAD_PX} px de côté au plus")
    params = detector.DetectParams(confidence=min(max(confidence, 0.01), 0.95), tile=max(320, min(int(tile), 1024)))
    params.overlap = min(params.overlap, params.tile // 4)
    result = detector.detect(np.asarray(img), params)
    return {
        "width": img.width, "height": img.height, "tiles": result.tiles, "seconds": round(result.seconds, 3),
        "model": result.model,
        "boxes": [{"x1": round(float(b.x1), 1), "y1": round(float(b.y1), 1), "x2": round(float(b.x2), 1),
                   "y2": round(float(b.y2), 1), "confidence": round(float(b.confidence), 3), "label": b.label,
                   "size_px": round(math.hypot(b.width, b.height), 1)}
                  for b in result.boxes],
    }


def _versioned(page: str):
    """Sert une page HTML en ajoutant ?v=<date de modification> à ses CSS et JS locaux.
    Un navigateur qui a gardé un vieux script en cache est obligé de prendre le nouveau."""
    import re

    from fastapi.responses import HTMLResponse

    def handler():
        html = (WEB / page).read_text()

        def bump(m):
            f = WEB / m.group(2)
            v = int(f.stat().st_mtime) if f.exists() else 0
            return f'{m.group(1)}/{m.group(2)}?v={v}"'

        html = re.sub(r'((?:src|href)=")/([\w.-]+\.(?:js|css))"', bump, html)
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    return handler


for _page, _route in (("index.html", "/"), ("index.html", "/index.html")):
    app.add_api_route(_route, _versioned(_page), include_in_schema=False)

if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
