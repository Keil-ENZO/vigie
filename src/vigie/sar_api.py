"""Routes radar, à inclure dans l'application : `app.include_router(sar_api.router)`.

    POST /api/scans-sar   même corps que POST /api/scans, rend une tâche
    GET  /api/scenes-sar  passes Sentinel-1 sur une zone

Le scan radar écrit le même dossier qu'un scan optique : la suite, de
`GET /api/scans/{id}` au croisement AIS, ne change pas.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from . import analyst, cfar, jobs, sar, scan_sar

router = APIRouter()
MAX_SIDE_DEG = 0.5


class SarScanRequest(BaseModel):
    """Les champs de `api.ScanRequest`. `max_cloud` est accepté et ignoré, `confidence` aussi :
    le radar n'a pas de nuages, et le CFAR règle son seuil en écarts-types, pas en score."""

    bbox: tuple[float, float, float, float]
    name: str | None = Field(default=None, max_length=80)
    scene_id: str | None = None
    days: int = Field(default=12, ge=1, le=365)
    max_cloud: float = Field(default=20, ge=0, le=100)
    confidence: float = Field(default=0.20, ge=0.01, le=0.95)
    analyse: bool = False
    k: float | None = Field(default=None, ge=2.0, le=12.0)

    @field_validator("bbox")
    @classmethod
    def check_bbox(cls, b):
        w, s, e, n = b
        if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
            raise ValueError("boîte invalide : ouest < est, sud < nord")
        if e - w > MAX_SIDE_DEG or n - s > MAX_SIDE_DEG:
            raise ValueError(f"zone trop grande, {MAX_SIDE_DEG}° de côté au plus")
        return b


@router.get("/api/scenes-sar")
def scenes_sar(west: float, south: float, east: float, north: float, days: int = 30):
    SarScanRequest(bbox=(west, south, east, north))  # même validation
    return [s.to_dict() for s in sar.search((west, south, east, north), days=min(max(days, 1), 365), limit=50)]


@router.post("/api/scans-sar", status_code=202)
def create_scan_sar(req: SarScanRequest):
    def work(job: jobs.Job):
        scene = sar.get_scene(req.scene_id) if req.scene_id else None
        job.progress = 0.05
        params = cfar.CfarParams() if req.k is None else cfar.CfarParams(k=req.k)
        meta = scan_sar.run(req.bbox, scene=scene, params=params, days=req.days, name=req.name, log=job.say)
        job.progress = 0.5 if req.analyse else 1.0
        if req.analyse and analyst.available():
            def cb(done, total, msg):
                job.progress = (total + done) / (2 * total) if total else 0
                job.say(msg)

            analyst.analyse(meta["id"], progress=cb)
        return {"scan_id": meta["id"]}

    return jobs.submit("scan-sar", work).to_dict()
