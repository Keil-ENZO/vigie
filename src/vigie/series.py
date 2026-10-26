"""Une zone suivie sur plusieurs passes : combien de navires, qui reste, qui arrive.

Deux passes sont comparées navire par navire, par le même appariement hongrois que
l'AIS. Un navire retrouvé à moins de 250 m d'une passe à l'autre est « resté » :
au mouillage ou à quai. Les autres sont arrivés ou partis entre les deux.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from . import ais, detector, scan, sentinel

SERIES = scan.DATA / "series"
STAY_M = 250


def _vessels(scan_id: str) -> list[dict]:
    _, det = scan.load_scan(scan_id)
    return [f for f in det["features"] if not f["properties"].get("fixed")]


def compare(a_id: str, b_id: str) -> dict:
    a, b = _vessels(a_id), _vessels(b_id)
    pts = [{"lat": f["properties"]["lat"], "lon": f["properties"]["lon"]} for f in b]
    pairs = ais.match(a, pts, gate=STAY_M)
    return {"stayed": len(pairs), "left": len(a) - len(pairs), "arrived": len(b) - len(pairs),
            "stayed_ids": [[a[r]["properties"]["id"], b[c]["properties"]["id"]] for r, c, _ in pairs]}


def _existing(bbox, scene_id: str, confidence: float) -> dict | None:
    """Un scan déjà fait sur la même zone, la même passe et le même seuil."""
    for m in scan.list_scans():
        if (m["scene"]["id"] == scene_id and m["params"]["confidence"] == confidence
                and all(abs(a - b) < 1e-6 for a, b in zip(m["bbox"], bbox)) and m["model"] == detector.DEFAULT_MODEL.name):
            return m
    return None


def run(bbox, name: str | None = None, n: int = 6, days: int = 90, max_cloud: float = 10.0,
        confidence: float = 0.20, log=print, progress=lambda done, total: None) -> dict:
    scenes = sentinel.search(bbox, days=days, max_cloud=max_cloud, limit=50)
    by_day: dict[str, sentinel.Scene] = {}
    for sc in scenes:  # une passe par jour, la mieux couverte, puis la moins nuageuse
        day = sc.datetime[:10]
        best = by_day.get(day)
        if best is None or ((sc.coverage or 0), -sc.cloud_cover) > ((best.coverage or 0), -best.cloud_cover):
            by_day[day] = sc
    chosen = sorted(by_day.values(), key=lambda s: s.datetime, reverse=True)[:n]
    chosen = [s for s in chosen if (s.coverage or 0) >= 0.9]
    if len(chosen) < 2:
        raise LookupError("moins de deux passes claires et complètes sur la période")
    chosen.sort(key=lambda s: s.datetime)
    steps = []
    for i, sc in enumerate(chosen):
        progress(i, len(chosen))
        log(f"passe {i + 1}/{len(chosen)} : {sc.datetime[:10]}")
        meta = _existing(bbox, sc.id, confidence) or scan.run(
            bbox, scene=sc, params=detector.DetectParams(confidence=confidence), name=name, log=log)
        steps.append({"scan_id": meta["id"], "datetime": sc.datetime, "cloud_cover": sc.cloud_cover,
                      "vessels": meta["counts"]["vessels"]})
    for prev, cur in zip(steps, steps[1:]):
        cur["vs_previous"] = {k: v for k, v in compare(prev["scan_id"], cur["scan_id"]).items() if k != "stayed_ids"}
    sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    out = {"id": sid, "name": name, "bbox": list(bbox), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "steps": steps}
    SERIES.mkdir(parents=True, exist_ok=True)
    (SERIES / f"{sid}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    progress(len(chosen), len(chosen))
    return out


def list_series() -> list[dict]:
    if not SERIES.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(SERIES.glob("*.json"), reverse=True)]
