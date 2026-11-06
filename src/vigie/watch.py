"""Veille d'une zone : une nouvelle passe, un scan, une notification ntfy.

Une exécution vérifie une fois, à cron de répéter. L'AIS danois sort avec du retard :
tant qu'il manque, la passe reste en attente et la prochaine exécution réessaie.
Hors de sa couverture, un navire sombre ne se distingue pas : on annonce le décompte.
"""

from __future__ import annotations

import json

import httpx

from . import ais, scan, sentinel

STATE = scan.DATA / "watch.json"
MIN_COVERAGE = 0.9
MAX_LISTED = 5


def _load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(key: str, entry: dict) -> None:
    state = _load() | {key: entry}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def notify(url: str, title: str, message: str) -> None:
    httpx.post(url, content=message.encode(), headers={"Title": title}, timeout=15).raise_for_status()


def _message(meta: dict, name: str | None, dark: list[dict] | None) -> str:
    zone, day, n = name or "Zone suivie", meta["scene"]["datetime"][:10], meta["counts"]["vessels"]
    if dark is None:
        return f"{zone}, passe du {day} : {n} navires, identités inconnues hors de l'AIS danois."
    lines = [f"{zone}, passe du {day} : {len(dark)} sans AIS sur {n} navires."]
    lines += [f"{p['id']} : {p['length_m']:.0f} m, {p['lat']:.4f}, {p['lon']:.4f}" for p in dark[:MAX_LISTED]]
    return "\n".join(lines)


def check(bbox, name: str | None = None, ntfy: str | None = None, days: int = 7, max_cloud: float = 20,
          log=print) -> dict | None:
    """Rend le résultat de la passe traitée, ou None si rien de neuf n'est prêt."""
    key = ",".join(str(v) for v in bbox)
    scenes = [s for s in sentinel.search(bbox, days=days, max_cloud=max_cloud) if (s.coverage or 0) >= MIN_COVERAGE]
    if not scenes:
        log("aucune passe claire et complète sur la période")
        return None
    scene = max(scenes, key=lambda s: s.datetime)
    seen = _load().get(key, {})
    if seen.get("scene") == scene.id and not seen.get("pending"):
        log(f"rien de neuf depuis {scene.id}")
        return None
    scan_id = seen["scan_id"] if seen.get("scene") == scene.id else scan.run(bbox, scene=scene, name=name, log=log)["id"]

    dark = None
    if ais.covers(bbox):
        try:
            ais.crosscheck(scan_id, log=log)
        except LookupError as exc:
            log(f"passe en attente : {exc}")
            _save(key, {"scene": scene.id, "scan_id": scan_id, "pending": True})
            return None
        _, det = scan.load_scan(scan_id)
        dark = [f["properties"] for f in det["features"] if f["properties"].get("ais", {}).get("status") == "dark"]

    meta, _ = scan.load_scan(scan_id)
    message = _message(meta, name, dark)
    if ntfy and (dark is None or dark):
        notify(ntfy, "Vigie", message)
    _save(key, {"scene": scene.id, "scan_id": scan_id, "pending": False})
    return {"scan_id": scan_id, "scene": scene.id, "dark": None if dark is None else len(dark), "message": message}
