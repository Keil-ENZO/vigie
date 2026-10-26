"""AIS en direct, par aisstream.io : la position actuelle des navires, partout.

L'image satellite a deux ou trois jours de retard. L'AIS en direct dit où sont les
navires maintenant. Une seule connexion WebSocket, dans un thread, abonnée aux
zones que l'interface regarde. Les positions vivent en mémoire, pas sur disque.

La clé API est gratuite. Elle se lit dans la variable VIGIE_AISSTREAM_KEY, ou dans
le fichier ~/.config/vigie/aisstream.key. Elle ne quitte jamais le serveur.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

URL = "wss://stream.aisstream.io/v0/stream"
KEY_FILE = Path(os.environ.get("VIGIE_AISSTREAM_KEY_FILE", Path.home() / ".config" / "vigie" / "aisstream.key"))
MAX_BOXES = 4  # zones suivies en même temps ; la plus ancienne sort
TTL_S = 30 * 60  # un navire muet depuis 30 min disparaît
WATCH_TTL_S = 15 * 60  # une zone que personne ne regarde plus est désabonnée
MESSAGES = ["PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport", "ShipStaticData",
            "StaticDataReport"]


def ship_category(code: int | None) -> str:
    """Code de type AIS vers les catégories de Vigie."""
    if code is None:
        return "unknown"
    if code == 30:
        return "fishing"
    if code == 35:
        return "military"
    if code in (36, 37):
        return "pleasure"
    if code in (31, 32, 33, 34) or 50 <= code <= 59:
        return "service"
    if 40 <= code <= 49 or 60 <= code <= 69:
        return "passenger"
    if 70 <= code <= 79:
        return "cargo"
    if 80 <= code <= 89:
        return "tanker"
    return "unknown"


def read_key() -> str | None:
    key = os.environ.get("VIGIE_AISSTREAM_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    return key or None


class Collector:
    def __init__(self):
        self.vessels: dict[str, dict] = {}
        self.boxes: dict[tuple, float] = {}  # boîte → dernier regard
        self.lock = threading.Lock()
        self.status = "arrêté"
        self.error: str | None = None
        self.messages = 0
        self.started: float | None = None
        self._thread: threading.Thread | None = None
        self._resubscribe = threading.Event()

    # --- côté API ---

    def watch(self, bbox) -> None:
        box = tuple(round(v, 4) for v in bbox)
        with self.lock:
            self.boxes[box] = time.time()
            if len(self.boxes) > MAX_BOXES:
                oldest = min(self.boxes, key=self.boxes.get)
                del self.boxes[oldest]
        self._resubscribe.set()
        self._ensure_running()

    def snapshot(self, bbox) -> list[dict]:
        w, s, e, n = bbox
        now = time.time()
        with self.lock:
            for k in [k for k, v in self.vessels.items() if now - v["seen"] > TTL_S]:
                del self.vessels[k]
            return [v | {"age_s": round(now - v["seen"])} for v in self.vessels.values()
                    if "lat" in v and w <= v["lon"] <= e and s <= v["lat"] <= n]

    def state(self) -> dict:
        return {"configured": read_key() is not None, "status": self.status, "error": self.error,
                "messages": self.messages, "vessels": len(self.vessels), "zones": len(self.boxes),
                "since": self.started}

    # --- le thread ---

    def _ensure_running(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if read_key() is None:
            self.status, self.error = "pas de clé", f"clé absente : {KEY_FILE}"
            return
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True, name="vigie-aislive")
        self._thread.start()

    def _subscription(self, key: str) -> str:
        now = time.time()
        with self.lock:
            for b in [b for b, t in self.boxes.items() if now - t > WATCH_TTL_S]:
                del self.boxes[b]
            # aisstream attend [[lat, lon], [lat, lon]]
            boxes = [[[s, w], [n, e]] for (w, s, e, n) in self.boxes]
        return json.dumps({"APIKey": key, "BoundingBoxes": boxes, "FilterMessageTypes": MESSAGES})

    async def _run(self) -> None:
        import websockets

        delay = 2
        while True:
            key = read_key()
            if key is None or not self.boxes:
                self.status = "en veille"
                await asyncio.sleep(5)
                continue
            try:
                self.status, self.error = "connexion", None
                async with websockets.connect(URL, open_timeout=15, max_size=2**22) as ws:
                    await ws.send(self._subscription(key))
                    self._resubscribe.clear()
                    self.status, self.started, delay = "connecté", time.time(), 2
                    last_sub = time.time()
                    while True:
                        if self._resubscribe.is_set() and time.time() - last_sub > 1.2:
                            self._resubscribe.clear()
                            if not self.boxes:
                                break
                            await ws.send(self._subscription(key))
                            last_sub = time.time()
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        self._ingest(json.loads(raw))
            except Exception as exc:  # noqa: BLE001 - réseau, clé refusée : on réessaie plus tard
                self.status, self.error = "déconnecté", f"{type(exc).__name__}: {exc}"[:200]
                await asyncio.sleep(delay)
                delay = min(delay * 2, 120)

    def _ingest(self, msg: dict) -> None:
        if "error" in msg:
            self.error = str(msg["error"])[:200]
            return
        kind, meta = msg.get("MessageType"), msg.get("MetaData", {})
        body = msg.get("Message", {}).get(kind, {})
        mmsi = str(meta.get("MMSI") or body.get("UserID") or "")
        if not mmsi:
            return
        self.messages += 1
        with self.lock:
            v = self.vessels.setdefault(mmsi, {"mmsi": mmsi})
            v["seen"] = time.time()
            if name := (meta.get("ShipName") or "").strip():
                v["name"] = name
            if "Position" in kind:
                lat, lon = meta.get("latitude"), meta.get("longitude")
                if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
                    return
                v.update(lat=round(lat, 6), lon=round(lon, 6), time_utc=meta.get("time_utc"),
                         sog_kn=body.get("Sog"), cog_deg=body.get("Cog"),
                         heading_deg=None if body.get("TrueHeading") in (None, 511) else body.get("TrueHeading"),
                         nav_status=body.get("NavigationalStatus"), cls="A" if kind == "PositionReport" else "B")
            elif kind in ("ShipStaticData", "StaticDataReport"):
                rep = body.get("ReportB", body) if kind == "StaticDataReport" else body
                code = rep.get("Type", rep.get("ShipType"))
                dim = rep.get("Dimension") or {}
                if dim:
                    v["length_m"] = (dim.get("A") or 0) + (dim.get("B") or 0) or v.get("length_m")
                    v["width_m"] = (dim.get("C") or 0) + (dim.get("D") or 0) or v.get("width_m")
                if code:
                    v["ship_type"], v["category"] = code, ship_category(code)
                if dest := (rep.get("Destination") or "").strip():
                    v["destination"] = dest
                if n := (rep.get("Name") or "").strip():
                    v["name"] = n


collector = Collector()
