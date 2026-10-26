"""Garde thermique de la VRAM de la 3090.

La GDDR6X chauffe vite. On ne touche ni aux ventilateurs ni à la puissance : on
attend. Les seuils sont ceux de hijackbench, mesurés au J0.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

PROM = Path(os.environ.get("VIGIE_GPUTEMPS_PROM", "/var/lib/node_exporter/textfile/gputemps.prom"))
PAUSE_AT = float(os.environ.get("VIGIE_VRAM_PAUSE", 84))
RESUME_AT = float(os.environ.get("VIGIE_VRAM_RESUME", 60))


def vram_celsius(path: Path = PROM) -> float | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("gputemps_vram_celsius"):
                return float(line.split()[-1])
    except (OSError, ValueError):
        return None
    return None


def wait_if_hot(log=print, path: Path = PROM, poll: float = 5.0) -> float:
    """Bloque tant que la VRAM est trop chaude. Rend le temps passé en pause."""
    temp = vram_celsius(path)
    if temp is None or temp < PAUSE_AT:
        return 0.0
    log(f"VRAM à {temp:.0f} °C, pause jusqu'à {RESUME_AT:.0f} °C")
    start = time.monotonic()
    while (temp := vram_celsius(path)) is not None and temp > RESUME_AT:
        time.sleep(poll)
    waited = time.monotonic() - start
    log(f"VRAM à {temp} °C, reprise après {waited:.0f} s")
    return waited
