"""Outils d'entraînement : garde thermique branchée sur les callbacks Ultralytics."""

from __future__ import annotations

import time
from pathlib import Path

from .thermal import PROM, vram_celsius, wait_if_hot


class ThermalGuard:
    """Met l'entraînement en pause quand la VRAM chauffe trop.

    Appelée à chaque batch, d'entraînement comme de validation. Ne touche ni aux
    ventilateurs ni à la puissance : elle attend que la mémoire refroidisse.
    """

    def __init__(self, log_path: Path | None = None, prom: Path = PROM, poll: float = 5.0, every: int = 1):
        self.log_path = log_path
        self.prom = prom
        self.poll = poll
        self.every = max(1, every)
        self.calls = 0
        self.pauses = 0
        self.paused_s = 0.0
        self.max_seen = 0.0

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [thermique] {msg}"
        print(line, flush=True)
        if self.log_path:
            with Path(self.log_path).open("a") as fh:
                fh.write(line + "\n")

    def check(self, *_args) -> float:
        self.calls += 1
        if self.calls % self.every:
            return 0.0
        temp = vram_celsius(self.prom)
        if temp is not None:
            self.max_seen = max(self.max_seen, temp)
        waited = wait_if_hot(log=self.log, path=self.prom, poll=self.poll)
        if waited > 0:
            self.pauses += 1
            self.paused_s += waited
        return waited

    def attach(self, model) -> None:
        """Branche la garde sur un `ultralytics.YOLO`."""
        for event in ("on_train_batch_end", "on_val_batch_end", "on_train_epoch_start"):
            model.add_callback(event, self.check)

    def summary(self) -> dict:
        return {"pauses": self.pauses, "paused_s": round(self.paused_s, 1), "max_vram_c": self.max_seen}
