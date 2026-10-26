"""Garde thermique de l'entraînement : elle bloque à chaud, laisse passer à froid."""

import threading
import time

from vigie.training import ThermalGuard


def _write(path, temp):
    path.write_text(f'gputemps_vram_celsius{{gpu="0"}} {temp}\n')


def test_guard_passes_when_cool(tmp_path):
    prom = tmp_path / "gputemps.prom"
    _write(prom, 55)
    g = ThermalGuard(prom=prom, poll=0.05)
    assert g.check() == 0.0
    assert g.summary()["pauses"] == 0


def test_guard_blocks_until_resume(tmp_path):
    prom = tmp_path / "gputemps.prom"
    log = tmp_path / "thermal.log"
    _write(prom, 90)

    def cool():
        time.sleep(0.3)
        _write(prom, 70)  # encore au-dessus de la reprise : on reste en pause
        time.sleep(0.3)
        _write(prom, 58)

    threading.Thread(target=cool).start()
    g = ThermalGuard(log_path=log, prom=prom, poll=0.05)
    waited = g.check()
    assert waited >= 0.5
    assert g.summary()["pauses"] == 1
    assert "pause" in log.read_text() and "reprise" in log.read_text()


def test_guard_survives_missing_file(tmp_path):
    g = ThermalGuard(prom=tmp_path / "absent.prom")
    assert g.check() == 0.0
