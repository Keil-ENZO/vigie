"""Tâches longues en arrière-plan. Une seule à la fois : elles partagent le GPU."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vigie-job")
_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()
MAX_KEPT = 100


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued, running, done, error
    progress: float = 0.0
    message: str = "en file d'attente"
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def say(self, msg: str) -> None:
        self.message = msg
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "kind", "status", "progress", "message", "log", "result",
                                              "error", "created", "finished")}


def submit(kind: str, fn: Callable[[Job], Any]) -> Job:
    job = Job(id=uuid.uuid4().hex[:10], kind=kind)
    with _lock:
        _jobs[job.id] = job
        if len(_jobs) > MAX_KEPT:
            for old in sorted(_jobs.values(), key=lambda j: j.created)[: len(_jobs) - MAX_KEPT]:
                if old.status in ("done", "error"):
                    _jobs.pop(old.id, None)

    def wrapper():
        job.status = "running"
        try:
            job.result = fn(job)
            job.status, job.progress = "done", 1.0
        except Exception as exc:  # noqa: BLE001 - l'erreur remonte au client
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            job.say(job.error)
            traceback.print_exc()
        finally:
            job.finished = time.time()

    _pool.submit(wrapper)
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def active() -> list[Job]:
    return [j for j in _jobs.values() if j.status in ("queued", "running")]
