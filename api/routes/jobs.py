"""
In-memory job tracking for long-running operations (train, shap).

Separate from SessionStore because jobs are polled by job_id alone
(GET /jobs/{job_id}) without needing to know which session spawned them.
A job references its session_id so handlers can write results back into
the right Session once finished.

Built on FastAPI's BackgroundTasks, which runs in-process. This means:
  - Jobs do NOT survive a process restart.
  - Jobs do NOT distribute across multiple worker processes.
Both are known limitations — fine for a single-worker dev/demo deployment,
but this should be swapped for Celery/ARQ + Redis before any real traffic.
"""
from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    session_id: str
    job_type: str  # "train" | "shap"
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    # Coarse progress signal for the UI — "3/6 bundles trained".
    # Not granular per-fold; that level of detail isn't worth the
    # complexity for a progress bar.
    total_steps: int = 0
    completed_steps: int = 0

    error: str | None = None
    error_traceback: str | None = None

    def mark_running(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = time.time()

    def mark_progress(self, completed_steps: int) -> None:
        self.completed_steps = completed_steps

    def mark_complete(self) -> None:
        self.status = JobStatus.COMPLETE
        self.finished_at = time.time()
        self.completed_steps = self.total_steps

    def mark_failed(self, exc: Exception) -> None:
        self.status = JobStatus.FAILED
        self.finished_at = time.time()
        self.error = str(exc)
        self.error_traceback = traceback.format_exc()


class JobStore:
    """Simple in-memory job registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, session_id: str, job_type: str, total_steps: int = 0) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            session_id=session_id,
            job_type=job_type,
            total_steps=total_steps,
        )
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


# Module-level singleton — imported by route handlers and background runners.
job_store = JobStore()