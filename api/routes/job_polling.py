"""
GET /jobs/{job_id}

Generic polling endpoint for any background job (train, shap, future
job types). Frontend polls this until status is "complete" or "failed",
then fetches the actual results from the job-type-specific endpoint
(e.g. GET /csv/{session_id}/metric-frames).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .jobs import job_store
from .train_models import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """
    Poll the status of a background job.

    status transitions: queued -> running -> complete | failed.
    completed_steps/total_steps give a coarse progress signal
    (e.g. "3/6 bundles trained") suitable for a progress bar.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return JobResponse(
        job_id=job.job_id,
        session_id=job.session_id,
        job_type=job.job_type,
        status=job.status.value,
        total_steps=job.total_steps,
        completed_steps=job.completed_steps,
        error=job.error,
    )