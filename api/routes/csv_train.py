"""
POST /csv/{session_id}/train
GET  /csv/{session_id}/metric-frames

Train kicks off prepare_fairlearn_data() + train_bundle() for every
non-skipped bundle, as a background job. The frontend polls
GET /jobs/{job_id} until status is "complete", then calls
GET /csv/{session_id}/metric-frames for chart-ready results.

Design note: prepare_fairlearn_data() runs synchronously inside the route
handler (before queuing the background task), not inside the background
task itself. It's fast — no model training, just DataFrame slicing and
sparsity checks — so there's no benefit to deferring it, and doing it
eagerly lets us return total_bundles/skipped_bundles in the immediate
response instead of making the frontend wait for the first poll.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from .jobs import job_store, JobStatus
from .sessions import session_store
from .train_models import (
    BundleTrainSummary,
    GroupMetricPoint,
    MetricFrameChart,
    MetricFramesResponse,
    TrainStartResponse,
)

from ..ai.csv.csv_processing import prepare_fairlearn_data
from ..ai.csv.csv_training import train_bundle

router = APIRouter(prefix="/csv", tags=["csv"])


def _bundle_key(protected_attr: str, target_col: str) -> str:
    return f"{protected_attr}::{target_col}"


def _run_training_job(job_id: str, session_id: str) -> None:
    """
    Background task body. Runs train_bundle() for every non-skipped bundle
    in the session's FairlearnDataset, writing BundleResults back into the
    session as each one completes (so partial results are visible even if
    a later bundle fails).
    """
    job = job_store.get(job_id)
    session = session_store.get(session_id)
    if job is None or session is None:
        return  # Session/job vanished — nothing to do.

    job.mark_running()

    try:
        dataset = session.fairlearn_dataset
        X = session.csv_data.df[session.csv_data.encoded_feature_cols]

        completed = 0
        for bundle in dataset.bundles:
            if bundle.skipped:
                continue
            result = train_bundle(bundle, X)
            key = _bundle_key(bundle.protected_attr, bundle.target_col)
            session.bundle_results[key] = result
            completed += 1
            job.mark_progress(completed)

        job.mark_complete()

    except Exception as e:
        job.mark_failed(e)


@router.post(
    "/{session_id}/train",
    response_model=TrainStartResponse,
    responses={404: {"description": "Session not found"}},
)
async def start_training(
    session_id: str,
    background_tasks: BackgroundTasks,
    mode: str = Query(default="individual", pattern="^(individual|intersectional)$"),
) -> TrainStartResponse:
    """
    Kick off model training for every (protected_attr, target_col) bundle.

    This is async/job-based because training time scales with
    bundle_count x cv_folds, which can exceed typical request timeouts
    on datasets with several protected attributes and targets.

    Returns immediately with a job_id. Poll GET /jobs/{job_id} until
    status is "complete", then call GET /csv/{session_id}/metric-frames.
    """
    session = session_store.get(session_id)
    if session is None or session.csv_data is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a CSV via POST /csv/upload first.",
        )

    if not session.csv_data.encoded_feature_cols:
        raise HTTPException(
            status_code=422,
            detail=(
                "No encoded feature columns found for this session. "
                "encode_features() may have failed during upload — "
                "check the `warnings` field from the original upload response."
            ),
        )

    try:
        dataset = prepare_fairlearn_data(session.csv_data, mode=mode)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    session.fairlearn_dataset = dataset

    trainable_bundles = [b for b in dataset.bundles if not b.skipped]
    skipped_count = len(dataset.bundles) - len(trainable_bundles)

    if not trainable_bundles:
        raise HTTPException(
            status_code=422,
            detail=(
                "All bundles were skipped due to sparse groups. "
                "No model can be trained for this dataset/mode combination."
            ),
        )

    job = job_store.create(
        session_id=session_id,
        job_type="train",
        total_steps=len(trainable_bundles),
    )

    background_tasks.add_task(_run_training_job, job.job_id, session_id)

    return TrainStartResponse(
        job_id=job.job_id,
        session_id=session_id,
        status=JobStatus.QUEUED.value,
        total_bundles=len(dataset.bundles),
        skipped_bundles=skipped_count,
        message=f"Training queued for {len(trainable_bundles)} bundle(s). "
                f"Poll GET /jobs/{job.job_id} for status.",
    )


def _metric_frame_to_charts(bundle, metric_frame) -> list[MetricFrameChart]:
    """
    Convert a fairlearn MetricFrame into Recharts-ready charts, one per
    metric (accuracy, precision, ... or mae, r2, ...).

    MetricFrame.by_group is a DataFrame: index = group, columns = metric names.
    MetricFrame.overall is a Series: index = metric names.
    """
    charts: list[MetricFrameChart] = []
    by_group = metric_frame.by_group
    overall = metric_frame.overall

    for metric_name in by_group.columns:
        points = [
            GroupMetricPoint(group=str(group), value=round(float(value), 4))
            for group, value in by_group[metric_name].items()
        ]
        charts.append(
            MetricFrameChart(
                protected_attr=bundle.protected_attr,
                target_col=bundle.target_col,
                task_type=bundle.task_type,
                metric_name=metric_name,
                overall=round(float(overall[metric_name]), 4),
                data=points,
            )
        )

    return charts


@router.get(
    "/{session_id}/metric-frames",
    response_model=MetricFramesResponse,
    responses={404: {"description": "Session not found or training not yet complete"}},
)
async def get_metric_frames(session_id: str) -> MetricFramesResponse:
    """
    Return chart-ready MetricFrame results for every trained bundle.

    Call this only after the training job (started via POST /train) has
    status "complete". Returns whatever bundles have finished — if you
    poll mid-training, you'll get a partial list rather than an error,
    since bundle_results is populated incrementally.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not session.bundle_results:
        raise HTTPException(
            status_code=404,
            detail=(
                "No trained bundles found for this session. "
                "Call POST /csv/{session_id}/train first and wait for "
                "the job to complete."
            ),
        )

    summaries: list[BundleTrainSummary] = []
    for key, result in session.bundle_results.items():
        bundle = result.bundle
        try:
            charts = _metric_frame_to_charts(bundle, result.metric_frame)
            summaries.append(
                BundleTrainSummary(
                    protected_attr=bundle.protected_attr,
                    target_col=bundle.target_col,
                    task_type=bundle.task_type,
                    n_folds=result.n_folds,
                    n_samples=len(result.y_pred_oof),
                    charts=charts,
                )
            )
        except Exception as e:
            summaries.append(
                BundleTrainSummary(
                    protected_attr=bundle.protected_attr,
                    target_col=bundle.target_col,
                    task_type=bundle.task_type,
                    n_folds=result.n_folds,
                    n_samples=len(result.y_pred_oof),
                    charts=[],
                    error=f"Could not build charts: {e}",
                )
            )

    return MetricFramesResponse(session_id=session_id, bundles=summaries)