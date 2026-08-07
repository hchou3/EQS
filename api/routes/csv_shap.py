"""
POST /csv/{session_id}/shap
GET  /csv/{session_id}/shap/{protected_attr}/{target_col}

SHAP requires a trained bundle to already exist (POST /csv/train must
have completed first) — compute_shap() refits its own model internally,
but it needs the bundle (sensitive_features, y_true, task_type) which
only exists after prepare_fairlearn_data() + train_bundle() have run.

Like /train, this is job-based: SHAP's TreeExplainer + full-data refit
is the second-most expensive operation in the pipeline after CV training
itself, so it gets the same queued/poll treatment.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from .jobs import job_store, JobStatus
from .sessions import session_store
from .train_models import ShapFeatureImportance, ShapResultResponse, ShapStartResponse

from ..ai.csv.csv_training import _make_lgbm, compute_shap

router = APIRouter(prefix="/csv", tags=["csv"])


def _bundle_key(protected_attr: str, target_col: str) -> str:
    return f"{protected_attr}::{target_col}"


def _run_shap_job(job_id: str, session_id: str, protected_attr: str, target_col: str) -> None:
    """Background task body. Computes SHAP for a single already-trained bundle."""
    job = job_store.get(job_id)
    session = session_store.get(session_id)
    if job is None or session is None:
        return

    job.mark_running()

    try:
        key = _bundle_key(protected_attr, target_col)
        bundle_result = session.bundle_results.get(key)
        if bundle_result is None:
            raise ValueError(
                f"No trained bundle found for '{protected_attr}' x '{target_col}'. "
                f"Run POST /csv/{session_id}/train first."
            )

        bundle = bundle_result.bundle
        X = session.csv_data.df[session.csv_data.encoded_feature_cols]

        # compute_shap() clones whatever estimator we pass and refits it on
        # full data internally — build a fresh, unfitted one sized to this
        # bundle, mirroring what train_bundle() did.
        estimator = _make_lgbm(bundle.task_type, n_samples=len(bundle.y_true))

        shap_result = compute_shap(bundle, estimator, X)
        session.shap_results[key] = shap_result

        job.mark_progress(1)
        job.mark_complete()

    except Exception as e:
        job.mark_failed(e)


@router.post(
    "/{session_id}/shap",
    response_model=ShapStartResponse,
    responses={404: {"description": "Session or trained bundle not found"}},
)
async def start_shap(
    session_id: str,
    protected_attr: str,
    target_col: str,
    background_tasks: BackgroundTasks,
) -> ShapStartResponse:
    """
    Kick off SHAP computation for one already-trained bundle.

    Requires POST /csv/{session_id}/train to have completed first —
    SHAP explains the model trained for a specific (protected_attr,
    target_col) pair, so that bundle must already exist in
    session.bundle_results.
    """
    session = session_store.get(session_id)
    if session is None or session.csv_data is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    key = _bundle_key(protected_attr, target_col)
    if key not in session.bundle_results:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trained bundle found for protected_attr='{protected_attr}', "
                f"target_col='{target_col}'. Run POST /csv/{session_id}/train "
                f"and wait for it to complete before requesting SHAP."
            ),
        )

    job = job_store.create(session_id=session_id, job_type="shap", total_steps=1)
    background_tasks.add_task(_run_shap_job, job.job_id, session_id, protected_attr, target_col)

    return ShapStartResponse(
        job_id=job.job_id,
        session_id=session_id,
        status=JobStatus.QUEUED.value,
        message=f"SHAP computation queued. Poll GET /jobs/{job.job_id} for status.",
    )


@router.get(
    "/{session_id}/shap/{protected_attr}/{target_col}",
    response_model=ShapResultResponse,
    responses={404: {"description": "SHAP results not found for this pair"}},
)
async def get_shap_result(
    session_id: str,
    protected_attr: str,
    target_col: str,
    top_n: int = 15,
) -> ShapResultResponse:
    """
    Return SHAP feature importances for an already-completed SHAP job.

    top_features is ranked by global importance; per_group importances
    on each feature let the frontend surface proxy signals — a feature
    with low global importance but high importance in one group suggests
    the model is using it as a stand-in for the protected attribute.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    key = _bundle_key(protected_attr, target_col)
    shap_result = session.shap_results.get(key)
    if shap_result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No SHAP results found for '{protected_attr}' x '{target_col}'. "
                f"Run POST /csv/{session_id}/shap first and wait for completion."
            ),
        )

    top_features = shap_result.global_importances.head(top_n)
    features: list[ShapFeatureImportance] = []
    for feature_name, global_val in top_features.items():
        per_group = {
            group: round(float(importances.get(feature_name, 0.0)), 4)
            for group, importances in shap_result.per_group_importances.items()
        }
        features.append(
            ShapFeatureImportance(
                feature=feature_name,
                global_importance=round(float(global_val), 4),
                per_group=per_group,
            )
        )

    return ShapResultResponse(
        session_id=session_id,
        protected_attr=protected_attr,
        target_col=target_col,
        top_features=features,
        n_samples=shap_result.n_samples,
    )