"""
Models for POST /csv/{session_id}/train, GET /jobs/{job_id},
GET /csv/{session_id}/metric-frames, and POST /csv/{session_id}/shap.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Job polling — shared across train and shap
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    job_id: str
    session_id: str
    job_type: str
    status: str  # "queued" | "running" | "complete" | "failed"
    total_steps: int
    completed_steps: int
    error: str | None = None


class TrainStartResponse(BaseModel):
    job_id: str
    session_id: str
    status: str
    total_bundles: int
    skipped_bundles: int
    message: str


# ---------------------------------------------------------------------------
# Metric frame results — Recharts-ready, mirrors bias_metrics_models shape
# ---------------------------------------------------------------------------

class GroupMetricPoint(BaseModel):
    """One bar in a per-group metric chart, e.g. recall by race."""
    group: str
    value: float


class MetricFrameChart(BaseModel):
    protected_attr: str
    target_col: str
    task_type: str
    metric_name: str  # "accuracy" | "precision" | ... | "mae" | "r2" | ...
    overall: float = Field(description="Metric computed over all groups combined.")
    data: list[GroupMetricPoint]


class BundleTrainSummary(BaseModel):
    protected_attr: str
    target_col: str
    task_type: str
    n_folds: int
    n_samples: int
    charts: list[MetricFrameChart]
    error: str | None = None


class MetricFramesResponse(BaseModel):
    session_id: str
    bundles: list[BundleTrainSummary]


# ---------------------------------------------------------------------------
# SHAP results
# ---------------------------------------------------------------------------

class ShapFeatureImportance(BaseModel):
    feature: str
    global_importance: float
    per_group: dict[str, float] = Field(
        description="Mean absolute SHAP value for this feature, keyed by group."
    )


class ShapStartResponse(BaseModel):
    job_id: str
    session_id: str
    status: str
    message: str


class ShapResultResponse(BaseModel):
    session_id: str
    protected_attr: str
    target_col: str
    top_features: list[ShapFeatureImportance]
    n_samples: int