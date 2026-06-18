"""
Pydantic models for request/response validation.

Kept separate from route handlers so the contracts are easy to review,
version, and reuse across endpoints (e.g. the bias-metrics chart shapes
will be reused by both /upload and /bias-metrics).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared chart-data shapes
# ---------------------------------------------------------------------------

class GroupDisparityPoint(BaseModel):
    """One row of chart-ready disparity data for a bar/scatter chart."""
    protected_attr: str
    target_col: str
    group: str
    value: float
    metric_name: str = Field(
        description="What `value` represents, e.g. 'outcome_rate', 'mean'."
    )


class DisparitySummary(BaseModel):
    """Top-level disparity number for one (protected_attr, target_col) pair."""
    protected_attr: str
    target_col: str
    task_type: str
    disparity: float | None = None
    p_value: float | None = Field(
        default=None,
        description="ANOVA p-value for regression targets; null for classification.",
    )
    error: str | None = None


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

class ColumnReasoning(BaseModel):
    protected_attributes_explanation: str = ""
    target_columns_explanation: str = ""
    domain_assessment: str = ""


class FavorableDirectionInfo(BaseModel):
    direction: str  # "higher" | "lower" | "neutral"
    rationale: str = ""
    over_predict_consequence: str = ""
    under_predict_consequence: str = ""
    confidence: str = "low"


class LLMClassification(BaseModel):
    protected_attributes: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    target_column_types: dict[str, str] = Field(default_factory=dict)
    reasoning: ColumnReasoning = Field(default_factory=ColumnReasoning)
    regression_favorable_directions: dict[str, FavorableDirectionInfo] = Field(
        default_factory=dict
    )


class DatasetInfoSummary(BaseModel):
    """Trimmed-down dataset_info for the response — full per-column detail
    is available via GET /csv/{session_id}/summary if needed."""
    n_rows: int
    n_cols: int
    column_names: list[str]


class CSVUploadResponse(BaseModel):
    session_id: str
    filename: str
    dataset_info: DatasetInfoSummary
    llm_classifications: LLMClassification
    equity_score: float | None
    disparity_summary: list[DisparitySummary]
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues surfaced during processing, e.g. "
                     "sparse groups or columns the LLM named that don't exist.",
    )


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None