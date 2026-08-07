"""
Models for GET /csv/{session_id}/bias-metrics.

Shaped for Recharts: each chart's `data` is a flat list of objects where
every object is one x-axis category (a group) and its numeric value(s)
are keys on that same object. This is the format Recharts' <BarChart>,
<XAxis dataKey="group">, and <Bar dataKey="value"> expect directly —
no client-side reshaping needed.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationChartPoint(BaseModel):
    """One bar in a classification outcome-rate chart."""
    group: str
    value: float = Field(description="Outcome rate for this group, 0-1.")


class RegressionChartPoint(BaseModel):
    """One bar in a regression group-mean chart."""
    group: str
    value: float = Field(description="Mean of the target column for this group.")
    std: float | None = Field(default=None, description="Standard deviation, for error bars.")
    n: int | None = Field(default=None, description="Sample size for this group.")


class BiasMetricChart(BaseModel):
    """
    One complete chart: a (protected_attr, target_col) pair plus its
    Recharts-ready data array and enough metadata to title/render it.
    """
    protected_attr: str
    target_col: str
    task_type: str  # "classification" | "regression"
    chart_type: str  # "bar" — reserved for future chart_type variety
    disparity: float | None = None
    p_value: float | None = Field(
        default=None,
        description="ANOVA p-value, regression only. Null for classification.",
    )
    is_significant: bool | None = Field(
        default=None,
        description="p_value < 0.05, regression only.",
    )
    data: list[ClassificationChartPoint] | list[RegressionChartPoint]
    error: str | None = Field(
        default=None,
        description="Set instead of `data` if this pair could not be computed.",
    )


class BiasMetricsResponse(BaseModel):
    session_id: str
    charts: list[BiasMetricChart]
    available_protected_attrs: list[str] = Field(
        description="All protected attributes with charts in this dataset, "
                     "for populating a filter dropdown in the UI."
    )