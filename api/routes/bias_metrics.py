"""
GET /csv/{session_id}/bias-metrics

Returns Recharts-ready chart data derived from the bias_results already
computed during /csv/upload (run_bias_analysis). This endpoint does NOT
recompute anything — it reshapes csv_data.bias_results into the flat,
per-group array format Recharts expects.

If you need fresher results (e.g. after the user dropped a column),
re-run /csv/upload rather than calling this — there's no mutation path
here by design, to keep this endpoint cheap and idempotent.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .bias_metrics_models import (
    BiasMetricChart,
    BiasMetricsResponse,
    ClassificationChartPoint,
    RegressionChartPoint,
)
from .sessions import session_store

router = APIRouter(prefix="/csv", tags=["csv"])


def _classification_points(rate_by_group: dict) -> list[ClassificationChartPoint] | None:
    """
    Build chart points from a classification rate_by_group dict.

    rate_by_group is either:
      - {group: scalar_rate}  (numeric/binary target)
      - {group: {category: proportion}}  (categorical target, value distribution)

    For the categorical case, charting every category per group is noisy
    for a first pass — we surface the modal (most common) category's share
    per group instead, which is enough to spot disparity at a glance.
    """
    if not rate_by_group:
        return None

    points: list[ClassificationChartPoint] = []
    for group, value in rate_by_group.items():
        if isinstance(value, dict):
            if not value:
                continue
            # Modal category share for this group
            modal_value = max(value.values())
            points.append(ClassificationChartPoint(group=str(group), value=round(modal_value, 4)))
        else:
            points.append(ClassificationChartPoint(group=str(group), value=round(float(value), 4)))

    return points or None


def _regression_points(group_statistics: dict) -> list[RegressionChartPoint] | None:
    """
    Build chart points from the group_statistics dict produced by
    pandas' groups.describe().to_dict() in _continuous_bias_analysis.

    IMPORTANT: this dict is oriented {stat_name: {group: value}}, e.g.
        {"mean": {"Black": 5.01, "White": 4.74}, "std": {...}, "count": {...}}
    NOT {group: {stat_name: value}}. We transpose before reading.
    """
    if not group_statistics:
        return None

    means = group_statistics.get("mean")
    if not means:
        return None

    stds = group_statistics.get("std", {})
    counts = group_statistics.get("count", {})

    points: list[RegressionChartPoint] = []
    for group, mean in means.items():
        if mean is None:
            continue
        std = stds.get(group)
        count = counts.get(group)
        points.append(
            RegressionChartPoint(
                group=str(group),
                value=round(float(mean), 4),
                std=round(float(std), 4) if std is not None else None,
                n=int(count) if count is not None else None,
            )
        )

    return points or None


def _build_chart(protected_attr: str, target_col: str, metrics: dict) -> BiasMetricChart:
    """Convert one bias_results[protected_attr][target_col] entry into a chart."""
    if "error" in metrics:
        return BiasMetricChart(
            protected_attr=protected_attr,
            target_col=target_col,
            task_type="unknown",
            chart_type="bar",
            data=[],
            error=metrics["error"],
        )

    # Regression pairs carry an "anova" key (see _continuous_bias_analysis).
    if "anova" in metrics:
        points = _regression_points(metrics.get("group_statistics"))
        p_value = metrics["anova"].get("p_value")
        if points is None:
            return BiasMetricChart(
                protected_attr=protected_attr,
                target_col=target_col,
                task_type="regression",
                chart_type="bar",
                data=[],
                error="No group statistics available to chart.",
            )
        return BiasMetricChart(
            protected_attr=protected_attr,
            target_col=target_col,
            task_type="regression",
            chart_type="bar",
            disparity=None,
            p_value=round(float(p_value), 6) if p_value is not None else None,
            is_significant=(p_value is not None and p_value < 0.05),
            data=points,
        )

    # Classification pairs carry "rate_by_group" and "disparity".
    points = _classification_points(metrics.get("rate_by_group"))
    if points is None:
        return BiasMetricChart(
            protected_attr=protected_attr,
            target_col=target_col,
            task_type="classification",
            chart_type="bar",
            data=[],
            error="No group rates available to chart.",
        )
    disparity = metrics.get("disparity")
    return BiasMetricChart(
        protected_attr=protected_attr,
        target_col=target_col,
        task_type="classification",
        chart_type="bar",
        disparity=round(float(disparity), 4) if disparity is not None else None,
        data=points,
    )


@router.get(
    "/{session_id}/bias-metrics",
    response_model=BiasMetricsResponse,
    responses={
        404: {"description": "Session not found, or bias analysis has not been run yet"},
    },
)
async def get_bias_metrics(
    session_id: str,
    protected_attr: str | None = Query(
        default=None,
        description="Filter to a single protected attribute, e.g. ?protected_attr=race. "
                     "Omit to return charts for all protected attributes.",
    ),
) -> BiasMetricsResponse:
    """
    Return Recharts-ready chart data for one or all protected attributes
    in the uploaded dataset.

    Each chart corresponds to one (protected_attr, target_col) pair and
    contains a flat `data` array suitable for direct use as a Recharts
    <BarChart data={chart.data}> prop, with `group` as the XAxis dataKey
    and `value` as the Bar dataKey.
    """
    session = session_store.get(session_id)
    if session is None or session.csv_data is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a CSV via POST /csv/upload first.",
        )

    bias_results = session.csv_data.bias_results
    if not bias_results:
        raise HTTPException(
            status_code=404,
            detail="No bias analysis results found for this session. "
                   "This shouldn't happen if /csv/upload completed successfully — "
                   "re-upload the file.",
        )

    available_attrs = list(bias_results.keys())

    if protected_attr is not None:
        if protected_attr not in bias_results:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Protected attribute '{protected_attr}' not found in this session. "
                    f"Available: {', '.join(available_attrs) or 'none'}."
                ),
            )
        attrs_to_render = {protected_attr: bias_results[protected_attr]}
    else:
        attrs_to_render = bias_results

    charts: list[BiasMetricChart] = []
    for attr, targets in attrs_to_render.items():
        for target_col, metrics in targets.items():
            charts.append(_build_chart(attr, target_col, metrics))

    return BiasMetricsResponse(
        session_id=session_id,
        charts=charts,
        available_protected_attrs=available_attrs,
    )