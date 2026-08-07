from __future__ import annotations
 
import warnings
 
import numpy as np
import pandas as pd
 
from .csv_processing import INTERSECTIONAL_SKIP_THRESHOLD, MIN_GROUP_SAMPLES
from .csv_training import compare_shap_across_groups, compute_shap
from .classes import ShapResult
 
 
# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
# compute_shap and compare_shap_across_groups live in csv_training.py.
# They are re-exported here so callers who import from statistics get the
# same surface they had before the refactor.
__all__ = [
    "compute_shap",
    "compare_shap_across_groups",
    "plot_intersectional_sparsity",
]
 
 
# ---------------------------------------------------------------------------
# Developer diagnostic — call before committing to an intersectional run
# ---------------------------------------------------------------------------
 
def plot_intersectional_sparsity(
    df: pd.DataFrame,
    protected_attributes: list[str],
    min_samples_range: tuple[int, int] = (10, 200),
    step: int = 10,
    current_threshold: float = INTERSECTIONAL_SKIP_THRESHOLD,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """
    Plot how the fraction of sparse intersectional groups changes as
    min_group_samples varies.
 
    Use this before calling prepare_fairlearn_data() in intersectional mode
    to choose appropriate MIN_GROUP_SAMPLES and INTERSECTIONAL_SKIP_THRESHOLD
    values for your specific dataset.
 
    The chart shows two things:
    - **Sparsity fraction** (left axis): fraction of groups below each
      threshold. When this line crosses your skip_threshold the bundle
      would be skipped.
    - **Group count** (right axis): total number of intersectional groups
      observed. A large group count with high sparsity signals the
      combination is too fine-grained for your dataset size.
 
    Reference lines mark the current MIN_GROUP_SAMPLES default (30) and
    INTERSECTIONAL_SKIP_THRESHOLD (0.30). The region where both thresholds
    are exceeded is shaded red — bundles built with those settings would
    be skipped.
 
    Parameters
    ----------
    df:
        The raw DataFrame (pre-split). Use the full dataset for the most
        accurate picture of group sizes.
    protected_attributes:
        The protected attribute columns to combine intersectionally.
        Tip: call this multiple times with different subsets (2-attr,
        3-attr) to find the maximum safe combination.
    min_samples_range:
        (min, max) range of min_group_samples values to sweep over.
    step:
        Step size for the sweep.
    current_threshold:
        The skip_threshold value to draw the horizontal reference line at.
    figsize:
        Matplotlib figure size.
 
    Examples
    --------
    >>> plot_intersectional_sparsity(
    ...     df=csv_data.df,
    ...     protected_attributes=["race", "gender"],
    ... )
    >>> # Try a 3-attribute combination:
    >>> plot_intersectional_sparsity(
    ...     df=csv_data.df,
    ...     protected_attributes=["race", "gender", "age"],
    ... )
 
    Raises
    ------
    ImportError  if matplotlib is not installed.
    ValueError   if none of the protected_attributes exist in df.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_intersectional_sparsity(). "
            "Install it with: pip install matplotlib"
        )
 
    available = [c for c in protected_attributes if c in df.columns]
    if not available:
        raise ValueError("None of the protected_attributes were found in df.columns.")
    if len(available) < len(protected_attributes):
        missing = set(protected_attributes) - set(available)
        warnings.warn(
            f"Columns not found in df, ignoring: {missing}",
            UserWarning,
            stacklevel=2,
        )
 
    group_series = df[available].astype(str).apply(tuple, axis=1)
    group_counts = group_series.value_counts()
    n_total_groups = len(group_counts)
    n_rows = len(df)
 
    thresholds = list(range(min_samples_range[0], min_samples_range[1] + step, step))
    sparsity_fractions = [
        int((group_counts < t).sum()) / n_total_groups if n_total_groups > 0 else 1.0
        for t in thresholds
    ]
 
    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
 
    color_sparse = "#D85A30"
    color_groups = "#185FA5"
 
    ax1.plot(
        thresholds, sparsity_fractions,
        color=color_sparse, linewidth=2, marker="o", markersize=4,
        label="Sparse group fraction",
    )
    ax2.axhline(
        n_total_groups, color=color_groups, linewidth=1,
        linestyle=":", alpha=0.6, label=f"Total groups ({n_total_groups})",
    )
 
    # Reference lines
    ax1.axvline(
        MIN_GROUP_SAMPLES, color="#888780", linewidth=1,
        linestyle="--", label=f"Default min_samples ({MIN_GROUP_SAMPLES})",
    )
    ax1.axhline(
        current_threshold, color="#993C1D", linewidth=1,
        linestyle="--", label=f"Skip threshold ({current_threshold:.0%})",
    )
 
    # Shade the skip zone
    skip_xs = [t for t, f in zip(thresholds, sparsity_fractions) if f > current_threshold]
    if skip_xs:
        ax1.axvspan(
            min(skip_xs), max(thresholds),
            alpha=0.07, color=color_sparse, label="Skip zone",
        )
 
    ax1.set_xlabel("min_group_samples", fontsize=11)
    ax1.set_ylabel("Fraction of sparse groups", color=color_sparse, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_sparse)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.set_ylim(0, 1.05)
 
    ax2.set_ylabel("Total intersectional groups", color=color_groups, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_groups)
    ax2.set_ylim(0, n_total_groups * 2)
 
    attr_label = " × ".join(available)
    ax1.set_title(
        f"Intersectional sparsity: {attr_label}\n"
        f"n={n_rows:,} rows  |  {n_total_groups} unique groups",
        fontsize=12,
    )
 
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
 
    fig.tight_layout()
    plt.show()