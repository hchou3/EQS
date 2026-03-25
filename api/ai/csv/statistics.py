import pandas as pd
from csv_training import INTERSECTIONAL_SKIP_THRESHOLD, MIN_GROUP_SAMPLES
import warnings
from classes import FairlearnBundle, ShapResult
import numpy as np
import sklearn.base as skbase
 
# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
 
 
def compute_shap(
    bundle: "FairlearnBundle",
    estimator,
    X: pd.DataFrame,
    store_shap_matrix: bool = True,
    max_shap_samples: int = 5000,
) -> ShapResult:
    """
    Refit estimator on the full dataset, then compute SHAP values for
    explainability and per-group feature importance analysis.
 
    The refit model is kept strictly separate from the CV model used to
    produce MetricFrame predictions. This avoids two problems:
    - Data leakage: the CV model's predictions are unbiased (each row was
      predicted on held-out data). Reusing it for SHAP would conflate
      explanation with evaluation.
    - Representational bias: the last CV fold's model was only trained on
      ~80% of rows. For SHAP, a model trained on the full dataset produces
      more stable and representative attribution values.
    """
    try:
        import shap as shap_lib
    except ImportError:
        raise ImportError(
            "shap is required for compute_shap(). "
            "Install it with: pip install shap"
        )
 
    if bundle.skipped:
        raise ValueError(
            f"Cannot compute SHAP for a skipped bundle "
            f"(protected_attr='{bundle.protected_attr}', "
            f"target_col='{bundle.target_col}'). "
            f"Reason: {bundle.skip_reason}"
        )
 
    # ------------------------------------------------------------------
    # 1. Align X to the bundle's index
    # ------------------------------------------------------------------
    common_idx = X.index.intersection(bundle.y_true.index)
    if len(common_idx) == 0:
        raise ValueError(
            "X and bundle.y_true share no common index values. "
            "Ensure X was built from the same DataFrame as the bundle."
        )
    if len(common_idx) < len(bundle.y_true):
        warnings.warn(
            f"X covers {len(common_idx)}/{len(bundle.y_true)} rows in the bundle. "
            "Missing rows will be excluded from SHAP computation.",
            UserWarning,
            stacklevel=2,
        )
 
    X_aligned = X.loc[common_idx]
    y_aligned = bundle.y_true.loc[common_idx]
 
    # ------------------------------------------------------------------
    # 2. Stratified sample if dataset exceeds max_shap_samples
    # ------------------------------------------------------------------
    if len(X_aligned) > max_shap_samples:
        # Stratify on sensitive_features to preserve group representation
        sensitive_aligned = (
            bundle.sensitive_features.loc[common_idx]
            if isinstance(bundle.sensitive_features, pd.Series)
            else bundle.sensitive_features.loc[common_idx].apply(tuple, axis=1)
        )
        # Sample proportionally within each group
        sampled_idx = (
            pd.Series(common_idx, index=common_idx)
            .groupby(sensitive_aligned)
            .apply(
                lambda g: g.sample(
                    min(len(g), max(1, int(max_shap_samples * len(g) / len(X_aligned)))),
                    random_state=42,
                )
            )
            .droplevel(0)
            .index
        )
        X_aligned = X_aligned.loc[sampled_idx]
        y_aligned = y_aligned.loc[sampled_idx]
        warnings.warn(
            f"Dataset has {len(common_idx)} rows, exceeding max_shap_samples="
            f"{max_shap_samples}. SHAP computed on a stratified sample of "
            f"{len(X_aligned)} rows.",
            UserWarning,
            stacklevel=2,
        )
 
    # ------------------------------------------------------------------
    # 3. Full-data refit — separate model purely for explanation.
    #    Critically, this uses X_full / y_full (the full common_idx data)
    #    NOT the downsampled X_aligned / y_aligned. The refit model must
    #    see the full distribution so SHAP values reflect all groups
    #    equally. SHAP computation itself runs on the (smaller) sample.
    # ------------------------------------------------------------------
    X_full = X.loc[common_idx]
    y_full = bundle.y_true.loc[common_idx]
 
    refit_model = skbase.clone(estimator)
    refit_model.fit(X_full, y_full)
 
    # ------------------------------------------------------------------
    # 4. Compute SHAP values using TreeExplainer on the sample.
    #    TreeExplainer is exact for tree-based models (XGBoost, LightGBM)
    #    and orders of magnitude faster than KernelExplainer.
    # ------------------------------------------------------------------
    explainer = shap_lib.TreeExplainer(refit_model)
    raw_shap = explainer.shap_values(X_aligned)
 
    # For multi-class classifiers, shap_values() returns a list of arrays
    # (one per class). Collapse to mean absolute across classes so
    # importances are a single (n_samples, n_features) array.
    if isinstance(raw_shap, list):
        shap_matrix = np.mean(np.abs(np.stack(raw_shap, axis=0)), axis=0)
    else:
        shap_matrix = raw_shap
 
    feature_names = list(X_aligned.columns)
 
    # ------------------------------------------------------------------
    # 5. Global importances — mean |SHAP| per feature
    # ------------------------------------------------------------------
    global_importances = pd.Series(
        np.abs(shap_matrix).mean(axis=0),
        index=feature_names,
        name="mean_abs_shap",
    ).sort_values(ascending=False)
 
    # ------------------------------------------------------------------
    # 6. Per-group importances — mean |SHAP| within each sensitive group
    # ------------------------------------------------------------------
    if isinstance(bundle.sensitive_features, pd.Series):
        group_labels = bundle.sensitive_features.loc[X_aligned.index]
    else:
        # Intersectional: represent each group as a joined string for readability
        group_labels = (
            bundle.sensitive_features.loc[X_aligned.index]
            .astype(str)
            .apply(" | ".join, axis=1)
        )
 
    per_group_importances: dict[str, pd.Series] = {}
    for group_val, group_idx in group_labels.groupby(group_labels).groups.items():
        group_shap = shap_matrix[X_aligned.index.get_indexer(group_idx)]
        per_group_importances[str(group_val)] = pd.Series(
            np.abs(group_shap).mean(axis=0),
            index=feature_names,
            name=f"mean_abs_shap_{group_val}",
        ).sort_values(ascending=False)
 
    return ShapResult(
        protected_attr=bundle.protected_attr,
        target_col=bundle.target_col,
        global_importances=global_importances,
        per_group_importances=per_group_importances,
        feature_names=feature_names,
        n_samples=len(X_aligned),
        refit_model=refit_model,
        shap_values=shap_matrix if store_shap_matrix else None,
    )

def compare_shap_across_groups(
    shap_result: ShapResult,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Build a comparison DataFrame of per-group SHAP importances side by side.
 
    This is the primary tool for proxy detection — features with low global
    importance but high importance in a specific subgroup indicate the model
    is treating that group differently, often via a proxy variable.
 
    Parameters
    ----------
    shap_result:
        Output of compute_shap().
    top_n:
        Number of top features to include (ranked by global importance).
 
    Returns
    -------
    pd.DataFrame
        Rows = top_n features (by global importance).
        Columns = groups + "global".
        Values = mean absolute SHAP value.
        Sorted by global importance descending.
 
    Example
    -------
    >>> df_comparison = compare_shap_across_groups(shap_result, top_n=15)
    >>> print(df_comparison)
    >>> # High value in one group column but low "global" = proxy signal
    """
    top_features = shap_result.global_importances.head(top_n).index.tolist()
 
    data = {"global": shap_result.global_importances.loc[top_features]}
    for group, importances in shap_result.per_group_importances.items():
        # Reindex to top_features — some features may be missing if group
        # had insufficient samples for all feature values
        data[str(group)] = importances.reindex(top_features, fill_value=0.0)
 
    return pd.DataFrame(data, index=top_features).round(4)

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
    min_group_samples varies. Use this to manually choose an appropriate
    INTERSECTIONAL_SKIP_THRESHOLD for your dataset before running the pipeline.
 
    The chart shows two things:
    - Sparsity fraction (left axis): fraction of groups below each threshold.
      When this line crosses your skip_threshold, the bundle would be skipped.
    - Group count (right axis): total number of intersectional groups observed.
      A large group count with high sparsity signals the combination is too
      fine-grained for your dataset size.
 
    A vertical dashed line marks the current MIN_GROUP_SAMPLES default (30).
    A horizontal dashed line marks the current INTERSECTIONAL_SKIP_THRESHOLD (0.30).
    The region where both thresholds are exceeded is shaded red — bundles
    built with those settings would be skipped.
 
    Parameters
    ----------
    df:
        The raw DataFrame (pre-split). Use the full dataset for the most
        accurate picture of group sizes.
    protected_attributes:
        The list of protected attribute columns to combine intersectionally.
        Tip: call this multiple times with different subsets (2-attr, 3-attr)
        to find the maximum safe combination.
    min_samples_range:
        (min, max) range of min_group_samples values to sweep over.
    step:
        Step size for the sweep.
    current_threshold:
        The skip_threshold value to draw the horizontal reference line at.
    figsize:
        Matplotlib figure size.
 
    Example
    -------
    >>> plot_intersectional_sparsity(
    ...     df=csv_data.df,
    ...     protected_attributes=["race", "gender"],
    ... )
    >>> # Then try 3-attr combination:
    >>> plot_intersectectional_sparsity(
    ...     df=csv_data.df,
    ...     protected_attributes=["race", "gender", "age"],
    ... )
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_intersectional_sparsity(). "
            "Install it with: pip install matplotlib"
        )
 
    # Build the composite group Series once
    available = [c for c in protected_attributes if c in df.columns]
    if not available:
        raise ValueError("None of the protected_attributes were found in df.columns.")
    if len(available) < len(protected_attributes):
        missing = set(protected_attributes) - set(available)
        warnings.warn(f"Columns not found in df, ignoring: {missing}", UserWarning)
 
    group_series = df[available].astype(str).apply(tuple, axis=1)
    group_counts = group_series.value_counts()
    n_total_groups = len(group_counts)
    n_rows = len(df)
 
    thresholds = list(range(min_samples_range[0], min_samples_range[1] + step, step))
    sparsity_fractions = []
 
    for t in thresholds:
        n_sparse = int((group_counts < t).sum())
        sparsity_fractions.append(n_sparse / n_total_groups if n_total_groups > 0 else 1.0)
 
    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
 
    color_sparse = "#D85A30"
    color_groups = "#185FA5"
 
    ax1.plot(thresholds, sparsity_fractions, color=color_sparse, linewidth=2,
             marker="o", markersize=4, label="Sparse group fraction")
    ax2.axhline(n_total_groups, color=color_groups, linewidth=1,
                linestyle=":", alpha=0.6, label=f"Total groups ({n_total_groups})")
 
    # Reference lines
    ax1.axvline(MIN_GROUP_SAMPLES, color="#888780", linewidth=1,
                linestyle="--", label=f"Default min_samples ({MIN_GROUP_SAMPLES})")
    ax1.axhline(current_threshold, color="#993C1D", linewidth=1,
                linestyle="--", label=f"Skip threshold ({current_threshold:.0%})")
 
    # Shade the skip zone (above threshold AND to the right of default min_samples)
    skip_xs = [t for t, f in zip(thresholds, sparsity_fractions) if f > current_threshold]
    if skip_xs:
        ax1.axvspan(min(skip_xs), max(thresholds), alpha=0.07,
                    color=color_sparse, label="Skip zone")
 
    # Annotations
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
 
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
 
    fig.tight_layout()
    plt.show()