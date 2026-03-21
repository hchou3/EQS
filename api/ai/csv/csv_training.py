import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from csv_processing import CSVData
from __future__ import annotations
import warnings
from classes import SparseGroupWarning, FairlearnBundle, FairlearnDataset, SensitivityMode
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
 
# Minimum samples a group must have before we emit a warning.
# Groups below this threshold produce unreliable fairness metrics.
MIN_GROUP_SAMPLES = 30

INTERSECTIONAL_SKIP_THRESHOLD = .30
INTERSECTIONAL_MAX_ATTRIBUTES = 3
# Classification metrics passed to MetricFrame.
# All are wrapped so they accept (y_true, y_pred) with no extra kwargs.
_CLF_METRICS = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "recall": lambda y_true, y_pred: recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
    "f1": lambda y_true, y_pred: f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    ),
}
 
# Regression metrics passed to MetricFrame.
_REG_METRICS = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "r2": r2_score,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
 
def _validate_columns(
    df: pd.DataFrame,
    protected_attributes: list[str],
    target_columns: list[str],
) -> tuple[list[str], list[str]]:
    """
    Drop any columns that are missing from df and warn the caller.
    Returns the surviving (protected_attributes, target_columns).
    """
    available = set(df.columns)
    valid_protected = []
    for col in protected_attributes:
        if col in available:
            valid_protected.append(col)
        else:
            warnings.warn(
                f"Protected attribute '{col}' not found in DataFrame — skipping.",
                UserWarning,
                stacklevel=3,
            )
 
    valid_targets = []
    for col in target_columns:
        if col in available:
            valid_targets.append(col)
        else:
            warnings.warn(
                f"Target column '{col}' not found in DataFrame — skipping.",
                UserWarning,
                stacklevel=3,
            )
 
    return valid_protected, valid_targets
 
 
def _check_sparse_groups(
    sensitive_features: pd.Series | pd.DataFrame,
    protected_attr: str,
    target_col: str,
    min_samples: int = MIN_GROUP_SAMPLES,
) -> list[SparseGroupWarning]:
    """
    Inspect every group in sensitive_features and return warnings for those
    with fewer than min_samples rows.
 
    Works for both a Series (individual mode) and a DataFrame (intersectional
    mode — groups are tuples of the row values).
    """
    sparse: list[SparseGroupWarning] = []
 
    if isinstance(sensitive_features, pd.Series):
        counts = sensitive_features.value_counts()
        for group_val, count in counts.items():
            if count < min_samples:
                sparse.append(
                    SparseGroupWarning(
                        protected_attr=protected_attr,
                        group_value=group_val,
                        n_samples=int(count),
                        target_col=target_col,
                    )
                )
    else:
        # Intersectional: group is the full row as a tuple
        counts = sensitive_features.apply(tuple, axis=1).value_counts()
        for group_val, count in counts.items():
            if count < min_samples:
                sparse.append(
                    SparseGroupWarning(
                        protected_attr=protected_attr,
                        group_value=group_val,
                        n_samples=int(count),
                        target_col=target_col,
                    )
                )
 
    return sparse
 
 
def _build_sensitive_features_individual(
    df: pd.DataFrame,
    protected_attr: str,
    idx: pd.Index,
) -> pd.Series:
    """Return a single-column Series aligned to idx."""
    return df.loc[idx, protected_attr]
 
 
def _build_sensitive_features_intersectional(
    df: pd.DataFrame,
    protected_attributes: list[str],
    idx: pd.Index,
) -> pd.DataFrame:
    """Return a multi-column DataFrame aligned to idx."""
    return df.loc[idx, protected_attributes]
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def prepare_fairlearn_data(
    csv_data: CSVData,
    mode: SensitivityMode = "individual",
    min_group_samples: int = MIN_GROUP_SAMPLES,
    drop_na: bool = True,
    intersectional_skip_threshold: float = INTERSECTIONAL_SKIP_THRESHOLD,
    allow_large_intersections: bool = False,
) -> FairlearnDataset:
    """
    Convert a CSVData object (post-LLM classification) into a FairlearnDataset
    ready to pass directly to MetricFrame.
 
    Parameters
    ----------
    csv_data:
        A CSVData instance that has already run identify_columns().
        Must have non-empty protected_attributes and target_columns.
    mode:
        "individual"     — one MetricFrame per protected attribute (independent).
                           sensitive_features is a pd.Series.
        "intersectional" — one MetricFrame across all protected attributes at once.
                           sensitive_features is a pd.DataFrame; groups are the
                           cartesian product of all attribute values.
    min_group_samples:
        Groups below this size emit a SparseGroupWarning. Metrics computed on
        tiny groups are statistically unreliable.
    drop_na:
        If True (default), rows where either the protected attribute or the
        target column are NaN are dropped before building each bundle. This
        means each bundle may use a slightly different subset of the data.
    intersectional_skip_threshold:
        Fraction of intersectional groups allowed to be sparse before a bundle
        is skipped. See INTERSECTIONAL_SKIP_THRESHOLD. Only used in
        intersectional mode.
    allow_large_intersections:
        If False (default), raises ValueError when intersectional mode is
        requested with more than INTERSECTIONAL_MAX_ATTRIBUTES (3) protected
        attributes. Set to True to override — but first run
        plot_intersectional_sparsity() to confirm your dataset can support it.
 
    Returns
    -------
    FairlearnDataset
        Contains one FairlearnBundle per (protected_attr, target_col) pair in
        individual mode, or one bundle per target_col (covering all protected
        attributes together) in intersectional mode.
 
    Raises
    ------
    ValueError
        If csv_data has not been classified yet (identify_columns not called),
        or if no valid protected attributes / target columns remain after
        validation.
 
    Examples
    --------
    >>> csv_data = CSVData("hospital.csv")
    >>> await csv_data.identify_columns(api_key=key, model="gemini-2.0-flash")
    >>> dataset = prepare_fairlearn_data(csv_data, mode="individual")
    >>> bundle = dataset.get_bundle("race", "readmitted")
    >>> mf = MetricFrame(
    ...     metrics=_CLF_METRICS,
    ...     y_true=bundle.y_true,
    ...     y_pred=your_model_predictions,
    ...     sensitive_features=bundle.sensitive_features,
    ... )
    """
    # ------------------------------------------------------------------
    # 1. Guard: ensure LLM classification has been run
    # ------------------------------------------------------------------
    if not csv_data.protected_attributes or not csv_data.target_columns:
        raise ValueError(
            "csv_data.protected_attributes and csv_data.target_columns are empty. "
            "Call csv_data.identify_columns() before prepare_fairlearn_data()."
        )
 
    # ------------------------------------------------------------------
    # 2. Validate columns exist in the DataFrame
    # ------------------------------------------------------------------
    valid_protected, valid_targets = _validate_columns(
        csv_data.df,
        csv_data.protected_attributes,
        csv_data.target_columns,
    )
 
    if not valid_protected:
        raise ValueError("No valid protected attributes found in the DataFrame.")
    if not valid_targets:
        raise ValueError("No valid target columns found in the DataFrame.")
 
    # ------------------------------------------------------------------
    # 3. Enforce intersectional attribute cap
    # ------------------------------------------------------------------
    if mode == "intersectional" and len(valid_protected) > INTERSECTIONAL_MAX_ATTRIBUTES:
        if not allow_large_intersections:
            raise ValueError(
                f"Intersectional mode received {len(valid_protected)} protected "
                f"attributes ({', '.join(valid_protected)}), exceeding "
                f"INTERSECTIONAL_MAX_ATTRIBUTES={INTERSECTIONAL_MAX_ATTRIBUTES}. "
                f"With {len(valid_protected)} attributes the number of intersectional "
                f"groups grows rapidly and most will be too sparse for reliable metrics. "
                f"Options:\n"
                f"  1. Pass the specific attributes you want to intersect directly: "
                f"csv_data.protected_attributes = ['race', 'gender']\n"
                f"  2. Set allow_large_intersections=True to override (run "
                f"plot_intersectional_sparsity() first to confirm your dataset supports it)."
            )
        else:
            warnings.warn(
                f"allow_large_intersections=True: proceeding with "
                f"{len(valid_protected)} intersectional attributes "
                f"({', '.join(valid_protected)}). Many groups may be sparse — "
                f"check bundle.skipped and bundle.sparse_warnings on the result.",
                UserWarning,
                stacklevel=2,
            )
 
    # ------------------------------------------------------------------
    # 4. Build bundles
    # ------------------------------------------------------------------
    bundles: list[FairlearnBundle] = []
    all_sparse: list[SparseGroupWarning] = []
 
    if mode == "individual":
        bundles, all_sparse = _build_individual_bundles(
            csv_data=csv_data,
            df=csv_data.df,
            valid_protected=valid_protected,
            valid_targets=valid_targets,
            min_group_samples=min_group_samples,
            drop_na=drop_na,
        )
 
    elif mode == "intersectional":
        bundles, all_sparse = _build_intersectional_bundles(
            csv_data=csv_data,
            df=csv_data.df,
            valid_protected=valid_protected,
            valid_targets=valid_targets,
            min_group_samples=min_group_samples,
            drop_na=drop_na,
            skip_threshold=intersectional_skip_threshold,
        )
 
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'individual' or 'intersectional'.")
 
    # ------------------------------------------------------------------
    # 5. Surface sparse group warnings
    # ------------------------------------------------------------------
    for w in all_sparse:
        warnings.warn(
            f"Sparse group detected — protected_attr='{w.protected_attr}', "
            f"group={w.group_value!r}, n={w.n_samples} (target='{w.target_col}'). "
            f"Fairness metrics may be unreliable. Consider grouping rare categories.",
            UserWarning,
            stacklevel=2,
        )
 
    return FairlearnDataset(
        df=csv_data.df,
        bundles=bundles,
        mode=mode,
        all_sparse_warnings=all_sparse,
    )
 
# ---------------------------------------------------------------------------
# Bundle builders
# ---------------------------------------------------------------------------
def _build_individual_bundles(
    csv_data: CSVData,
    df: pd.DataFrame,
    valid_protected: list[str],
    valid_targets: list[str],
    min_group_samples: int,
    drop_na: bool,
) -> tuple[list[FairlearnBundle], list[SparseGroupWarning]]:
    """
    Build one FairlearnBundle per (protected_attr, target_col) pair.
    sensitive_features is always a pd.Series.
    """
    bundles: list[FairlearnBundle] = []
    all_sparse: list[SparseGroupWarning] = []
 
    for target_col in valid_targets:
        task_type = (csv_data.target_column_types or {}).get(target_col, "classification")
        favorable = (csv_data.regression_favorable_directions or {}).get(target_col, {})
        favorable_direction = favorable.get("direction") if favorable else None
 
        for protected_attr in valid_protected:
            # Drop rows where either column is NaN (per-bundle subset)
            working_cols = [protected_attr, target_col]
            if drop_na:
                subset_idx = df[working_cols].dropna().index
            else:
                subset_idx = df.index
 
            sensitive_features = _build_sensitive_features_individual(
                df, protected_attr, subset_idx
            )
            y_true = df.loc[subset_idx, target_col]
 
            # Sparse group check
            sparse = _check_sparse_groups(
                sensitive_features, protected_attr, target_col, min_group_samples
            )
            all_sparse.extend(sparse)
 
            bundles.append(
                FairlearnBundle(
                    protected_attr=protected_attr,
                    target_col=target_col,
                    task_type=task_type,
                    sensitive_features=sensitive_features,
                    y_true=y_true,
                    mode="individual",
                    favorable_direction=favorable_direction,
                    sparse_warnings=sparse,
                )
            )
 
    return bundles, all_sparse
 
 
def _build_intersectional_bundles(
    csv_data: CSVData,
    df: pd.DataFrame,
    valid_protected: list[str],
    valid_targets: list[str],
    min_group_samples: int,
    drop_na: bool,
    skip_threshold: float = INTERSECTIONAL_SKIP_THRESHOLD,
) -> tuple[list[FairlearnBundle], list[SparseGroupWarning]]:
    """
    Build one FairlearnBundle per target_col with sensitive_features as a
    multi-column DataFrame covering all protected attributes simultaneously.
 
    Sparsity gate
    -------------
    After computing intersectional groups, we calculate the fraction of groups
    that fall below min_group_samples. If that fraction exceeds skip_threshold,
    the bundle is marked skipped=True and excluded from downstream training and
    MetricFrame computation. The sparse_warnings are still populated so the
    caller can inspect which groups caused the skip.
 
    Use plot_intersectional_sparsity() to visualise how sparsity fraction varies
    with different min_group_samples values before committing to a threshold.
 
    Parameters
    ----------
    skip_threshold:
        Fraction of intersectional groups allowed to be sparse before the
        bundle is skipped entirely. Default: INTERSECTIONAL_SKIP_THRESHOLD (0.30).
    """
    bundles: list[FairlearnBundle] = []
    all_sparse: list[SparseGroupWarning] = []
 
    combined_attr_label = ", ".join(valid_protected)
 
    for target_col in valid_targets:
        task_type = (csv_data.target_column_types or {}).get(target_col, "classification")
        favorable = (csv_data.regression_favorable_directions or {}).get(target_col, {})
        favorable_direction = favorable.get("direction") if favorable else None
 
        working_cols = valid_protected + [target_col]
        if drop_na:
            subset_idx = df[working_cols].dropna().index
        else:
            subset_idx = df.index
 
        sensitive_features = _build_sensitive_features_intersectional(
            df, valid_protected, subset_idx
        )
        y_true = df.loc[subset_idx, target_col]
 
        # --- Sparsity check ---
        sparse = _check_sparse_groups(
            sensitive_features, combined_attr_label, target_col, min_group_samples
        )
        all_sparse.extend(sparse)
 
        # Compute sparsity fraction across all intersectional groups
        group_counts = sensitive_features.apply(tuple, axis=1).value_counts()
        n_total_groups = len(group_counts)
        n_sparse_groups = len(sparse)
        sparse_fraction = n_sparse_groups / n_total_groups if n_total_groups > 0 else 1.0
 
        # Gate: skip bundle if too many intersectional groups are sparse
        if sparse_fraction > skip_threshold:
            skip_reason = (
                f"{n_sparse_groups}/{n_total_groups} intersectional groups "
                f"({sparse_fraction:.0%}) are below min_group_samples={min_group_samples}, "
                f"exceeding skip_threshold={skip_threshold:.0%}. "
                f"Run plot_intersectional_sparsity() to inspect and tune thresholds."
            )
            warnings.warn(
                f"Intersectional bundle skipped for target='{target_col}' "
                f"({combined_attr_label}): {skip_reason}",
                UserWarning,
                stacklevel=4,
            )
            bundles.append(
                FairlearnBundle(
                    protected_attr=combined_attr_label,
                    target_col=target_col,
                    task_type=task_type,
                    sensitive_features=sensitive_features,
                    y_true=y_true,
                    mode="intersectional",
                    favorable_direction=favorable_direction,
                    sparse_warnings=sparse,
                    skipped=True,
                    skip_reason=skip_reason,
                )
            )
        else:
            bundles.append(
                FairlearnBundle(
                    protected_attr=combined_attr_label,
                    target_col=target_col,
                    task_type=task_type,
                    sensitive_features=sensitive_features,
                    y_true=y_true,
                    mode="intersectional",
                    favorable_direction=favorable_direction,
                    sparse_warnings=sparse,
                )
            )
 
    return bundles, all_sparse
 
 
# ---------------------------------------------------------------------------
# MetricFrame runner
# ---------------------------------------------------------------------------
def compute_metric_frames(
    dataset: FairlearnDataset,
    predictions: dict[str, np.ndarray | pd.Series],
) -> dict[str, dict[str, MetricFrame]]:
    """
    Run MetricFrame for every bundle in the dataset using pre-computed
    model predictions.
 
    Parameters
    ----------
    dataset:
        Output of prepare_fairlearn_data().
    predictions:
        Mapping of target_col → array-like of model predictions, aligned to
        the full DataFrame index. Each bundle slices to its own subset index
        automatically.
 
    Returns
    -------
    Nested dict: results[protected_attr][target_col] = MetricFrame
 
    Notes
    -----
    In intersectional mode, protected_attr is the comma-joined label string
    (e.g. "race, gender, age") — use that as the key.
    """
    results: dict[str, dict[str, MetricFrame]] = {}
 
    for bundle in dataset.bundles:
        # Skip bundles that were flagged during preparation (e.g. too sparse)
        if bundle.skipped:
            warnings.warn(
                f"Skipping MetricFrame for '{bundle.protected_attr}' / "
                f"'{bundle.target_col}': {bundle.skip_reason}",
                UserWarning,
                stacklevel=2,
            )
            continue
 
        if bundle.target_col not in predictions:
            warnings.warn(
                f"No predictions provided for target '{bundle.target_col}' — skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
 
        # Align predictions to this bundle's subset index
        raw_preds = predictions[bundle.target_col]
        if isinstance(raw_preds, np.ndarray):
            # Assume predictions are aligned to the full df; slice by position
            full_idx = dataset.df.index
            bundle_positions = full_idx.get_indexer(bundle.sensitive_features.index)
            y_pred = raw_preds[bundle_positions]
        else:
            y_pred = raw_preds.loc[bundle.sensitive_features.index]
 
        metrics = _CLF_METRICS if bundle.task_type == "classification" else _REG_METRICS
 
        mf = MetricFrame(
            metrics=metrics,
            y_true=bundle.y_true,
            y_pred=y_pred,
            sensitive_features=bundle.sensitive_features,
        )
 
        results.setdefault(bundle.protected_attr, {})[bundle.target_col] = mf
 
    return results
 
 
# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------
def summarise_dataset(dataset: FairlearnDataset) -> dict:
    """
    Return a JSON-serialisable summary of the prepared dataset — useful for
    logging or displaying to the user before running models.
    """
    bundle_summaries = []
    for b in dataset.bundles:
        if isinstance(b.sensitive_features, pd.Series):
            group_counts = b.sensitive_features.value_counts().to_dict()
        else:
            group_counts = (
                b.sensitive_features.apply(tuple, axis=1)
                .value_counts()
                .to_dict()
            )
            # Convert tuple keys to strings for JSON safety
            group_counts = {str(k): v for k, v in group_counts.items()}
 
        bundle_summaries.append({
            "protected_attr": b.protected_attr,
            "target_col": b.target_col,
            "task_type": b.task_type,
            "mode": b.mode,
            "n_samples": len(b.y_true),
            "n_groups": len(group_counts),
            "group_counts": group_counts,
            "favorable_direction": b.favorable_direction,
            "n_sparse_groups": len(b.sparse_warnings),
        })
 
    return {
        "mode": dataset.mode,
        "total_bundles": len(dataset.bundles),
        "total_sparse_warnings": len(dataset.all_sparse_warnings),
        "bundles": bundle_summaries,

    }