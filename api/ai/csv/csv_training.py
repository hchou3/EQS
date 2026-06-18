from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn.base as skbase

from .csv_processing import MIN_GROUP_SAMPLES
from .classes import BundleResult, FairlearnBundle, ShapResult

# ---------------------------------------------------------------------------
# Hyperparameter configuration
# ---------------------------------------------------------------------------

LGBM_BASE_PARAMS: dict = {
    "n_estimators": 300,
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,   # L1 — sparse feature selection
    "reg_lambda": 1.0,  # L2 — weight regularisation
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

# Cross-validation
CV_N_SPLITS = 5

# min_child_samples is set adaptively per dataset size (see _make_lgbm).
# These constants define the floor and the percentage used in the calculation.
CV_MIN_CHILD_SAMPLES_FLOOR = 20   # hard minimum rows per leaf
CV_MIN_CHILD_SAMPLES_PCT   = 0.005  # 0.5% of dataset rows

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# All metrics are wrapped as plain (y_true, y_pred) callables so MetricFrame
# doesn't need to manage extra kwargs.
_CLF_METRICS: dict = {
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

_REG_METRICS: dict = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    "r2": r2_score,
}


# ===========================================================================
# Estimator factory
# ===========================================================================

def _make_lgbm(task_type: str, n_samples: int) -> LGBMClassifier | LGBMRegressor:
    """
    Build a configured but unfitted LightGBM estimator.

    min_child_samples is set adaptively so the model cannot create leaves that
    represent only a handful of people from a single demographic subgroup:

        min_child = max(CV_MIN_CHILD_SAMPLES_FLOOR, 0.5% of n_samples)

    Examples
    --------
    10 000-row dataset → max(20, 50) = 50
      500-row dataset  → max(20,  2) = 20  (floor applies)

    Classification models also receive class_weight="balanced" to handle
    label imbalance without requiring manual resampling.
    """
    min_child = max(
        CV_MIN_CHILD_SAMPLES_FLOOR,
        int(CV_MIN_CHILD_SAMPLES_PCT * n_samples),
    )
    params = {**LGBM_BASE_PARAMS, "min_child_samples": min_child}

    if task_type == "classification":
        return LGBMClassifier(**params, class_weight="balanced")
    return LGBMRegressor(**params)


# ===========================================================================
# Training
# ===========================================================================

def train_bundle(
    bundle: FairlearnBundle,
    X: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
) -> BundleResult:
    """
    Train a LightGBM model on one FairlearnBundle using cross-validated
    out-of-fold (OOF) predictions, then compute MetricFrame on the full bundle.

    Why OOF predictions matter for fairness
    ----------------------------------------
    Evaluating fairness on training-set predictions produces optimistic metrics
    because the model has already seen every row. OOF predictions guarantee that
    each row is scored exactly once on held-out data, giving an honest picture
    of per-group performance on unseen examples.

    CV strategy selection
    ---------------------
    The strategy adapts to minority-class size to avoid errors from
    StratifiedKFold receiving classes with fewer members than n_splits:

    | Condition                  | Strategy                         |
    |----------------------------|----------------------------------|
    | min_class >= n_splits      | StratifiedKFold(n_splits)        |
    | 2 <= min_class < n_splits  | StratifiedKFold(min_class)       |
    | min_class == 1             | KFold fallback (no stratification)|
    | Regression                 | KFold always                     |

    Parameters
    ----------
    bundle:
        A non-skipped FairlearnBundle produced by prepare_fairlearn_data().
    X:
        Encoded feature matrix built from csv_data.df[csv_data.encoded_feature_cols].
        Must share an index with bundle.y_true.
    n_splits:
        Target number of CV folds. May be reduced for minority-class safety.

    Returns
    -------
    BundleResult containing OOF predictions, MetricFrame, last fold's model,
    and the actual fold count used.

    Raises
    ------
    ValueError
        If bundle.skipped is True, or X and bundle.y_true share no common index.
    """
    if bundle.skipped:
        raise ValueError(
            f"Cannot train a skipped bundle "
            f"(protected_attr='{bundle.protected_attr}', "
            f"target_col='{bundle.target_col}'). "
            f"Reason: {bundle.skip_reason}"
        )

    # ------------------------------------------------------------------
    # 1. Align X to the bundle's NaN-dropped index
    # ------------------------------------------------------------------
    common_idx = X.index.intersection(bundle.y_true.index)
    if len(common_idx) == 0:
        raise ValueError(
            "X and bundle.y_true share no common index values. "
            "Ensure X was built from csv_data.df[csv_data.encoded_feature_cols] "
            "using the same CSVData instance as the bundle."
        )

    X_aligned = X.loc[common_idx]
    y_aligned = bundle.y_true.loc[common_idx]

    # ------------------------------------------------------------------
    # 2. Choose CV strategy
    # ------------------------------------------------------------------
    if bundle.task_type == "classification":
        min_class_count = int(y_aligned.value_counts().min())

        if min_class_count >= n_splits:
            actual_splits = n_splits
            cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
        elif min_class_count >= 2:
            actual_splits = min_class_count
            cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
            warnings.warn(
                f"train_bundle: minority class has only {min_class_count} samples "
                f"for target='{bundle.target_col}'. Reducing to "
                f"StratifiedKFold(n_splits={actual_splits}).",
                UserWarning,
                stacklevel=2,
            )
        else:
            actual_splits = min(n_splits, len(y_aligned))
            cv = KFold(n_splits=actual_splits, shuffle=True, random_state=42)
            warnings.warn(
                f"train_bundle: a class has only 1 sample for "
                f"target='{bundle.target_col}'. Falling back to "
                f"KFold(n_splits={actual_splits}).",
                UserWarning,
                stacklevel=2,
            )
    else:
        actual_splits = n_splits
        cv = KFold(n_splits=actual_splits, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # 3. Build estimator — sized to this bundle's dataset
    # ------------------------------------------------------------------
    model = _make_lgbm(bundle.task_type, n_samples=len(y_aligned))

    # ------------------------------------------------------------------
    # 4. OOF loop — collect out-of-fold predictions
    # ------------------------------------------------------------------
    if bundle.task_type == "regression":
        y_pred_oof = pd.Series(
            np.full(len(y_aligned), np.nan),
            index=y_aligned.index,
            dtype=float,
        )
    else:
        y_pred_oof = pd.Series(index=y_aligned.index, dtype=object)

    cv_split_input = y_aligned if bundle.task_type == "classification" else None
    last_model = None

    for _, (train_pos, test_pos) in enumerate(
        cv.split(X_aligned, cv_split_input), start=1
    ):
        fold_model = skbase.clone(model)
        fold_model.fit(X_aligned.iloc[train_pos], y_aligned.iloc[train_pos])
        y_pred_oof.iloc[test_pos] = fold_model.predict(X_aligned.iloc[test_pos])
        last_model = fold_model

    # ------------------------------------------------------------------
    # 5. MetricFrame on OOF predictions
    # ------------------------------------------------------------------
    metrics = _CLF_METRICS if bundle.task_type == "classification" else _REG_METRICS
    sf_aligned = bundle.sensitive_features.loc[common_idx]

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_aligned,
        y_pred=y_pred_oof,
        sensitive_features=sf_aligned,
    )

    return BundleResult(
        bundle=bundle,
        y_pred_oof=y_pred_oof,
        metric_frame=mf,
        cv_model=last_model,
        n_folds=actual_splits,
    )


# ===========================================================================
# SHAP explainability
# ===========================================================================

def compute_shap(
    bundle: FairlearnBundle,
    estimator,
    X: pd.DataFrame,
    store_shap_matrix: bool = True,
    max_shap_samples: int = 5000,
) -> ShapResult:
    """
    Refit the estimator on the full dataset, then compute SHAP values for
    explainability and per-group feature importance analysis.

    Why a separate refit model?
    ---------------------------
    The CV model from train_bundle() was trained on only ~80% of the data
    (the last fold's training split). For SHAP, a model trained on the full
    dataset produces more stable and representative attribution values.
    Critically, this refit model is kept strictly separate from the CV model
    used to produce OOF predictions — mixing them would cause data leakage
    into fairness metrics.

    Proxy detection
    ---------------
    compare_shap_across_groups() uses per_group_importances to surface features
    with low global importance but high importance within a specific group.
    These are proxy signals — the model is using a feature as a stand-in for
    the protected attribute (e.g. zip code proxying for race).

    Parameters
    ----------
    bundle:
        A non-skipped FairlearnBundle (same bundle used in train_bundle()).
    estimator:
        An unfitted sklearn-compatible estimator. A fresh clone is refit here.
    X:
        Encoded feature matrix. Must share an index with bundle.y_true.
    store_shap_matrix:
        If True, the full (n_samples, n_features) SHAP matrix is stored in
        ShapResult.shap_values. Set False to save memory on large datasets.
    max_shap_samples:
        Datasets larger than this are stratified-sampled before SHAP
        computation to keep runtime tractable while preserving group
        representation.

    Returns
    -------
    ShapResult with global_importances, per_group_importances, and optionally
    the raw shap_values matrix.

    Raises
    ------
    ImportError  if the `shap` package is not installed.
    ValueError   if bundle.skipped is True or X shares no index with the bundle.
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

    X_full = X.loc[common_idx]
    y_full = bundle.y_true.loc[common_idx]

    # ------------------------------------------------------------------
    # 2. Stratified sample if dataset exceeds max_shap_samples
    # ------------------------------------------------------------------
    X_shap = X_full
    if len(X_full) > max_shap_samples:
        sensitive_aligned = (
            bundle.sensitive_features.loc[common_idx]
            if isinstance(bundle.sensitive_features, pd.Series)
            else bundle.sensitive_features.loc[common_idx].apply(tuple, axis=1)
        )
        sampled_idx = (
            pd.Series(common_idx, index=common_idx)
            .groupby(sensitive_aligned)
            .apply(
                lambda g: g.sample(
                    min(len(g), max(1, int(max_shap_samples * len(g) / len(X_full)))),
                    random_state=42,
                )
            )
            .droplevel(0)
            .index
        )
        X_shap = X_full.loc[sampled_idx]
        warnings.warn(
            f"Dataset has {len(common_idx)} rows, exceeding max_shap_samples="
            f"{max_shap_samples}. SHAP computed on a stratified sample of "
            f"{len(X_shap)} rows.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 3. Full-data refit — separate model purely for explanation
    # ------------------------------------------------------------------
    refit_model = skbase.clone(estimator)
    refit_model.fit(X_full, y_full)

    # ------------------------------------------------------------------
    # 4. SHAP values via TreeExplainer
    # ------------------------------------------------------------------
    explainer = shap_lib.TreeExplainer(refit_model)
    raw_shap = explainer.shap_values(X_shap)

    # Multi-class: collapse list of per-class arrays to mean absolute
    if isinstance(raw_shap, list):
        shap_matrix = np.mean(np.abs(np.stack(raw_shap, axis=0)), axis=0)
    else:
        shap_matrix = raw_shap

    feature_names = list(X_shap.columns)

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
        group_labels = bundle.sensitive_features.loc[X_shap.index]
    else:
        group_labels = (
            bundle.sensitive_features.loc[X_shap.index]
            .astype(str)
            .apply(" | ".join, axis=1)
        )

    per_group_importances: dict[str, pd.Series] = {}
    for group_val, group_idx in group_labels.groupby(group_labels).groups.items():
        group_shap = shap_matrix[X_shap.index.get_indexer(group_idx)]
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
        n_samples=len(X_shap),
        refit_model=refit_model,
        shap_values=shap_matrix if store_shap_matrix else None,
    )


def compare_shap_across_groups(
    shap_result: ShapResult,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison of per-group SHAP importances.

    The primary tool for proxy detection: features with low global importance
    but high importance in a specific subgroup indicate the model treats that
    group differently, often via a proxy variable (e.g. zip code for race).

    Parameters
    ----------
    shap_result:
        Output of compute_shap().
    top_n:
        Number of top features to include, ranked by global importance.

    Returns
    -------
    pd.DataFrame
        Rows = top_n features (by global importance).
        Columns = each group + "global".
        Values = mean absolute SHAP value, rounded to 4 decimal places.
        Sorted by global importance descending.

    Example
    -------
    >>> comparison = compare_shap_across_groups(shap_result, top_n=15)
    >>> # High value in one group column but low "global" = proxy signal
    """
    top_features = shap_result.global_importances.head(top_n).index.tolist()
    data = {"global": shap_result.global_importances.loc[top_features]}

    for group, importances in shap_result.per_group_importances.items():
        # Reindex to top_features; fill missing with 0 if group had too few
        # samples for all feature values to be represented
        data[str(group)] = importances.reindex(top_features, fill_value=0.0)

    return pd.DataFrame(data, index=top_features).round(4)