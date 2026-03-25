import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from fairlearn.metrics import MetricFrame
# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
SensitivityMode = Literal["individual", "intersectional"]

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
@dataclass
class SparseGroupWarning:
    """Describes a group that has too few samples for reliable metrics."""
 
    protected_attr: str
    group_value: object
    n_samples: int
    target_col: str | None = None  # None means the warning is attr-level
 
 
@dataclass
class FairlearnBundle:
    """
    Everything fairlearn needs for one (protected_attr, target_col) pair,
    plus provenance metadata.
 
    In intersectional mode, a bundle may be marked skipped=True when the
    fraction of sparse groups exceeds INTERSECTIONAL_SKIP_THRESHOLD. Skipped
    bundles carry their sparse_warnings for inspection but are excluded from
    MetricFrame computation and training.
    """
 
    protected_attr: str
    target_col: str
    task_type: str                          # "classification" | "regression"
    sensitive_features: pd.Series | pd.DataFrame
    y_true: pd.Series
    mode: SensitivityMode
    favorable_direction: str | None = None  # "higher" | "lower" | "neutral" | None
    sparse_warnings: list[SparseGroupWarning] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
 
 
@dataclass
class FairlearnDataset:
    """
    The full prepared dataset: a bundle for every
    (protected_attr × target_col) combination, plus the validated DataFrame.
    """
 
    df: pd.DataFrame
    bundles: list[FairlearnBundle]
    mode: SensitivityMode
    all_sparse_warnings: list[SparseGroupWarning] = field(default_factory=list)
 
    # Convenience: look up a specific bundle quickly
    def get_bundle(
        self, protected_attr: str, target_col: str
    ) -> FairlearnBundle | None:
        for b in self.bundles:
            if b.protected_attr == protected_attr and b.target_col == target_col:
                return b
        return None

@dataclass
class ShapResult:
    """
    Stores SHAP outputs for one (protected_attr, target_col) bundle.
 
    Attributes
    ----------
    protected_attr:
        The protected attribute this result corresponds to (mirrors the bundle).
    target_col:
        The target column this result corresponds to (mirrors the bundle).
    global_importances : pd.Series
        Mean absolute SHAP value per feature, sorted descending. Use this as
        your primary feature importance — it is more reliable than the native
        importance from XGBoost/LightGBM because it accounts for interaction
        effects and is consistent across model types.
    per_group_importances : dict[str, pd.Series]
        Mapping of group label → mean absolute SHAP value per feature for rows
        belonging to that group. Compare across groups to surface proxy features
        whose influence is concentrated in one demographic subgroup.
    shap_values : np.ndarray or None
        Full (n_samples, n_features) SHAP value matrix. None if
        store_shap_matrix=False was passed to compute_shap(). Needed for
        waterfall plots, force plots, and individual-level explanation.
    feature_names : list[str]
        Ordered list of feature names corresponding to columns in shap_values.
    n_samples : int
        Number of rows the SHAP model was trained and explained on.
    refit_model:
        The full-data refitted estimator used for SHAP computation. Stored so
        callers can run additional explanations (e.g. dependence plots) without
        refitting. Not the same model as the CV model used for MetricFrame.
    """
 
    protected_attr: str
    target_col: str
    global_importances: pd.Series
    per_group_importances: dict[str, pd.Series]
    feature_names: list[str]
    n_samples: int
    refit_model: object
    shap_values: np.ndarray | None = None


@dataclass
class BundleResult:
    """
    All outputs from training one (protected_attr, target_col) bundle.
 
    Separates CV-based evaluation outputs (used for fairness metrics) from
    the full-data refit model (used exclusively for SHAP). The two models
    are intentionally kept apart — mixing them would either leak training
    data into performance metrics or produce SHAP values from an
    underrepresented model.
 
    Attributes
    ----------
    bundle:
        Reference back to the source FairlearnBundle for provenance.
    y_pred_oof : pd.Series
        Out-of-fold predictions aligned to bundle.y_true.index. Every row
        was predicted on held-out data, making these unbiased estimates
        suitable for MetricFrame and fairness metric computation.
    metric_frame : MetricFrame
        Fairlearn MetricFrame computed from y_pred_oof vs bundle.y_true,
        broken down by bundle.sensitive_features.
    cv_model:
        The estimator fitted on the last CV fold. Retained for inspection
        but NOT used for SHAP — see refit_model on ShapResult instead.
    n_folds : int
        Number of CV folds actually used (may be less than CV_N_SPLITS if
        a class had too few samples for full stratification).
    """
 
    bundle: FairlearnBundle
    y_pred_oof: pd.Series
    metric_frame: MetricFrame
    cv_model: object
    n_folds: int