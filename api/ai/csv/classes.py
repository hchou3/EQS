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
    protected_attr: str
    group_value: object
    n_samples: int
    target_col: str | None = None
 
 
@dataclass
class FairlearnBundle:
    protected_attr: str
    target_col: str
    task_type: str
    sensitive_features: pd.Series | pd.DataFrame
    y_true: pd.Series
    mode: SensitivityMode
    favorable_direction: str | None = None
    sparse_warnings: list[SparseGroupWarning] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
 
 
@dataclass
class FairlearnDataset:
    df: pd.DataFrame
    bundles: list[FairlearnBundle]
    mode: SensitivityMode
    all_sparse_warnings: list[SparseGroupWarning] = field(default_factory=list)
 
    def get_bundle(self, protected_attr: str, target_col: str) -> FairlearnBundle | None:
        for b in self.bundles:
            if b.protected_attr == protected_attr and b.target_col == target_col:
                return b
        return None
 
 
@dataclass
class ShapResult:
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
    bundle: FairlearnBundle
    y_pred_oof: pd.Series
    metric_frame: MetricFrame
    cv_model: object
    n_folds: int

@dataclass
class DisparityComparison:
    protected_attr: str
    target_col: str
    task_type: str

    # Raw data signals (pre-training)
    raw_group_values: dict[str, float]   # per-group mean/rate
    raw_disparity: float                  # max - min in raw data

    # Model prediction signals (post-training)
    model_group_values: dict[str, float] # per-group predicted mean/rate
    model_outcome_disparity: float        # max - min in predictions

    # Amplification — the key comparison
    amplification_ratio: float | None    # model_disparity / raw_disparity
    amplification_direction: str         # "amplified" | "attenuated" | "neutral"

    # Performance disparity — separate signal
    performance_disparity: dict[str, float]  # mf.difference() per metric
    performance_by_group: pd.DataFrame       # mf.by_group

    # Per-group breakdown — where the amplification is coming from
    per_group_shift: dict[str, float]    # model_mean - raw_mean per group