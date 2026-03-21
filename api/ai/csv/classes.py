import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

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