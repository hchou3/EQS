from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from fairlearn.metrics import MetricFrame

from ..pretraining_tools.prompt import create_csv_prompt
from ..pretraining_tools.pretraining import parse_llm_response, llm_call
from .classes import (
    FairlearnBundle,
    FairlearnDataset,
    SensitivityMode,
    SparseGroupWarning,
)
from ...util import _to_json_safe

# ---------------------------------------------------------------------------
# Constants (shared with csv_training.py via import)
# ---------------------------------------------------------------------------

MIN_GROUP_SAMPLES = 30
INTERSECTIONAL_SKIP_THRESHOLD = 0.30
INTERSECTIONAL_MAX_ATTRIBUTES = 3

SENSITIVE_ATTRIBUTES = {
    "demographic": ["race", "ethnicity", "gender", "sex", "age", "age_group"],
    "socioeconomic": ["income", "zipcode", "education", "payer_code", "address", "city"],
    "health": ["disability", "medical_condition"],
    "other_protected": ["veteran_status", "marital_status", "religion"],
}


# ===========================================================================
# Stage 1 — Ingestion & profiling
# ===========================================================================

async def _prepare_dataset_info(df: pd.DataFrame, sample_size: int = 20) -> dict:
    """
    Build a structured column-level summary of *df* for LLM consumption.

    Returns a dict with keys:
        n_rows, n_cols, column_info (per-column metadata), sample_data (5 rows).
    """
    n_rows, n_cols = df.shape
    column_info: dict = {}

    for col in df.columns:
        col_data = df[col]
        missing_pct = round((col_data.isna().sum() / len(col_data)) * 100, 2)

        if pd.api.types.is_numeric_dtype(col_data):
            col_stats: dict = {
                "min": float(col_data.min()) if not col_data.isna().all() else None,
                "max": float(col_data.max()) if not col_data.isna().all() else None,
                "mean": float(col_data.mean()) if not col_data.isna().all() else None,
            }
        else:
            col_stats = {"value_distribution": col_data.value_counts().head(10).to_dict()}

        column_info[col] = {
            "dtype": str(col_data.dtype),
            "n_unique": col_data.nunique(),
            "sample_values": col_data.dropna().head(10).tolist(),
            "missing_pct": missing_pct,
            "stats": col_stats,
        }

    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "column_info": column_info,
        "sample_data": sample_df.head(5).to_dict("records"),
    }


async def analyze_dataframe(
    df: pd.DataFrame,
    api_key: str,
    model: str,
    sample_size: int = 20,
) -> dict:
    """
    Send column metadata to the LLM and return its column classifications.

    Returns
    -------
    dict with keys:
        protected_attributes, target_columns, reasoning,
        target_column_types, regression_favorable_directions.
    """
    dataset_info = await _prepare_dataset_info(df, sample_size)
    prompt = create_csv_prompt(dataset_info)
    response = await llm_call(prompt, api_key, model)
    return parse_llm_response(response)


# ===========================================================================
# Stage 2 — Validation helpers
# ===========================================================================

def _validate_columns(
    df: pd.DataFrame,
    protected_attributes: list[str],
    target_columns: list[str],
) -> tuple[list[str], list[str]]:
    """
    Drop any column names that are absent from *df* and warn the caller.

    Returns the surviving (protected_attributes, target_columns) lists.
    Prevents cryptic KeyErrors deep in the pipeline when the LLM hallucinates
    a column name or the user drops a column after classification.
    """
    available = set(df.columns)

    valid_protected: list[str] = []
    for col in protected_attributes:
        if col in available:
            valid_protected.append(col)
        else:
            warnings.warn(
                f"Protected attribute '{col}' not found in DataFrame — skipping.",
                UserWarning,
                stacklevel=3,
            )

    valid_targets: list[str] = []
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


# ===========================================================================
# Stage 3 — Statistical bias analysis (no model required)
# ===========================================================================

def _classification_bias_analysis(
    df: pd.DataFrame,
    protected_attr: str,
    target_col: str,
) -> dict:
    """
    Compute per-group outcome rates and disparity for a classification target.

    Disparity = max group rate − min group rate across all groups.
    For non-numeric targets the per-group value distribution is returned
    instead of a scalar rate.
    """
    subset = df[[protected_attr, target_col]].dropna()
    groups = subset.groupby(protected_attr)[target_col]

    if pd.api.types.is_numeric_dtype(subset[target_col]):
        group_means = groups.mean()
        return {
            "rate_by_group": group_means.to_dict(),
            "disparity": float(group_means.max() - group_means.min()) if len(group_means) else 0.0,
            "task_type": "classification",
        }

    # Categorical target: report normalized value distribution per group
    rate_by_group = (
        subset.groupby(protected_attr)[target_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .to_dict("index")
    )
    return {
        "rate_by_group": rate_by_group,
        "disparity": 0.0,
        "task_type": "classification",
    }


def _continuous_bias_analysis(
    df: pd.DataFrame,
    protected_attr: str,
    target_col: str,
) -> dict:
    """
    Run ANOVA, Kruskal-Wallis, and pairwise t-tests for a regression target.

    Returns a dict with:
        anova, kruskal_wallis, pairwise_comparisons, group_statistics.
    """
    groups = df.groupby(protected_attr)[target_col]
    group_data = [group for _, group in groups]

    f_stat, anova_p = stats.f_oneway(*group_data)
    h_stat, kw_p = stats.kruskal(*group_data)

    group_names = list(groups.groups.keys())
    pairwise: dict = {}
    for g1, g2 in combinations(group_names, 2):
        data1 = df[df[protected_attr] == g1][target_col]
        data2 = df[df[protected_attr] == g2][target_col]
        t_stat, t_p = stats.ttest_ind(data1, data2)
        cohens_d = (data1.mean() - data2.mean()) / (
            np.sqrt((data1.std() ** 2 + data2.std() ** 2) / 2)
        )
        pairwise[f"{g1}_vs_{g2}"] = {
            "t_statistic": t_stat,
            "p_value": t_p,
            "cohens_d": cohens_d,
            "mean_difference": data1.mean() - data2.mean(),
        }

    return {
        "anova": {"f_stat": f_stat, "p_value": anova_p},
        "kruskal_wallis": {"h_stat": h_stat, "p_value": kw_p},
        "pairwise_comparisons": pairwise,
        "group_statistics": groups.describe().to_dict(),
    }


def _calculate_bias_metrics(
    df: pd.DataFrame,
    protected_attributes: list[str],
    target_columns: list[str],
    target_column_types: dict[str, str],
) -> dict:
    """
    Dispatch to the appropriate statistical analysis for every
    (protected_attr, target_col) pair.

    Returns
    -------
    Nested dict: { protected_attr: { target_col: { ...metrics } } }
    """
    results: dict = {}

    for protected_attr in (protected_attributes or []):
        if protected_attr not in df.columns:
            continue
        results[protected_attr] = {}

        for target_col in (target_columns or []):
            if target_col not in df.columns:
                continue

            subset = df[[protected_attr, target_col]].dropna()
            if len(subset) < 2:
                results[protected_attr][target_col] = {
                    "error": "Insufficient data after dropping NaN"
                }
                continue

            task_type = (target_column_types or {}).get(target_col, "classification")

            if task_type == "regression" and pd.api.types.is_numeric_dtype(subset[target_col]):
                results[protected_attr][target_col] = _continuous_bias_analysis(
                    df, protected_attr, target_col
                )
            else:
                results[protected_attr][target_col] = _classification_bias_analysis(
                    df, protected_attr, target_col
                )

    return results


def _calculate_bias_score(detailed_bias_results: dict) -> float:
    """
    Aggregate per-pair bias metrics into a single equity score in [0, 100].

    Higher = more equitable (less disparity).
    Penalty sources:
      - Classification disparity value scaled to 0-100
      - Regression: low ANOVA p-value → high penalty
      - Missing/errored pairs: fixed 50-point penalty
    """
    if not detailed_bias_results:
        return 100.0

    penalties: list[float] = []
    for targets in detailed_bias_results.values():
        for metrics in targets.values():
            if not isinstance(metrics, dict) or "error" in metrics:
                penalties.append(50.0)
            elif "disparity" in metrics:
                penalties.append(min(100.0, abs(metrics["disparity"]) * 100))
            elif "anova" in metrics:
                p = metrics["anova"].get("p_value", 1.0)
                penalties.append(0.0 if p > 0.05 else 100.0 * (1.0 - p))
            else:
                penalties.append(0.0)

    overall_penalty = float(np.mean(penalties)) if penalties else 0.0
    return max(0.0, min(100.0, round(100.0 - overall_penalty, 2)))


# ===========================================================================
# Stage 4 — Sparse-group checking
# ===========================================================================

def _check_sparse_groups(
    sensitive_features: pd.Series | pd.DataFrame,
    protected_attr: str,
    target_col: str,
    min_samples: int = MIN_GROUP_SAMPLES,
) -> list[SparseGroupWarning]:
    """
    Return a SparseGroupWarning for every group below *min_samples*.

    Works for both individual mode (Series) and intersectional mode
    (DataFrame — groups are tuples of row values).
    """
    sparse: list[SparseGroupWarning] = []

    if isinstance(sensitive_features, pd.Series):
        counts = sensitive_features.value_counts()
    else:
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


# ===========================================================================
# Stage 5 — Sensitive-feature builders
# ===========================================================================

def _build_sensitive_features_individual(
    df: pd.DataFrame,
    protected_attr: str,
    idx: pd.Index,
) -> pd.Series:
    """Return a single-column Series aligned to *idx*."""
    return df.loc[idx, protected_attr]


def _build_sensitive_features_intersectional(
    df: pd.DataFrame,
    protected_attributes: list[str],
    idx: pd.Index,
) -> pd.DataFrame:
    """Return a multi-column DataFrame aligned to *idx*."""
    return df.loc[idx, protected_attributes]


# ===========================================================================
# Stage 6 — Bundle builders
# ===========================================================================

def _build_individual_bundles(
    csv_data: "CSVData",
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
            working_cols = [protected_attr, target_col]
            subset_idx = df[working_cols].dropna().index if drop_na else df.index

            sensitive_features = _build_sensitive_features_individual(
                df, protected_attr, subset_idx
            )
            y_true = df.loc[subset_idx, target_col]

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
    csv_data: "CSVData",
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
    If the fraction of intersectional groups below min_group_samples exceeds
    skip_threshold, the bundle is marked skipped=True and excluded from
    training. sparse_warnings are still populated for inspection.
    """
    bundles: list[FairlearnBundle] = []
    all_sparse: list[SparseGroupWarning] = []
    combined_attr_label = ", ".join(valid_protected)

    for target_col in valid_targets:
        task_type = (csv_data.target_column_types or {}).get(target_col, "classification")
        favorable = (csv_data.regression_favorable_directions or {}).get(target_col, {})
        favorable_direction = favorable.get("direction") if favorable else None

        working_cols = valid_protected + [target_col]
        subset_idx = df[working_cols].dropna().index if drop_na else df.index

        sensitive_features = _build_sensitive_features_intersectional(
            df, valid_protected, subset_idx
        )
        y_true = df.loc[subset_idx, target_col]

        sparse = _check_sparse_groups(
            sensitive_features, combined_attr_label, target_col, min_group_samples
        )
        all_sparse.extend(sparse)

        group_counts = sensitive_features.apply(tuple, axis=1).value_counts()
        n_total_groups = len(group_counts)
        sparse_fraction = len(sparse) / n_total_groups if n_total_groups > 0 else 1.0

        if sparse_fraction > skip_threshold:
            skip_reason = (
                f"{len(sparse)}/{n_total_groups} intersectional groups "
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


# ===========================================================================
# Stage 7 — Bundle orchestration
# ===========================================================================

def prepare_fairlearn_data(
    csv_data: "CSVData",
    mode: SensitivityMode = "individual",
    min_group_samples: int = MIN_GROUP_SAMPLES,
    drop_na: bool = True,
    intersectional_skip_threshold: float = INTERSECTIONAL_SKIP_THRESHOLD,
    allow_large_intersections: bool = False,
) -> FairlearnDataset:
    """
    Convert a CSVData object into a FairlearnDataset ready for training.

    Must be called after identify_columns() and encode_features().

    Parameters
    ----------
    csv_data:
        A CSVData instance with non-empty protected_attributes and target_columns.
    mode:
        "individual"     — one bundle per protected attribute (independent).
        "intersectional" — one bundle across all attributes simultaneously.
    min_group_samples:
        Groups below this size emit a SparseGroupWarning.
    drop_na:
        Drop rows where either the protected attribute or target is NaN.
        Each bundle may therefore use a slightly different data subset.
    intersectional_skip_threshold:
        Fraction of sparse intersectional groups allowed before skipping a bundle.
    allow_large_intersections:
        Override the INTERSECTIONAL_MAX_ATTRIBUTES cap. Run
        plot_intersectional_sparsity() first to confirm your dataset supports it.

    Returns
    -------
    FairlearnDataset containing one FairlearnBundle per pair (individual mode)
    or per target_col (intersectional mode).

    Raises
    ------
    ValueError
        If identify_columns() has not been called, or no valid columns remain.
    """
    if not csv_data.protected_attributes or not csv_data.target_columns:
        raise ValueError(
            "csv_data.protected_attributes and csv_data.target_columns are empty. "
            "Call csv_data.identify_columns() before prepare_fairlearn_data()."
        )

    valid_protected, valid_targets = _validate_columns(
        csv_data.df,
        csv_data.protected_attributes,
        csv_data.target_columns,
    )

    if not valid_protected:
        raise ValueError("No valid protected attributes found in the DataFrame.")
    if not valid_targets:
        raise ValueError("No valid target columns found in the DataFrame.")

    if mode == "intersectional" and len(valid_protected) > INTERSECTIONAL_MAX_ATTRIBUTES:
        if not allow_large_intersections:
            raise ValueError(
                f"Intersectional mode received {len(valid_protected)} protected "
                f"attributes ({', '.join(valid_protected)}), exceeding "
                f"INTERSECTIONAL_MAX_ATTRIBUTES={INTERSECTIONAL_MAX_ATTRIBUTES}. "
                f"Options:\n"
                f"  1. Limit csv_data.protected_attributes to ≤3 columns.\n"
                f"  2. Set allow_large_intersections=True after running "
                f"plot_intersectional_sparsity() to confirm your dataset supports it."
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

    for w in all_sparse:
        warnings.warn(
            f"Sparse group — protected_attr='{w.protected_attr}', "
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


# ===========================================================================
# Reporting
# ===========================================================================

def summarise_dataset(dataset: FairlearnDataset) -> dict:
    """
    Return a JSON-serializable summary of the prepared dataset.

    Useful for logging or displaying to the user before committing to training.
    Surfaces skipped bundles and sparse group counts so problems are visible
    before they become silent failures downstream.
    """
    bundle_summaries = []
    for b in dataset.bundles:
        if isinstance(b.sensitive_features, pd.Series):
            group_counts = b.sensitive_features.value_counts().to_dict()
        else:
            group_counts = {
                str(k): v
                for k, v in b.sensitive_features.apply(tuple, axis=1)
                .value_counts()
                .to_dict()
                .items()
            }

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
            "skipped": b.skipped,
            "skip_reason": b.skip_reason,
        })

    return {
        "mode": dataset.mode,
        "total_bundles": len(dataset.bundles),
        "skipped_bundles": sum(1 for b in dataset.bundles if b.skipped),
        "total_sparse_warnings": len(dataset.all_sparse_warnings),
        "bundles": bundle_summaries,
    }


# ===========================================================================
# CSVData — single-CSV container and pipeline entry point
# ===========================================================================

class CSVData:
    """
    Represents a single CSV file and drives the full preprocessing pipeline.

    Typical usage
    -------------
    >>> csv_data = CSVData("hospital.csv")
    >>> await csv_data.identify_columns(api_key=key, model="gemini-2.0-flash")
    >>> csv_data.encode_features()
    >>> csv_data.run_bias_analysis()
    >>> dataset = prepare_fairlearn_data(csv_data, mode="individual")
    """

    def __init__(self, filepath: str) -> None:
        """
        Load a CSV file and store it as both a working copy and an immutable
        original so encoding and column drops never corrupt the source data.
        """
        self.filepath = filepath
        self._df = pd.read_csv(filepath)
        self.raw_data = self._df.copy()

        # Populated by the pipeline in order
        self.dataset_info: dict | None = None
        self.protected_attributes: list[str] | None = None
        self.target_columns: list[str] | None = None
        self.target_column_types: dict[str, str] | None = None
        self.reasoning: dict | None = None
        self.regression_favorable_directions: dict | None = None
        self.original_feature_cols: list[str] | None = None
        self.encoded_feature_cols: list[str] | None = None
        self.bias_results: dict | None = None
        self.equity_score: float | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """Current working DataFrame (may have columns dropped or encoded)."""
        return self._df

    # ------------------------------------------------------------------
    # Stage 1 — Ingestion helpers
    # ------------------------------------------------------------------

    def drop_columns(self, columns: list[str]) -> None:
        """
        Drop columns from the working DataFrame in-place.
        Ignores names that are not present. Typically used to remove ID
        columns or other irrelevant features before encoding.
        """
        existing = [c for c in columns if c in self._df.columns]
        self._df = self._df.drop(columns=existing)

    # ------------------------------------------------------------------
    # Stage 2 — Dataset profiling
    # ------------------------------------------------------------------

    async def load_dataset_info(self, sample_size: int = 20) -> None:
        """
        Profile all columns and store the result in self.dataset_info.
        Subsequent calls to identify_columns() reuse this result rather
        than re-profiling the DataFrame.
        """
        self.dataset_info = await _prepare_dataset_info(self._df, sample_size)

    # ------------------------------------------------------------------
    # Stage 3 — LLM column classification
    # ------------------------------------------------------------------

    async def identify_columns(
        self,
        api_key: str,
        model: str,
        provider: str = "gemini",
        sample_size: int = 20,
    ) -> None:
        """
        Call the LLM to classify columns as protected attributes or targets.

        Reuses self.dataset_info if load_dataset_info() was already called,
        avoiding a second full pass over the DataFrame.

        Populates
        ---------
        self.protected_attributes, self.target_columns,
        self.target_column_types, self.reasoning,
        self.regression_favorable_directions
        """
        # Reuse existing profile to avoid redundant DataFrame pass
        if self.dataset_info is None:
            await self.load_dataset_info(sample_size)

        prompt = create_csv_prompt(self.dataset_info)
        response = await llm_call(prompt, api_key, model, provider)
        result = parse_llm_response(response)

        self.protected_attributes = result.get("protected_attributes") or []
        self.target_columns = result.get("target_columns") or []
        self.target_column_types = result.get("target_column_types") or {}
        self.reasoning = result.get("reasoning") or {}
        self.regression_favorable_directions = result.get("regression_favorable_directions") or {}

    # ------------------------------------------------------------------
    # Stage 4 — Feature encoding
    # ------------------------------------------------------------------

    def encode_features(self, drop_first: bool = False) -> None:
        """
        One-hot encode all feature columns, leaving protected attributes and
        targets in their original form.

        Must be called after identify_columns(). Always starts from raw_data
        to prevent double-encoding if called more than once.

        Parameters
        ----------
        drop_first:
            Pass True to drop the first dummy category per column, reducing
            multicollinearity. Default False preserves all categories.
        """
        if not self.protected_attributes or not self.target_columns:
            raise RuntimeError(
                "encode_features() must be called after identify_columns(). "
                f"protected_attributes={self.protected_attributes!r}, "
                f"target_columns={self.target_columns!r}."
            )

        df = self.raw_data.copy()

        skip_cols = set(self.protected_attributes) | set(self.target_columns)
        feature_cols = [c for c in df.columns if c not in skip_cols]
        preserved_cols = [c for c in df.columns if c in skip_cols]

        if not feature_cols:
            warnings.warn(
                "No feature columns remain after excluding protected attributes "
                "and target columns. encode_features() has nothing to encode.",
                UserWarning,
                stacklevel=2,
            )
            self.original_feature_cols = []
            self.encoded_feature_cols = []
            return

        self.original_feature_cols = feature_cols

        X_encoded = pd.get_dummies(df[feature_cols], drop_first=drop_first)

        # Cast bool → int for LightGBM compatibility
        bool_cols = X_encoded.select_dtypes(include="bool").columns
        if len(bool_cols):
            X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

        self.encoded_feature_cols = list(X_encoded.columns)
        self._df = pd.concat([X_encoded, df[preserved_cols]], axis=1)

        n_orig, n_enc = len(feature_cols), len(self.encoded_feature_cols)
        print(
            f"encode_features(): {n_orig} feature columns → {n_enc} encoded columns "
            f"(+{n_enc - n_orig} from categorical expansion). "
            f"{len(preserved_cols)} columns preserved unencoded "
            f"({', '.join(preserved_cols)})."
        )

    # ------------------------------------------------------------------
    # Stage 5 — Statistical bias analysis
    # ------------------------------------------------------------------

    def run_bias_analysis(self) -> None:
        """
        Run statistical bias tests across all (protected_attr, target_col) pairs
        and compute an overall equity score.

        Can be called before encode_features() since it operates on raw values,
        not encoded features. Populates self.bias_results and self.equity_score.
        """
        self.bias_results = _calculate_bias_metrics(
            self._df,
            self.protected_attributes,
            self.target_columns,
            self.target_column_types,
        )
        self.equity_score = _calculate_bias_score(self.bias_results)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_consolidated_report(self) -> dict:
        """
        Gather all analysis results into a single JSON-serializable dict.

        Includes dataset metadata, LLM classifications, statistical bias
        metrics, and the equity score. MetricFrame and SHAP outputs from
        the training stage are appended by the API layer.
        """
        report = {
            "filepath": self.filepath,
            "dataset_info": self.dataset_info,
            "llm_classifications": {
                "protected_attributes": self.protected_attributes or [],
                "target_columns": self.target_columns or [],
                "target_column_types": self.target_column_types or {},
                "reasoning": self.reasoning or {},
                "regression_favorable_directions": self.regression_favorable_directions or {},
            },
            "bias_metrics": self.bias_results or {},
            "equity_score": float(self.equity_score) if self.equity_score is not None else None,
        }
        return _to_json_safe(report)