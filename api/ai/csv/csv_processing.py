import pandas as pd
import numpy as np
from scipy import stats
import pdfplumber
from ...ai.pretraining_tools.prompt import create_csv_prompt
from ...ai.pretraining_tools.pretraining import parse_llm_response, llm_call
from ...ai.csv.classes import _to_json_safe
import warnings

SENSITIVE_ATTRIBUTES = {
    'demographic': ['race', 'ethnicity', 'gender', 'sex', 'age', 'age_group'],
    'socioeconomic': ['income', 'zipcode', 'education', 'payer_code', 'address', 'city'],
    'health': ['disability', 'medical_condition'],
    'other_protected': ['veteran_status', 'marital_status', 'religion']

}

def encode_features(
    self,
    drop_first: bool = False,
) -> None:
    """
    One-hot encode all feature columns in the working DataFrame, leaving
    protected attributes and target columns in their original form.
 
    Must be called AFTER identify_columns() so that protected_attributes
    and target_columns are known. The encoded DataFrame replaces self._df
    in-place so all downstream steps (prepare_fairlearn_data, train_bundle,
    compute_shap) read from a uniform, fully-encoded feature space.
    """
    if self.protected_attributes is None or self.target_columns is None:
        raise RuntimeError(
            "encode_features() must be called after identify_columns(). "
            "protected_attributes and target_columns are not set yet."
        )
 
    # Always start from raw_data to prevent double-encoding if called again
    df = self.raw_data.copy()
 
    # Columns that must remain unencoded
    skip_cols = (
        set(self.protected_attributes or [])
        | set(self.target_columns or [])
    )
 
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
 
    # --- Encode ---
    X_raw = df[feature_cols]
    X_encoded = pd.get_dummies(X_raw, drop_first=drop_first)
 
    # Cast bool → int for XGBoost / LightGBM compatibility
    bool_cols = X_encoded.select_dtypes(include="bool").columns
    if len(bool_cols):
        X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)
 
    self.encoded_feature_cols = list(X_encoded.columns)
 
    # Reconstruct working DataFrame: encoded features + preserved cols
    self._df = pd.concat([X_encoded, df[preserved_cols]], axis=1)
 
    n_orig = len(feature_cols)
    n_enc = len(self.encoded_feature_cols)
    print(
        f"encode_features(): {n_orig} feature columns → {n_enc} encoded columns "
        f"(+{n_enc - n_orig} from categorical expansion). "
        f"{len(preserved_cols)} columns preserved unencoded "
        f"({', '.join(preserved_cols)})."
    )

async def prepare_df(data):
    file_path = data if isinstance(data, str) else getattr(data, 'name', '')

    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(data)
    elif file_path.lower().endswith('.pdf'):
        all_tables = []
        with pdfplumber.open(data) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables or []:
                    if table:
                        df_page = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append(df_page)
        df = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    else:
        raise ValueError(f"Unsupported file type. Expected .csv or .pdf, got: {file_path or type(data).__name__}")

    return df

async def _prepare_dataset_info(df, sample_size):
        """
        Prepare comprehensive dataset information for LLM analysis.
        """
        n_rows, n_cols = df.shape

        # Column information
        column_info = {}
        for col in df.columns:
            col_data = df[col]

            dtype = str(col_data.dtype)

            n_unique = col_data.nunique()

            sample_values = col_data.dropna().head(10).tolist()

            # Calculate missing pctages
            missing_pct = (col_data.isna().sum() / len(col_data)) * 100

            # For numeric columns, get statistics
            if pd.api.types.is_numeric_dtype(col_data):
                stats = {
                    'min': float(col_data.min()) if not col_data.isna().all() else None,
                    'max': float(col_data.max()) if not col_data.isna().all() else None,
                    'mean': float(col_data.mean()) if not col_data.isna().all() else None
                }
            else:
                # For categorical, get value counts
                value_counts = col_data.value_counts().head(10).to_dict()
                stats = {'value_distribution': value_counts}

            column_info[col] = {
                'dtype': dtype,
                'n_unique': n_unique,
                'sample_values': sample_values,
                'missing_pct': round(missing_pct, 2),
                'stats': stats
            }

        # Sample rows for context
        sample_df = df.sample(min(sample_size, len(df)))
        sample_data = sample_df.head(5).to_dict('records')

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "column_info": column_info,
            "sample_data": sample_data,
        }


async def analyze_dataframe(
    df: pd.DataFrame,
    api_key: str,
    model: str,
    sample_size: int = 20,
) -> dict:
    """
    Analyze a dataframe to identify bias-related columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to analyze.
    api_key : str
        API key for the LLM provider.
    model : str
        Model name (e.g., gemini-2.0-flash).
    sample_size : int, optional
        Number of rows to sample for LLM analysis (default: 20).

    Returns
    -------
    dict
        protected_attributes, target_columns, reasoning, target_column_types,
        regression_favorable_directions.
    """
    dataset_info = await _prepare_dataset_info(df, sample_size)
    prompt = create_csv_prompt(dataset_info)
    response = await llm_call(prompt, api_key, model)
    return parse_llm_response(response)

def continuous_bias_analysis(df, protected_attr, target_col):
    """
    Statistical tests for continuous variable bias
    """
    groups = df.groupby(protected_attr)[target_col]

    # 1. ANOVA - Are means different across groups?
    f_stat, p_value = stats.f_oneway(*[group for name, group in groups])

    # 2. Kruskal-Wallis - Non-parametric alternative
    h_stat, kw_p_value = stats.kruskal(*[group for name, group in groups])

    # 3. Pairwise comparisons
    group_names = list(groups.groups.keys())
    pairwise = {}
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i+1:]:
            data1 = df[df[protected_attr] == g1][target_col]
            data2 = df[df[protected_attr] == g2][target_col]

            # T-test
            t_stat, t_p = stats.ttest_ind(data1, data2)

            # Cohen's d (effect size)
            cohens_d = (data1.mean() - data2.mean()) / \
                       np.sqrt((data1.std()**2 + data2.std()**2) / 2)

            pairwise[f"{g1}_vs_{g2}"] = {
                't_statistic': t_stat,
                'p_value': t_p,
                'cohens_d': cohens_d,
                'mean_difference': data1.mean() - data2.mean()
            }

    return {
        'anova': {'f_stat': f_stat, 'p_value': p_value},
        'kruskal_wallis': {'h_stat': h_stat, 'p_value': kw_p_value},
        'pairwise_comparisons': pairwise,
        'group_statistics': groups.describe().to_dict()
    }


def _calculate_bias_metrics(
    df: pd.DataFrame,
    protected_attributes: list,
    target_columns: list,
    target_column_types: dict,
) -> dict:
    """
    Compute bias metrics for each (protected_attribute, target) pair.

    For classification targets: computes per-group positive/outcome rate and
    disparity (max - min across groups). For regression targets: uses
    continuous_bias_analysis (ANOVA, Kruskal-Wallis, pairwise comparisons).

    Returns
    -------
    dict
        Nested structure: { "protected_attr": { "target_col": { ... metrics } } }
    """
    results = {}
    for protected_attr in protected_attributes or []:
        if protected_attr not in df.columns:
            continue
        results[protected_attr] = {}
        for target_col in target_columns or []:
            if target_col not in df.columns:
                continue
            # Drop rows where protected or target is missing for this pair
            subset = df[[protected_attr, target_col]].dropna()
            if len(subset) < 2:
                results[protected_attr][target_col] = {"error": "Insufficient data after dropping NaN"}
                continue
            task_type = (target_column_types or {}).get(target_col, "classification")
            if task_type == "regression" and pd.api.types.is_numeric_dtype(subset[target_col]):
                results[protected_attr][target_col] = continuous_bias_analysis(
                    df, protected_attr, target_col
                )
            else:
                # Classification or categorical: outcome rate per group and disparity
                group_means = subset.groupby(protected_attr)[target_col].mean()
                if pd.api.types.is_numeric_dtype(subset[target_col]):
                    rate_by_group = group_means.to_dict()
                    disparity = float(group_means.max() - group_means.min()) if len(group_means) else 0
                else:
                    # Categorical: use value_counts and largest group share per group
                    rate_by_group = (
                        subset.groupby(protected_attr)[target_col]
                        .value_counts(normalize=True)
                        .unstack(fill_value=0)
                        .to_dict("index")
                    )
                    disparity = 0  # Placeholder; could compute max difference in modal share
                results[protected_attr][target_col] = {
                    "rate_by_group": rate_by_group,
                    "disparity": disparity,
                    "task_type": task_type,
                }
    return results


def _calculate_bias_score(detailed_bias_results: dict) -> float:
    """
    Aggregate detailed bias metrics into a single equity score in [0, 100].
    Higher score = more equitable (less disparity).

    Uses disparity values and p-values where available; converts to a
    penalty and then score = 100 - penalty.
    """
    if not detailed_bias_results:
        return 100.0
    penalties = []
    for protected_attr, targets in detailed_bias_results.items():
        for target_col, metrics in targets.items():
            if isinstance(metrics, dict) and "error" in metrics:
                penalties.append(50)  # Unknown/missing data
                continue
            if "disparity" in metrics:
                d = metrics["disparity"]
                penalties.append(min(100, abs(d) * 100))  # Scale disparity into penalty
            elif "anova" in metrics and "p_value" in metrics["anova"]:
                p = metrics["anova"]["p_value"]
                penalties.append(0 if p > 0.05 else (100 * (1 - p)))  # Low p -> higher penalty
            else:
                penalties.append(0)
    overall_penalty = np.mean(penalties) if penalties else 0
    return max(0, min(100, round(100 - overall_penalty, 2)))


# -----------------------------------------------------------------------------
# CSVData: single-CSV container with integrated analysis pipeline
# -----------------------------------------------------------------------------
class CSVData:
    """
    Represents a single CSV file with methods to load, preprocess, and run
    the full bias/equity analysis pipeline. Designed as a building block for
    FastAPI endpoints that process uploaded CSV files.
    """

    def __init__(self, filepath: str):
        """
        Load a CSV file into a pandas DataFrame and store it as raw data.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        """
        self.filepath = filepath
        self._df = pd.read_csv(filepath)
        self.raw_data = self._df.copy()

        # Populated by the pipeline
        self.dataset_info = None
        self.protected_attributes = None
        self.target_columns = None
        self.target_column_types = None
        self.reasoning = None
        self.regression_favorable_directions = None
        self.bias_results = None
        self.equity_score = None

    @property
    def df(self) -> pd.DataFrame:
        """Current working DataFrame (may have columns dropped)."""
        return self._df

    def drop_columns(self, columns: list) -> None:
        """
        Drop specified columns from the working DataFrame (in-place).
        Mimics typical notebook preprocessing that removes ID or irrelevant columns.

        Parameters
        ----------
        columns : list
            Column names to drop. Ignores names that are not present.
        """
        existing = [c for c in columns if c in self._df.columns]
        self._df = self._df.drop(columns=existing)

    async def load_dataset_info(self, sample_size: int = 20) -> None:
        """
        Use _prepare_dataset_info to generate a summary of columns and sample data.
        Stores the result in self.dataset_info.
        """
        self.dataset_info = await _prepare_dataset_info(self._df, sample_size)

    async def identify_columns(
        self,
        api_key: str,
        model: str,
        sample_size: int = 20,
    ) -> None:
        """
        Call the LLM via analyze_dataframe to get protected attributes, target
        columns, and their types. Stores all classifications on this instance.
        """
        result = await analyze_dataframe(
            self._df, api_key=api_key, model=model, sample_size=sample_size
        )
        self.protected_attributes = result.get("protected_attributes") or []
        self.target_columns = result.get("target_columns") or []
        self.target_column_types = result.get("target_column_types") or {}
        self.reasoning = result.get("reasoning") or {}
        self.regression_favorable_directions = result.get("regression_favorable_directions") or {}

    def run_bias_analysis(self) -> None:
        """
        Run bias analysis using the stored protected attributes, target columns,
        and types. Calls _calculate_bias_metrics and _calculate_bias_score.
        Stores detailed bias results and overall equity score on this instance.
        """
        self.bias_results = _calculate_bias_metrics(
            self._df,
            self.protected_attributes,
            self.target_columns,
            self.target_column_types,
        )
        self.equity_score = _calculate_bias_score(self.bias_results)

    def get_consolidated_report(self) -> dict:
        """
        Gather all analysis results into a single JSON-serializable dictionary:
        dataset info, LLM classifications, detailed bias metrics, and equity score.
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


