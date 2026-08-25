"""
CSV upload endpoint.

Responsibility boundary: this module only handles the *fast* path —
file ingestion, LLM column classification, and statistical bias analysis.
Model training (train_bundle) and SHAP are deliberately excluded; those
are separate, slower endpoints the user opts into (see csv_train.py).
"""
from __future__ import annotations

import os
import tempfile
import warnings as warnings_module

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import (
    ColumnReasoning,
    CSVUploadResponse,
    DatasetInfoSummary,
    DisparitySummary,
    FavorableDirectionInfo,
    LLMClassification,
)
from .sessions import session_store


from ..ai.csv.csv_processing import CSVData
from ..config import settings

router = APIRouter(prefix="/csv", tags=["csv"])

ALLOWED_EXTENSIONS = {".csv"}



def _build_disparity_summary(bias_results: dict) -> List[DisparitySummary]:
    """
    Flatten the nested bias_results dict into a list the frontend can
    iterate directly without knowing the nested protected_attr -> target_col
    structure.
    """
    summary: list[DisparitySummary] = []
    for protected_attr, targets in (bias_results or {}).items():
        for target_col, metrics in targets.items():
            if not isinstance(metrics, dict):
                continue
            if "error" in metrics:
                summary.append(
                    DisparitySummary(
                        protected_attr=protected_attr,
                        target_col=target_col,
                        task_type="unknown",
                        error=metrics["error"],
                    )
                )
                continue

            task_type = metrics.get("task_type", "classification")
            disparity = metrics.get("disparity")
            p_value = None
            if "anova" in metrics:
                task_type = "regression"
                p_value = metrics["anova"].get("p_value")

            summary.append(
                DisparitySummary(
                    protected_attr=protected_attr,
                    target_col=target_col,
                    task_type=task_type,
                    disparity=disparity,
                    p_value=p_value,
                )
            )
    return summary


@router.post(
    "/upload",
    response_model=CSVUploadResponse,
    responses={400: {"description": "Invalid file or unparseable CSV"}},
)
async def upload_csv(
    file: UploadFile = File(...),
    llm_provider: str = Form(...),
    llm_api_key: str = Form(...),
) -> CSVUploadResponse:
    """
    Accept a CSV upload and run the fast preprocessing pipeline:

    1. Parse the file into a DataFrame
    2. Profile columns (dataset_info)
    3. Classify columns via LLM (protected attrs, targets, task types)
    4. Encode features for later training
    5. Run statistical bias analysis (no model training yet)

    Returns a session_id that all subsequent calls (/train, /shap,
    /bias-metrics, /chat) use to reference this dataset.

    Does NOT train a model — that's a separate, slower step the user
    opts into via POST /csv/{session_id}/train.
    """
    # --- Extract provider and API key from request ---
    provider = llm_provider  # No fallback - llm_provider is required
    api_key = llm_api_key or getattr(settings, f"{provider}_api_key", None)

    # DEBUG: Log what's happening
    print(f"DEBUG: provider={provider}, api_key={api_key}, llm_provider={llm_provider}, llm_api_key={llm_api_key}")

    if not api_key or not api_key.strip():
        raise HTTPException(
            status_code=400,
            detail=f"LLM API key required for provider '{provider}'. "
                   f"Please provide it via frontend or set {provider.upper()}_API_KEY in .env",
        )

    # --- Validate file basics before touching disk ---
    if not file.filename or not any(
        file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {settings.max_upload_bytes // (1024*1024)}MB.",
        )

    # --- Write to a temp path; CSVData expects a filepath, not bytes ---
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    collected_warnings: list[str] = []

    try:
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")

            try:
                csv_data = CSVData(tmp_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not parse file as CSV: {e}",
                )

            if csv_data.df.empty or len(csv_data.df.columns) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="CSV parsed but contains no usable rows or columns.",
                )

            await csv_data.load_dataset_info()

            if not api_key or not api_key.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"LLM API key required for provider '{provider}'. "
                           f"Please provide it via frontend or set {provider.upper()}_API_KEY in .env",
                )

            # Determine which model to use for the selected provider
            PROVIDER_MODEL_MAP = {
                "gemini": "gemini-2.0-flash",
                "groq": "qwen/qwen3.8-27b",
                "openai": "gpt-4",
                # add more as needed
            }
            model_name = PROVIDER_MODEL_MAP.get(provider, settings.csv_classifier_model)

            try:
                await csv_data.identify_columns(
                    api_key=api_key,
                    model=model_name,
                    provider=provider,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Column classification failed: {e}",
                )

            if not csv_data.protected_attributes or not csv_data.target_columns:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "The classifier could not identify any protected attributes "
                        "or target columns in this dataset. Bias analysis requires "
                        "at least one of each."
                    ),
                )

            # Guard against the LLM naming columns that don't actually exist
            # in this DataFrame. Without this check, a hallucinated column
            # name silently passes /upload and only fails later inside
            # /train with a less informative error.
            available_cols = set(csv_data.df.columns)
            invalid_protected = [c for c in csv_data.protected_attributes if c not in available_cols]
            invalid_targets = [c for c in csv_data.target_columns if c not in available_cols]

            if invalid_protected or invalid_targets:
                collected_warnings.append(
                    f"Classifier returned column names not present in the dataset "
                    f"and they were dropped — protected: {invalid_protected or 'none'}, "
                    f"targets: {invalid_targets or 'none'}."
                )
                csv_data.protected_attributes = [
                    c for c in csv_data.protected_attributes if c in available_cols
                ]
                csv_data.target_columns = [
                    c for c in csv_data.target_columns if c in available_cols
                ]

            if not csv_data.protected_attributes or not csv_data.target_columns:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "After removing column names that don't exist in this dataset, "
                        "no valid protected attributes or target columns remained. "
                        f"Invalid protected: {invalid_protected or 'none'}; "
                        f"invalid targets: {invalid_targets or 'none'}."
                    ),
                )

            try:
                csv_data.encode_features()
            except Exception as e:
                # Encoding failure shouldn't block the user from seeing the
                # statistical results — training will be unavailable, but
                # bias analysis on raw columns still works.
                collected_warnings.append(f"Feature encoding failed: {e}")

            csv_data.run_bias_analysis()

            collected_warnings.extend(str(w.message) for w in caught)

    finally:
        os.unlink(tmp_path)

    # --- Persist into a new session ---
    session = session_store.create()
    session.csv_data = csv_data

    # --- Build response ---
    reasoning = csv_data.reasoning or {}
    favorable = csv_data.regression_favorable_directions or {}

    return CSVUploadResponse(
        session_id=session.session_id,
        filename=file.filename,
        dataset_info=DatasetInfoSummary(
            n_rows=csv_data.dataset_info["n_rows"],
            n_cols=csv_data.dataset_info["n_cols"],
            column_names=list(csv_data.dataset_info["column_info"].keys()),
        ),
        llm_classifications=LLMClassification(
            protected_attributes=csv_data.protected_attributes or [],
            target_columns=csv_data.target_columns or [],
            target_column_types=csv_data.target_column_types or {},
            reasoning=ColumnReasoning(**reasoning) if reasoning else ColumnReasoning(),
            regression_favorable_directions={
                k: FavorableDirectionInfo(**v) for k, v in favorable.items()
            },
        ),
        equity_score=csv_data.equity_score,
        disparity_summary=_build_disparity_summary(csv_data.bias_results),
        warnings=collected_warnings,
    )