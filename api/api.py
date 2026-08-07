import os
from fastapi import FastAPI

from .routes.csv_upload import router as csv_upload_router
from .routes.bias_metrics import router as bias_metrics_router
from .routes.csv_train import router as csv_train_router
from .routes.csv_shap import router as csv_shap_router
from .routes.job_polling import router as jobs_router
from .config import settings

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Bias Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ⬅️ Update to match your React dev server
    allow_credentials=True,
    allow_methods=["POST"],  # Add other methods if needed
    allow_headers=["*"],
)

app.include_router(csv_upload_router)
app.include_router(bias_metrics_router)
app.include_router(csv_train_router)
app.include_router(csv_shap_router)
app.include_router(jobs_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/provider-config")
async def provider_config() -> dict:
    """
    Returns which LLM providers have API keys configured on the backend.
    Used by the frontend to decide whether to prompt the user for an API key.
    Exposes only boolean flags—never the actual secrets.
    """
    return {
        "providers": {
            "gemini": bool(getattr(settings, "gemini_api_key", None)),
            "groq": bool(getattr(settings, "groq_api_key", None)),
            "openai": bool(getattr(settings, "openai_api_key", None)),
        },
        "debug": {
            "gemini_api_key": getattr(settings, "gemini_api_key", None),
            "groq_api_key": getattr(settings, "groq_api_key", None),
            "openai_api_key": getattr(settings, "openai_api_key", None),
            "working_dir": os.getcwd(),
            "env_file_exists": os.path.exists(".env"),
        }
    }


@app.get("/debug/settings")
async def debug_settings() -> dict:
    """Debug endpoint to see all loaded settings"""
    import os
    return {
        "settings_module": str(settings.__dict__),
        "working_dir": os.getcwd(),
        "env_file": ".env",
        "env_exists": os.path.exists(".env"),
        "env_content_preview": (open(".env").read()[:200] if os.path.exists(".env") else "NOT FOUND"),
    }

# Add a simple test endpoint
@app.get("/test")
async def test_endpoint() -> dict:
    return {"message": "test"}