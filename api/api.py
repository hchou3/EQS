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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Support both localhost formats
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],  # Added GET for provider-config
    allow_headers=["*"],
    expose_headers=["*"],  # Ensure responses can include necessary headers
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
    """Debug endpoint to show loaded keys"""
    debug_info = {
        "key_checks": {
            "gemini": bool(settings.gemini_api_key),
            "groq": bool(settings.groq_api_key),
            "openai": bool(settings.openai_api_key)
        },
        "working_dir": os.getcwd(),
        "env_file": ".env",
        "env_exists": os.path.exists(".env"),
        # NEW: Log actual .env content (first 50 chars)
        "env_content": open(".env").read()[:50] if os.path.exists(".env") else "NOT FOUND"
    }
    print("DEBUG: provider-config endpoint called with keys:", debug_info["key_checks"])
    return debug_info

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