from fastapi import FastAPI

from .routes.csv_upload import router as csv_upload_router
from .routes.bias_metrics import router as bias_metrics_router
from .routes.csv_train import router as csv_train_router
from .routes.csv_shap import router as csv_shap_router
from .routes.job_polling import router as jobs_router

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

# Add a simple test endpoint
@app.get("/test")
async def test_endpoint() -> dict:
    return {"message": "test"}