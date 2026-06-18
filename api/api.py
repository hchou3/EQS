from fastapi import FastAPI

from .routes.csv_upload import router as csv_upload_router
from .routes.bias_metrics import router as bias_metrics_router
from .routes.csv_train import router as csv_train_router
from .routes.csv_shap import router as csv_shap_router
from .routes.job_polling import router as jobs_router

app = FastAPI(title="Bias Detection API")

app.include_router(csv_upload_router)
app.include_router(bias_metrics_router)
app.include_router(csv_train_router)
app.include_router(csv_shap_router)
app.include_router(jobs_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}