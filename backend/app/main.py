import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    DriftStatus,
    HealthResponse,
    PredictionResponse,
    RetrainWebhookPayload,
    SimulateBatchRequest,
    Transaction,
)
from app.services.drift_monitor import DriftMonitor
from app.services.predictor import FraudPredictor, train_and_log_model
from app.services.retrain_trigger import trigger_github_retrain
from scripts.generate_data import generate_drifted_batch, generate_normal_batch, row_to_transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = FraudPredictor()
drift_monitor = DriftMonitor()


async def _handle_drift_retrain(drift_score: float) -> None:
    drift_monitor.mark_retrain_triggered()
    result = await trigger_github_retrain(drift_score)
    logger.info("Retrain trigger result: %s", result)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fraud detection API started — model version: %s", predictor.model_version)
    yield


app = FastAPI(
    title="Autonomous ML Platform",
    description="Real-time fraud detection with Evidently drift monitoring and automated retraining",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: Transaction,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    features = transaction.model_dump()
    proba, is_fraud = predictor.predict(features)

    drift_monitor.append_transaction(features)

    # Check drift only when the window is full and we haven't already checked this fill
    if (
        drift_monitor.buffer_size >= settings.drift_window_size
        and not drift_monitor.retrain_triggered
        and drift_monitor.should_check()
    ):
        status = drift_monitor.check_drift()
        if status["drift_detected"] and not status["retrain_triggered"]:
            background_tasks.add_task(_handle_drift_retrain, status["drift_score"])

    return PredictionResponse(
        transaction_id=str(uuid.uuid4()),
        fraud_probability=round(proba, 4),
        is_fraud=is_fraud,
        model_version=predictor.model_version,
    )


@app.get("/drift", response_model=DriftStatus)
def get_drift_status() -> DriftStatus:
    # Return cached status for polling; simulate/predict trigger real checks
    return DriftStatus(**drift_monitor.status())


@app.post("/simulate", response_model=list[PredictionResponse])
async def simulate_batch(
    request: SimulateBatchRequest,
    background_tasks: BackgroundTasks,
) -> list[PredictionResponse]:
    if request.inject_drift:
        batch = generate_drifted_batch(n_samples=request.count)
    else:
        batch = generate_normal_batch(n_samples=request.count)

    responses = []
    for _, row in batch.iterrows():
        txn = Transaction(**row_to_transaction(row))
        responses.append(await predict(txn, background_tasks))
    return responses


@app.post("/retrain/webhook")
async def retrain_webhook(payload: RetrainWebhookPayload) -> dict:
    """Local fallback: trigger retraining without GitHub Actions."""
    if payload.secret != settings.retrain_webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    from pathlib import Path

    try:
        train_df = drift_monitor.training_frame()
        data_path = Path(settings.data_dir) / "retrain.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(data_path, index=False)

        result = train_and_log_model(str(data_path))
        version = predictor.reload(version=result["model_version"])
        drift_monitor.reset_after_retrain()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Retrain failed")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {exc}") from exc

    return {
        "status": "retrained",
        "model_version": version,
        "mlflow_run_id": result["run_id"],
        "metrics": result["metrics"],
    }


@app.post("/model/reload")
def reload_model() -> dict:
    version = predictor.reload()
    return {"model_version": version, "status": "reloaded"}
