from pydantic import BaseModel, Field


class Transaction(BaseModel):
    time: float = Field(..., description="Seconds elapsed between transaction and first transaction")
    amount: float = Field(..., gt=0, description="Transaction amount")
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    model_version: str


class DriftStatus(BaseModel):
    drift_detected: bool
    drift_score: float
    threshold: float
    buffer_size: int
    window_size: int
    last_check: str | None
    retrain_triggered: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class SimulateBatchRequest(BaseModel):
    count: int = Field(10, ge=1, le=500)
    inject_drift: bool = False


class RetrainWebhookPayload(BaseModel):
    secret: str
    drift_score: float | None = None
