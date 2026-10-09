from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Local default; Docker Compose overrides to http://mlflow:5000
    mlflow_tracking_uri: str = "file:./mlruns"
    model_registry_name: str = "fraud-detector"
    model_artifact_path: str = "artifacts/model.joblib"
    model_version_path: str = "artifacts/version.txt"
    reference_data_path: str = "data/reference.csv"
    drift_buffer_path: str = "data/drift_buffer.csv"
    drift_threshold: float = 0.25
    drift_window_size: int = 100
    github_repo: str = ""
    github_token: str = ""
    github_workflow_id: str = "retrain-on-drift.yml"
    retrain_webhook_secret: str = "change-me-in-production"

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.model_artifact_path).parent

    @property
    def data_dir(self) -> Path:
        return Path(self.reference_data_path).parent


settings = Settings()
