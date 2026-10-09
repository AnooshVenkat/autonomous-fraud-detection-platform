import logging
import threading
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from app.config import settings
from app.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _read_version() -> str:
    path = Path(settings.model_version_path)
    if path.exists():
        return path.read_text(encoding="utf-8").strip() or "v1"
    return "v1"


def _next_version(current: str | None = None) -> str:
    current = current or _read_version()
    try:
        n = int(current.lstrip("v").split("-")[-1])
    except ValueError:
        n = 1
    return f"v{n + 1}"


def _write_version(version: str) -> None:
    path = Path(settings.model_version_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(version, encoding="utf-8")


class FraudPredictor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: LGBMClassifier | None = None
        self._model_version = "unloaded"
        self._load_model()

    def _load_model(self) -> None:
        artifact_path = Path(settings.model_artifact_path)
        if artifact_path.exists():
            with self._lock:
                self._model = joblib.load(artifact_path)
                self._model_version = _read_version()
                return

        self._train_bootstrap_model(artifact_path)

    def _train_bootstrap_model(self, artifact_path: Path) -> None:
        from scripts.generate_data import generate_reference_dataset

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        Path(settings.reference_data_path).parent.mkdir(parents=True, exist_ok=True)

        df = generate_reference_dataset(n_samples=5000, fraud_rate=0.02, seed=42)
        df.to_csv(settings.reference_data_path, index=False)

        X = df[FEATURE_COLUMNS]
        y = df["class"]

        model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)

        version = "v1"
        joblib.dump(model, artifact_path)
        _write_version(version)
        with self._lock:
            self._model = model
            self._model_version = version

    def reload(self, version: str | None = None) -> str:
        artifact_path = Path(settings.model_artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}")

        new_model = joblib.load(artifact_path)
        new_version = version or _read_version()
        with self._lock:
            self._model = new_model
            self._model_version = new_version
        return self._model_version

    def predict(self, features: dict) -> tuple[float, bool]:
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded")

            row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
            proba = float(self._model.predict_proba(row)[0][1])
            return proba, proba >= 0.5

    @property
    def model_version(self) -> str:
        return str(self._model_version)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


def train_and_log_model(
    data_path: str,
    experiment_name: str = "fraud-detection",
) -> dict:
    df = pd.read_csv(data_path)
    if "class" not in df.columns:
        raise ValueError("Training data is missing the class label column")

    df = df.dropna(subset=FEATURE_COLUMNS + ["class"]).copy()
    df["class"] = df["class"].astype(int)
    if len(df) < 50:
        raise ValueError(f"Not enough clean training rows after dropping NaNs: {len(df)}")

    X = df[FEATURE_COLUMNS]
    y = df["class"]

    split_idx = max(1, int(len(df) * 0.8))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LGBMClassifier(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.08,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = model.predict(X_test) if len(X_test) else model.predict(X_train)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if len(X_test)
        else model.predict_proba(X_train)[:, 1]
    )
    y_eval = y_test if len(y_test) else y_train

    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
        "f1": float(f1_score(y_eval, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_eval, y_proba)) if y_eval.nunique() > 1 else 0.0,
    }

    version = _next_version()
    artifact_path = Path(settings.model_artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    _write_version(version)

    run_id = "local-offline"
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"fraud-retrain-{version}") as run:
            mlflow.log_metrics(metrics)
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("fraud_rate", float(y.mean()))
            mlflow.log_param("model_version", version)
            mlflow.lightgbm.log_model(model, "model")
            run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001 — local demo must work without MLflow server
        logger.warning("MLflow logging skipped: %s", exc)

    return {
        "run_id": run_id,
        "metrics": metrics,
        "artifact_path": str(artifact_path),
        "model_version": version,
    }
