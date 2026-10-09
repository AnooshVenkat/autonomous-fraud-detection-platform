import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from app.config import settings
from app.features import FEATURE_COLUMNS


class DriftMonitor:
    def __init__(self) -> None:
        self.drift_detected = False
        self.drift_score = 0.0
        self.last_check: str | None = None
        self.retrain_triggered = False
        self._checks_since_append = 0
        self._appends_since_check = 0
        self._ensure_paths()

    def _ensure_paths(self) -> None:
        Path(settings.reference_data_path).parent.mkdir(parents=True, exist_ok=True)
        buffer_path = Path(settings.drift_buffer_path)
        if not buffer_path.exists():
            buffer_path.write_text("", encoding="utf-8")

    def _load_reference(self) -> pd.DataFrame:
        ref_path = Path(settings.reference_data_path)
        if not ref_path.exists():
            from scripts.generate_data import generate_reference_dataset

            df = generate_reference_dataset(n_samples=3000, fraud_rate=0.02, seed=42)
            df.to_csv(ref_path, index=False)
            return df
        return pd.read_csv(ref_path)

    def _load_buffer(self) -> pd.DataFrame:
        buffer_path = Path(settings.drift_buffer_path)
        if not buffer_path.exists() or buffer_path.stat().st_size == 0:
            return pd.DataFrame(columns=FEATURE_COLUMNS + ["class"])
        return pd.read_csv(buffer_path)

    def append_transaction(self, features: dict, label: int | None = None) -> None:
        row = {col: features[col] for col in FEATURE_COLUMNS}
        if label is not None:
            row["class"] = label

        buffer = self._load_buffer()
        buffer = pd.concat([buffer, pd.DataFrame([row])], ignore_index=True)

        # Keep at most one window of samples — UI shows buffer / window_size
        if len(buffer) > settings.drift_window_size:
            buffer = buffer.tail(settings.drift_window_size)

        buffer.to_csv(settings.drift_buffer_path, index=False)
        self._appends_since_check += 1

    def should_check(self) -> bool:
        """Avoid running Evidently on every prediction once the window is full."""
        return self._appends_since_check >= max(10, settings.drift_window_size // 4)

    def check_drift(self) -> dict:
        reference = self._load_reference()
        current = self._load_buffer()

        if len(current) < settings.drift_window_size:
            return self._status_payload()

        self._appends_since_check = 0
        current_window = current.tail(settings.drift_window_size)

        # PSI with a moderate threshold avoids false positives on small windows
        # while still flagging the intentionally shifted columns (~30–40%).
        report = Report(
            metrics=[DataDriftPreset(stattest="psi", stattest_threshold=0.2)]
        )
        report.run(
            reference_data=reference[FEATURE_COLUMNS],
            current_data=current_window[FEATURE_COLUMNS],
        )

        result = json.loads(report.json())
        drift_score = self._extract_drift_share(result)
        self.drift_score = drift_score
        self.drift_detected = drift_score >= settings.drift_threshold
        self.last_check = datetime.now(timezone.utc).isoformat()

        return self._status_payload()

    def _extract_drift_share(self, report_json: dict) -> float:
        for metric in report_json.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                result = metric.get("result", {})
                return float(result.get("share_of_drifted_columns", 0.0))
            if "drift_share" in str(metric).lower():
                result = metric.get("result", {})
                if "share_of_drifted_columns" in result:
                    return float(result["share_of_drifted_columns"])
        return 0.0

    def mark_retrain_triggered(self) -> None:
        self.retrain_triggered = True

    def _labeled_reference(self) -> pd.DataFrame:
        from scripts.generate_data import generate_reference_dataset

        ref_path = Path(settings.reference_data_path)
        if ref_path.exists():
            ref = pd.read_csv(ref_path)
            if "class" in ref.columns and not ref["class"].isna().any():
                return ref[FEATURE_COLUMNS + ["class"]]

        ref = generate_reference_dataset(n_samples=3000, fraud_rate=0.02, seed=42)
        ref.to_csv(ref_path, index=False)
        return ref

    def training_frame(self) -> pd.DataFrame:
        """Build a clean labeled training set (reference + drifted distribution)."""
        from scripts.generate_data import generate_drifted_batch

        frames = [self._labeled_reference()]
        buffer = self._load_buffer()
        n_drift = max(len(buffer), settings.drift_window_size)
        # Live buffer is unlabeled inference traffic — synthesize matching drifted labels
        frames.append(generate_drifted_batch(n_samples=n_drift, fraud_rate=0.06, seed=7))
        return pd.concat(frames, ignore_index=True)

    def reset_after_retrain(self) -> None:
        from scripts.generate_data import generate_drifted_batch, generate_reference_dataset

        # Promote the new (drift-aware) distribution as labeled reference
        ref = generate_reference_dataset(n_samples=2500, fraud_rate=0.02, seed=42)
        drift = generate_drifted_batch(n_samples=1500, fraud_rate=0.06, seed=99)
        pd.concat([ref, drift], ignore_index=True).to_csv(
            settings.reference_data_path, index=False
        )
        Path(settings.drift_buffer_path).write_text("", encoding="utf-8")
        self.drift_detected = False
        self.drift_score = 0.0
        self.retrain_triggered = False
        self.last_check = datetime.now(timezone.utc).isoformat()

    def status(self) -> dict:
        return self._status_payload()

    def _status_payload(self) -> dict:
        buffer = self._load_buffer()
        return {
            "drift_detected": self.drift_detected,
            "drift_score": round(self.drift_score, 4),
            "threshold": settings.drift_threshold,
            "buffer_size": len(buffer),
            "window_size": settings.drift_window_size,
            "last_check": self.last_check,
            "retrain_triggered": self.retrain_triggered,
        }

    @property
    def buffer_size(self) -> int:
        return len(self._load_buffer())
