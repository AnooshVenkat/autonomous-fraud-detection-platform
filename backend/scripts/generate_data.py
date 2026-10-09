"""Synthetic credit-card-style fraud data generator (PCA feature space)."""

from pathlib import Path

import numpy as np
import pandas as pd

from app.config import settings
from app.features import FEATURE_COLUMNS

# ~8/30 columns ≈ 27–35% share of drifted columns in Evidently
DRIFT_COLUMNS = ("amount", "v1", "v2", "v3", "v11", "v12", "v14", "v17")


def _generate_block(
    n_samples: int,
    fraud_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    rows = []
    for is_fraud in [0] * n_legit + [1] * n_fraud:
        row = {
            "time": float(rng.exponential(50000)),
            "amount": float(rng.lognormal(mean=3.0, sigma=1.2)),
            "class": is_fraud,
        }
        for i in range(1, 29):
            key = f"v{i}"
            if is_fraud:
                row[key] = float(rng.normal(0.5, 1.8))
            else:
                row[key] = float(rng.normal(0.0, 1.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)[FEATURE_COLUMNS + ["class"]]


def generate_reference_dataset(
    n_samples: int = 5000,
    fraud_rate: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    return _generate_block(n_samples, fraud_rate, seed)


def _load_or_create_reference() -> pd.DataFrame:
    ref_path = Path(settings.reference_data_path)
    if ref_path.exists() and ref_path.stat().st_size > 0:
        ref = pd.read_csv(ref_path)
        if set(FEATURE_COLUMNS).issubset(ref.columns):
            return ref
    ref = generate_reference_dataset(n_samples=4000, fraud_rate=0.02, seed=42)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(ref_path, index=False)
    return ref


def generate_drifted_batch(
    n_samples: int = 50,
    fraud_rate: float = 0.02,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Realistic drift: resample from the reference distribution, then shift only
    a small subset of columns. Unshifted columns stay distribution-matched, so
    Evidently reports ~30% column drift instead of ~100%.
    """
    seed = seed if seed is not None else int(np.random.randint(0, 10_000))
    rng = np.random.default_rng(seed)

    reference = _load_or_create_reference()
    df = reference.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)
    df = df.copy()

    # Targeted shifts on a minority of features
    df["amount"] = df["amount"] * 1.7 + 20.0
    for col in DRIFT_COLUMNS:
        if col == "amount":
            continue
        df[col] = df[col] + float(rng.normal(1.1, 0.05))

    # Keep fraud rate roughly similar; optional mild uptick
    if "class" in df.columns and fraud_rate != 0.02:
        # leave labels as sampled from reference for supervised retrain
        pass

    return df[FEATURE_COLUMNS + (["class"] if "class" in df.columns else [])]


def generate_normal_batch(n_samples: int = 10, seed: int | None = None) -> pd.DataFrame:
    seed = seed if seed is not None else int(np.random.randint(0, 10_000))
    reference = _load_or_create_reference()
    if len(reference) >= n_samples:
        return (
            reference.sample(n=n_samples, replace=True, random_state=seed)
            .reset_index(drop=True)[FEATURE_COLUMNS + (["class"] if "class" in reference.columns else [])]
        )
    return _generate_block(n_samples, fraud_rate=0.02, seed=seed)


def row_to_transaction(row: pd.Series) -> dict:
    return {col: float(row[col]) for col in FEATURE_COLUMNS}
