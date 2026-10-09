"""CLI entry point for training and logging a fraud model to MLflow."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.services.predictor import train_and_log_model
from scripts.generate_data import generate_drifted_batch, generate_reference_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to training CSV (defaults to generated reference + drift data)",
    )
    parser.add_argument("--output", default=settings.reference_data_path)
    args = parser.parse_args()

    if args.data:
        data_path = args.data
    else:
        import pandas as pd

        ref = generate_reference_dataset(n_samples=6000, fraud_rate=0.02, seed=42)
        drift = generate_drifted_batch(n_samples=2000, fraud_rate=0.06, seed=99)
        combined = pd.concat([ref, drift], ignore_index=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.output, index=False)
        data_path = args.output

    result = train_and_log_model(data_path)
    print(f"Training complete. Run ID: {result['run_id']}")
    print(f"Metrics: {result['metrics']}")
    print(f"Model saved to: {result['artifact_path']}")


if __name__ == "__main__":
    main()
