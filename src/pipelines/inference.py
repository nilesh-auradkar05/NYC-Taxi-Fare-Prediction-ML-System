"""
Local batch inference runner for the NYC Taxi fare model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from dotenv import load_dotenv

file_path = Path(__file__).resolve()
root_path = file_path.parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.common.features import engineer_features  # noqa: E402
from src.common.pipeline import (  # noqa: E402
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_PROJECT_CONFIG,
    Pipeline,
)

load_dotenv()

DEFAULT_INPUT_DATA = "Dataset/yellow_tripdata_2025-08.parquet"
DEFAULT_MODEL_NAME = "nyc-taxi-model"
DEFAULT_MODEL_VERSION = "latest"
DEFAULT_OUTPUT_PATH = "predictions/batch_predictions.parquet"
DEFAULT_EXPERIMENT_NAME = "nyc-taxi-model-v2"


def get_experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)


class Inference(Pipeline):
    """Load a registered ONNX model and write batch predictions."""

    def __init__(
        self,
        input_data: str = DEFAULT_INPUT_DATA,
        model_name: str = DEFAULT_MODEL_NAME,
        model_version: str = DEFAULT_MODEL_VERSION,
        output_path: str = DEFAULT_OUTPUT_PATH,
        mlflow_tracking_uri: str | None = None,
        project_config_path: str = DEFAULT_PROJECT_CONFIG,
        run_id: str | None = None,
        production: bool = False,
    ) -> None:
        super().__init__(
            project_config_path=project_config_path,
            dataset=input_data,
            mlflow_tracking_uri=mlflow_tracking_uri or DEFAULT_MLFLOW_TRACKING_URI,
            production=production,
            **({"run_id": run_id} if run_id else {}),
        )
        self.input_data = input_data
        self.model_name = model_name
        self.model_version = model_version
        self.output_path = output_path
        self.mlflow_run_id: str | None = None

    def run(self) -> Inference:
        self.start()
        self.load_data()
        self.feature_engineering()
        self.predict()
        self.end()
        return self

    def start(self) -> None:
        """Connect to MLflow and download model artifacts."""
        import tempfile

        import joblib
        import mlflow
        from mlflow.tracking import MlflowClient

        self.logger.info("=" * 60)
        self.logger.info("NYC TAXI FARE PREDICTION - BATCH INFERENCE")
        self.logger.info("=" * 60)
        self.logger.info(f"MLflow tracking server: {self.mlflow_tracking_uri}")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Model version: {self.model_version}")

        try:
            mlflow.set_tracking_uri(str(self.mlflow_tracking_uri))
            self.logger.info(f"Connected to MLflow at {self.mlflow_tracking_uri}")
        except Exception as exc:
            message = f"Failed to connect to MLflow server: {self.mlflow_tracking_uri}"
            self.logger.error(message)
            raise RuntimeError(message) from exc

        client = MlflowClient()
        try:
            if self.model_version == "latest":
                versions = client.search_model_versions(f"name='{self.model_name}'")
                if not versions:
                    raise RuntimeError(f"No versions found for model '{self.model_name}'")
                version_info = sorted(versions, key=lambda version: int(version.version))[-1]
            else:
                version_info = client.get_model_version(
                    str(self.model_name), str(self.model_version)
                )

            version_number = version_info.version
            run_id = version_info.run_id
            self.logger.info(f"Resolved model: version {version_number}, run {run_id}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not resolve model '{self.model_name}' version '{self.model_version}': {exc}"
            ) from exc

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_artifact = client.download_artifacts(str(run_id), "onnx_model", str(tmp_dir))
            onnx_path = Path(onnx_artifact) / "model.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model not at {onnx_path}")

            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self.onnx_input_name = self.ort_session.get_inputs()[0].name
            self.logger.info(f"ONNX model loaded (input: '{self.onnx_input_name}')")

            transformer_artifact = client.download_artifacts(
                str(run_id), "preprocessing", str(tmp_dir)
            )
            transformer_path = Path(transformer_artifact) / "features.joblib"
            if not transformer_path.exists():
                raise FileNotFoundError(f"Transformer not at {transformer_path}")

            self.features_transformer = joblib.load(transformer_path)
            self.logger.info(f"Transformer loaded: {type(self.features_transformer).__name__}")

        self._version_number = version_number

        try:
            mlflow.set_experiment(get_experiment_name())
            run = mlflow.start_run(run_name=f"inference-{self.run_id}")
            self.mlflow_run_id = run.info.run_id
            mlflow.log_params(
                {
                    "model_name": self.model_name,
                    "model_version": str(version_number),
                    "model_run_id": str(run_id),
                    "input_data": self.input_data,
                    "pipeline_type": "inference",
                }
            )
            mlflow.end_run()
            self.logger.info(f"Started MLflow inference run: {self.mlflow_run_id}")
        except Exception as exc:
            self.logger.warning(f"Could not start MLflow inference run: {str(exc)}")
            self.logger.warning("Inference will continue but metrics will not be logged")
            self.mlflow_run_id = None

        self.logger.info("Initialization complete. Proceeding to data loading...")

    def load_data(self) -> None:
        """Load and validate the input data for prediction."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: LOAD DATA")
        self.logger.info("=" * 60)
        self.logger.info(f"Loading input data from: {self.input_data}")

        try:
            self.raw_data = pd.read_parquet(str(self.input_data))
            self.n_records = len(self.raw_data)
            self.logger.info(f"Loaded {self.n_records:,} records")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Input file not found: {self.input_data}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to read input file: {self.input_data}") from exc

        required_columns = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_distance",
            "passenger_count",
            "fare_amount",
            "tip_amount",
            "tolls_amount",
            "VendorID",
            "payment_type",
            "RatecodeID",
            "store_and_fwd_flag",
        ]
        missing_columns = set(required_columns) - set(self.raw_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info("All required columns present")
        self.logger.info(
            f"Schema OK. Columns: {len(self.raw_data.columns)}, "
            f"Memory: {self.raw_data.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )
        self.n_records = len(self.raw_data)

    def feature_engineering(self) -> None:
        """Create derived features and transform them for inference."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: FEATURE ENGINEERING")
        self.logger.info("=" * 60)

        self.engineered_data = engineer_features(self.raw_data)
        self.logger.info(f"Engineered {len(self.engineered_data.columns)} features")
        df = self.engineered_data.copy()

        if "total_amount" in df.columns:
            df = df.drop(columns=["total_amount"])

        self.X_transformed = self.features_transformer.transform(df)
        self.logger.info(f"Transformed: {df.shape} -> {self.X_transformed.shape}")

    def predict(self) -> None:
        """Generate predictions and optionally log summary statistics."""
        import mlflow

        self.logger.info("=" * 60)
        self.logger.info("STEP 4: PREDICT")
        self.logger.info("=" * 60)
        self.logger.info(f"Generating predictions for {self.X_transformed.shape[0]:,} records...")

        X_float = self.X_transformed.astype(np.float32)
        if hasattr(X_float, "toarray"):
            X_float = X_float.toarray().astype(np.float32)

        ort_output = self.ort_session.run(None, {self.onnx_input_name: X_float})
        self.predictions = ort_output[0].flatten()
        self.logger.info(f"Generated {len(self.predictions):,} predictions")

        self.prediction_stats = {
            "prediction_mean": float(np.mean(self.predictions)),
            "prediction_std": float(np.std(self.predictions)),
            "prediction_min": float(np.min(self.predictions)),
            "prediction_max": float(np.max(self.predictions)),
            "prediction_median": float(np.median(self.predictions)),
            "n_predictions": len(self.predictions),
        }

        self.logger.info("Prediction Statistics:")
        for key, value in self.prediction_stats.items():
            self.logger.info(
                f"  {key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value:,}"
            )

        if self.mlflow_run_id:
            try:
                mlflow.set_tracking_uri(str(self.mlflow_tracking_uri))
                with mlflow.start_run(run_id=self.mlflow_run_id):
                    mlflow.log_metrics(self.prediction_stats)
                    self.logger.info("Prediction metrics logged to MLflow")
            except Exception as exc:
                self.logger.warning(f"Could not log metrics to MLflow: {str(exc)}")

    def end(self) -> None:
        """Save predictions and log final summary."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: END - SAVE PREDICTIONS")
        self.logger.info("=" * 60)
        self.logger.info("Combining predictions with original data...")

        output_df = self.raw_data.copy()
        output_df["predicted_total_amount"] = self.predictions

        output_path = Path(str(self.output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving predictions to: {output_path}")
        output_df.to_parquet(output_path, index=False)

        self.logger.info(f"Saved {len(output_df):,} predictions")
        self.logger.info("=" * 60)
        self.logger.info("INFERENCE RUNNER COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.model_name} (version {self.model_version})")
        self.logger.info(f"Input records: {self.n_records:,}")
        self.logger.info(f"Predictions generated: {len(self.predictions):,}")
        self.logger.info(f"Output file: {output_path}")
        self.logger.info("Prediction Summary:")
        self.logger.info(f"  Mean predicted fare: ${self.prediction_stats['prediction_mean']:.2f}")
        self.logger.info(
            f"  Median predicted fare: ${self.prediction_stats['prediction_median']:.2f}"
        )
        self.logger.info(f"  Min predicted fare: ${self.prediction_stats['prediction_min']:.2f}")
        self.logger.info(f"  Max predicted fare: ${self.prediction_stats['prediction_max']:.2f}")
        self.logger.info("=" * 60)

        if self.mlflow_run_id:
            self.logger.info(
                f"MLflow Run: {self.mlflow_tracking_uri}/#/experiments/runs/{self.mlflow_run_id}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local NYC Taxi batch inference")
    parser.add_argument("command", nargs="?", choices=["run"], default="run")
    parser.add_argument("--input-data", default=DEFAULT_INPUT_DATA)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--project-config", default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--production", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> Inference:
    args = build_arg_parser().parse_args(argv)
    _ = args.command
    runner = Inference(
        input_data=args.input_data,
        model_name=args.model_name,
        model_version=args.model_version,
        output_path=args.output_path,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        project_config_path=args.project_config,
        production=args.production,
    )
    return runner.run()


if __name__ == "__main__":
    main()
