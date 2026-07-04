"""
Local training runner for the NYC Taxi fare model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

file_path = Path(__file__).resolve()
root_path = file_path.parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.common.features import (  # noqa: E402
    TARGET_COLUMN,
    build_model,
    build_transformer,
    clean_training_data,
    convert_to_onnx,
    engineer_features,
)
from src.common.pipeline import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_PROJECT_CONFIG,
    Pipeline,
)

load_dotenv()

DEFAULT_EXPERIMENT_NAME = "nyc-taxi-model-v2"


def get_experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)


class Training(Pipeline):
    """Train, evaluate, and optionally register the NYC Taxi fare model."""

    def __init__(
        self,
        dataset: str = DEFAULT_DATASET,
        mlflow_tracking_uri: str | None = None,
        project_config_path: str = DEFAULT_PROJECT_CONFIG,
        training_epochs: int = 100,
        accuracy_threshold: float = 0.5,
        run_id: str | None = None,
        production: bool = False,
    ) -> None:
        super().__init__(
            project_config_path=project_config_path,
            dataset=dataset,
            mlflow_tracking_uri=mlflow_tracking_uri or DEFAULT_MLFLOW_TRACKING_URI,
            production=production,
            **({"run_id": run_id} if run_id else {}),
        )
        self.training_epochs = training_epochs
        self.accuracy_threshold = accuracy_threshold

    def run(self) -> Training:
        self.start()
        self.feature_engineering()
        self.transform()
        self.prepare_cross_validation()
        fold_metrics = [
            self.cross_validation(fold=fold, fold_id=fold_id)
            for fold_id, fold in enumerate(self.folds)
        ]
        self.evaluate_cross_validation(fold_metrics)
        self.train()
        self.register()
        self.end()
        return self

    def start(self) -> None:
        """Prepare the staged dataset and create the MLflow run."""
        import mlflow

        self.logger.info(f"Running training in {self.mode} mode")
        self.data_path = self.prepare_dataset()

        try:
            mlflow.set_tracking_uri(str(self.mlflow_tracking_uri))
            mlflow.set_experiment(get_experiment_name())
        except Exception as exc:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}"
            raise RuntimeError(message) from exc

        try:
            run = mlflow.start_run(run_name=f"training-{self.run_id}")
            self.mlflow_run_id = run.info.run_id
            self.logger.info(f"Started MLflow run: {self.mlflow_run_id}")
            mlflow.end_run()
        except Exception as exc:
            message = f"Failed to start MLflow run at {self.mlflow_tracking_uri}"
            raise RuntimeError(message) from exc

    def feature_engineering(self) -> None:
        """Create derived features and clean training outliers."""
        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        self.logger.info(f"Loaded {len(df)} rows")

        df = engineer_features(df)
        self.logger.info(f"Engineered {len(df.columns)} columns for {len(df)} rows.")

        df, clean_summary = clean_training_data(df)
        self.logger.info(
            f"Data cleaning: removed {clean_summary['rows_removed']:,} rows "
            f"({clean_summary['pct_removed']:.1f}%) - {clean_summary['rows_after']:,} remaining"
        )

        for reason, count in clean_summary["reasons"].items():
            if count > 0:
                self.logger.info(f"\t{reason}: {count:,}")

        df.to_parquet(self.data_path)

    def transform(self) -> None:
        """Fit the transformation pipeline and persist train/test matrices."""
        import gc

        import joblib

        output_dir = Path("processed_dataset")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.X_train_path = str((output_dir / f"X_train_{self.run_id}.joblib").resolve())
        self.X_train_raw_path = str((output_dir / f"X_train_raw_{self.run_id}.joblib").resolve())
        self.y_train_path = str((output_dir / f"y_train_{self.run_id}.joblib").resolve())
        self.X_test_path = str((output_dir / f"X_test_{self.run_id}.joblib").resolve())
        self.y_test_path = str((output_dir / f"y_test_{self.run_id}.joblib").resolve())

        self.logger.info(f"Loading data from {self.data_path}....")
        df = pd.read_parquet(self.data_path)

        df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.logger.info(
            f"Time-based split: train {len(X_train):,} rows "
            f"(up to {X_train['tpep_pickup_datetime'].max()}), "
            f"test {len(X_test):,} rows"
        )

        del df
        gc.collect()

        self.logger.info("Fitting transformer....")
        self.features_transformer = build_transformer()
        X_train_transformed = self.features_transformer.fit_transform(X_train)
        X_test_transformed = self.features_transformer.transform(X_test)

        self.logger.info("Saving transformed data to disk....")
        joblib.dump(X_train_transformed, self.X_train_path, compress=3)
        joblib.dump(X_train, self.X_train_raw_path, compress=3)
        joblib.dump(y_train, self.y_train_path, compress=3)
        joblib.dump(X_test_transformed, self.X_test_path, compress=3)
        joblib.dump(y_test, self.y_test_path, compress=3)

        del X_train, X_test, y_train, y_test, X_train_transformed, X_test_transformed
        gc.collect()

    def prepare_cross_validation(self) -> None:
        """Prepare indices for cross validation."""
        import joblib
        from sklearn.model_selection import KFold

        self.logger.info("Preparing cross-validation folds....")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        X_train_raw = joblib.load(self.X_train_raw_path)

        self.folds = []
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_raw)):
            self.folds.append((train_idx.tolist(), val_idx.tolist()))
            self.logger.info(
                f"Fold {fold_num}: Train {len(train_idx):,} train, {len(val_idx):,} val"
            )

        del X_train_raw

    def cross_validation(self, fold: tuple[list[int], list[int]], fold_id: int) -> dict[str, float]:
        """Run cross-validation on a single fold with fold-local preprocessing."""
        import joblib
        from sklearn.metrics import mean_squared_error, r2_score

        train_idx, val_idx = fold
        self.logger.info(
            f"Training with Fold {fold_id + 1}: {len(train_idx):,} train, {len(val_idx):,} val"
        )

        X_train_raw = joblib.load(self.X_train_raw_path)
        y_train = joblib.load(self.y_train_path)

        X_fold_train = X_train_raw.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train_raw.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_transformer = build_transformer()
        X_fold_train_transformed = fold_transformer.fit_transform(X_fold_train)
        X_fold_val_transformed = fold_transformer.transform(X_fold_val)

        model = build_model(n_estimators=self.training_epochs)
        model.fit(
            X_fold_train_transformed,
            y_fold_train,
            eval_set=[(X_fold_val_transformed, y_fold_val)],
            verbose=False,
        )

        y_pred = model.predict(X_fold_val_transformed)
        metrics = {
            "mse": float(mean_squared_error(y_fold_val, y_pred)),
            "r2": float(r2_score(y_fold_val, y_pred)),
        }
        self.logger.info(
            f"Fold {fold_id + 1} metrics - MSE: {metrics['mse']:.4f}, "
            f"R2: {metrics['r2']:.4f}."
        )
        return metrics

    def evaluate_cross_validation(self, fold_metrics: list[dict[str, float]]) -> None:
        """Aggregate cross-validation results."""
        mse_scores = [metrics["mse"] for metrics in fold_metrics]
        r2_scores = [metrics["r2"] for metrics in fold_metrics]

        self.avg_cv_mse = float(np.mean(mse_scores))
        self.std_cv_mse = float(np.std(mse_scores))
        self.avg_cv_r2 = float(np.mean(r2_scores))
        self.std_cv_r2 = float(np.std(r2_scores))

        self.logger.info("Cross-validation Results:")
        self.logger.info(f"MSE: {self.avg_cv_mse:.4f} (+/- {self.std_cv_mse:.4f})")
        self.logger.info(f"R2: {self.avg_cv_r2:.4f} (+/- {self.std_cv_r2:.4f})")

    def train(self) -> None:
        """Train the final model."""
        import gc

        import joblib
        import mlflow
        from sklearn.metrics import mean_squared_error, r2_score

        self.logger.info("Training final model on full training dataset....")

        try:
            mlflow.set_tracking_uri(str(self.mlflow_tracking_uri))
            mlflow.set_experiment(get_experiment_name())
        except Exception:
            self.logger.exception("Failed to set up MLflow")
            raise

        self.logger.info(f"Loading transformed data from {self.X_train_path}...")
        X_train = joblib.load(self.X_train_path)
        y_train = joblib.load(self.y_train_path)
        X_test = joblib.load(self.X_test_path)
        y_test = joblib.load(self.y_test_path)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)
            self.model = build_model(n_estimators=self.training_epochs)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = self.model.predict(X_test)
            self.mse = float(mean_squared_error(y_test, y_pred))
            self.r2 = float(r2_score(y_test, y_pred))
            self.logger.info(f"Test MSE: {self.mse:.4f}, Test R2: {self.r2:.4f}")

            mlflow.log_metrics(
                {
                    "test_mse": self.mse,
                    "test_r2": self.r2,
                    "cv_avg_mse": self.avg_cv_mse,
                    "cv_std_mse": self.std_cv_mse,
                    "cv_avg_r2": self.avg_cv_r2,
                    "cv_std_r2": self.std_cv_r2,
                }
            )
            mlflow.log_params(
                {
                    "training_epochs": int(self.training_epochs),
                    "accuracy_threshold": float(self.accuracy_threshold),
                    "mode": self.mode,
                }
            )

        del X_train, y_train, X_test, y_test
        gc.collect()

    def register(self) -> None:
        """Register the model when it meets the configured quality threshold."""
        import tempfile

        import joblib
        import mlflow

        try:
            mlflow.set_tracking_uri(str(self.mlflow_tracking_uri))
            mlflow.set_experiment(get_experiment_name())
        except Exception as exc:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}"
            raise RuntimeError(message) from exc

        if float(self.r2) < float(self.accuracy_threshold):
            self.registered = False
            self.logger.info(
                f"R2 ({self.r2:.4f}) < accuracy threshold ({self.accuracy_threshold}). "
                "Skipping registration...."
            )
            return

        self.registered = True
        self.logger.info(
            f"R2 ({self.r2:.4f}) >= accuracy threshold ({self.accuracy_threshold}). "
            "Registering model...."
        )

        onnx_model = convert_to_onnx(self.model, self.features_transformer)
        self.logger.info("Model converted to ONNX format successfully.")

        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            tempfile.TemporaryDirectory() as directory,
        ):
            transformer_path = (Path(directory) / "features.joblib").as_posix()
            joblib.dump(self.features_transformer, transformer_path)
            mlflow.log_artifact(transformer_path, "preprocessing")

            onnx_path = (Path(directory) / "model.onnx").as_posix()
            with open(onnx_path, "wb") as file:
                file.write(onnx_model.SerializeToString())

            mlflow.log_artifact(onnx_path, "onnx_model")
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="model",
                registered_model_name="nyc-taxi-model",
            )

        self.logger.info("Model registered successfully as 'nyc-taxi-model'")

    def end(self) -> None:
        """Log final training summary."""
        self.logger.info("=" * 60)
        self.logger.info("The training runner finished successfully.")
        self.logger.info("=" * 60)
        self.logger.info(f"Test R2: {self.r2:.4f}")
        self.logger.info(f"Model registered: {self.registered}")
        self.logger.info("=" * 60)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local NYC Taxi model training")
    parser.add_argument("command", nargs="?", choices=["run"], default="run")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--project-config", default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--training-epochs", type=int, default=100)
    parser.add_argument("--accuracy-threshold", type=float, default=0.5)
    parser.add_argument("--production", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> Training:
    args = build_arg_parser().parse_args(argv)
    _ = args.command
    runner = Training(
        dataset=args.dataset,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        project_config_path=args.project_config,
        training_epochs=args.training_epochs,
        accuracy_threshold=args.accuracy_threshold,
        production=args.production,
    )
    return runner.run()


if __name__ == "__main__":
    main()
