"""
This module contains the training pipeline for the NYC Taxi dataset.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from common
# This is necessary because the script is run from the root of the repository
# and the common module is located in src/common
file_path = Path(__file__).resolve()
root_path = file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from metaflow import (
    Parameter,
    card,
    current,
    environment,
    step,
)

from common.pipeline import Pipeline, dataset

load_dotenv()

environment_variables = {
    "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"),
    "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN"),
    "DATABRICKS_WORKSPACE_ID": os.getenv("DATABRICKS_WORKSPACE_ID"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
    "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME"),
}


def build_features_transformer():
    """Build a Scikit-learn transformer to preprocess the feature columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Numerical features from the notebook
    numerical_features = [
        "trip_distance",
        "trip_duration_minutes",
        "passenger_count",
        "fare_amount",
        "tip_amount",
        "tolls_amount",
        "fare_per_mile",
        "tip_percentage",
        "speed_mph",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
        "is_rush_hour",
        "is_night",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "refund_amount",
        "has_negative_fare",
        "is_full_refund",
    ]

    # Categorical features from the notebook
    categorical_features = [
        "VendorID",
        "payment_type",
        "RatecodeID",
        "store_and_fwd_flag",
        "time_of_day",
        "negative_fare_category",
        "vendor_payment_interaction",
    ]

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numerical_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ],
    )


def build_model(learning_rate=0.1, n_estimators=100, max_depth=6):
    """Building the XGBoost model."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        objective="reg:squarederror",
        random_state=42,
    )

    return model


class Training(Pipeline):
    """Training pipeline from the NYC Taxi dataset.

    This pipeline trains, evaluates, and registers a model to predict the total amount of a taxi trip.
    """

    training_epochs = Parameter(
        "training_epochs",
        help="Number of estimators for XGBoost",
        default=100,
    )

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help="Minimum R2 score required to register the model.",
        default=0.5,
    )

    @dataset
    @card
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        self.logger.info(f"MLflow tracking server: {self.mlflow_tracking_uri}")

        self.mode = "production" if current.is_production else "development"
        self.logger.info(f"Running flow in {self.mode} mode")

        try:
            # 1. Databricks setup
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

            # 2. Set the experiment name
            mlflow.set_experiment(environment_variables["MLFLOW_EXPERIMENT_NAME"])
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}"
            raise RuntimeError(message) from e

        try:
            run = mlflow.start_run(run_name=f"metaflow-{current.run_id}")
            self.mlflow_run_id = run.info.run_id
            self.logger.info(f"Started MLFlow run: {self.mlflow_run_id}")
            mlflow.end_run()
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}"
            raise RuntimeError(message) from e

        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Creating derived features."""
        self.logger.info("Creating derived features")

        self.logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        self.logger.info(f"Loaded {len(df)} rows")

        # conver timestamps to datetime objects
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

        # Trip duration
        df["trip_duration_minutes"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60

        # Temporal features
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
        df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)

        # Cyclical features
        df["hour_sin"] = np.sin(df["pickup_hour"] * (2 * np.pi / 24))
        df["hour_cos"] = np.cos(df["pickup_hour"] * (2 * np.pi / 24))
        df["dayofweek_sin"] = np.sin(df["pickup_dayofweek"] * (2 * np.pi / 7))
        df["dayofweek_cos"] = np.cos(df["pickup_dayofweek"] * (2 * np.pi / 7))

        # Financial Features
        df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"].replace(0, np.nan)
        df["revenue_per_mile"] = df["total_amount"] / df["trip_distance"].replace(
            0, np.nan
        )
        df["tip_percentage"] = df["tip_amount"] / df["total_amount"].replace(0, np.nan)

        # Efficiency
        df["speed_mph"] = df["trip_distance"] / (
            df["trip_duration_minutes"] / 60
        ).replace(0, np.nan)

        # Flags
        df["is_rush_hour"] = 0
        df["is_night"] = 0
        df["refund_amount"] = 0
        df["has_negative_fare"] = (df["total_amount"] < 0).astype(int)
        df["is_full_refund"] = 0

        # Categorical derived
        df["time_of_day"] = "unknown"
        df["negative_fare_category"] = "none"
        df["vendor_payment_interaction"] = (
            df["VendorID"].astype(str) + "_" + df["payment_type"].astype(str)
        )

        df.fillna(0, inplace=True)

        self.logger.info(f"Created {len(df)} columns")

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)

        df.to_parquet(self.data_path)
        self.next(self.transform)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the dataset."""
        from sklearn.model_selection import train_test_split
        import joblib
        import os
        import pandas as pd
        import gc

        self.X_train_path = os.path.abspath(
            f"processed_dataset/X_train_{current.run_id}.joblib"
        )
        self.y_train_path = os.path.abspath(
            f"processed_dataset/y_train_{current.run_id}.joblib"
        )
        self.X_test_path = os.path.abspath(
            f"processed_dataset/X_test_{current.run_id}.joblib"
        )
        self.y_test_path = os.path.abspath(
            f"processed_dataset/y_test_{current.run_id}.joblib"
        )

        self.logger.info(f"Loading data from {self.data_path}....")
        df = pd.read_parquet(self.data_path)

        # Define Target column
        target_column = "total_amount"

        # split data
        X = df.drop(columns=[target_column])
        y = df[target_column]

        del df
        gc.collect()

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # We need to ensure all features in the list exist in X
        # For now, just filtering the list tot what's available or add missing columns as 0
        # The build_features_transformer uses specific lists
        # Adding missing columns as 0 for saftey if they were not created in feature engineering
        self.logger.info("Fitting transformer....")
        self.features_transformer = build_features_transformer()

        # We need to filter X to only include columns expected by the transformer
        # Assuming the feature_engineering() step created all the expected columns
        X_train_transformed = self.features_transformer.fit_transform(X_train)
        X_test_transformed = self.features_transformer.transform(X_test)

        # Save
        self.logger.info("Saving transformed data to disk....")
        joblib.dump(X_train_transformed, self.X_train_path)
        joblib.dump(y_train, self.y_train_path)
        joblib.dump(X_test_transformed, self.X_test_path)
        joblib.dump(y_test, self.y_test_path)

        del X_train, X_test, y_train, y_test, X_train_transformed, X_test_transformed
        gc.collect()

        self.next(self.prepare_cross_validation)

    @step
    def prepare_cross_validation(self):
        """Prepare indices for cross validation."""
        from sklearn.model_selection import KFold
        import joblib

        self.logger.info("Preparing cross-validation folds....")

        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Load X_train to get indices
        X_train = joblib.load(self.X_train_path)

        # Index needs to reset of X_train to ensure iloc works correctly with KFold indices
        self.folds = []
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            self.folds.append((train_idx.tolist(), val_idx.tolist()))
            self.logger.info(
                f"Fold {fold_num}: Train {len(train_idx):,} train, {len(val_idx):,} val"
            )

        del X_train

        self.next(self.cross_validation, foreach="folds")

    @step
    def cross_validation(self):
        """Run cross-validation on a single fold."""
        from sklearn.metrics import mean_squared_error, r2_score
        import joblib

        train_idx, val_idx = self.input
        fold_id = self.index

        self.logger.info(
            f"Training with Fold {fold_id + 1}: {len(train_idx):,} train, {len(val_idx):,} val"
        )
        # Load data
        X_train = joblib.load(self.X_train_path)
        y_train = joblib.load(self.y_train_path)

        # Split data for this fold
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Fit a New transformer on this fold's training data
        # transformer = build_features_transformer()
        # X_fold_train_transformed = transformer.fit_transform(X_fold_train)
        # X_fold_val_transformed = transformer.transform(X_fold_val)

        # Train model
        model = build_model(n_estimators=self.training_epochs)
        model.fit(
            X_fold_train,
            y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_fold_val)
        mse = mean_squared_error(y_fold_val, y_pred)
        r2 = r2_score(y_fold_val, y_pred)

        self.cv_metrics = {"mse": mse, "r2": r2}
        self.logger.info(f"Fold metrics - MSE: {mse}, R2: {r2}")

        self.next(self.evaluate_cross_validation)

    @step
    def evaluate_cross_validation(self, inputs):
        """Aggregate cross-validation results."""
        import numpy as np

        self.merge_artifacts(inputs, exclude=["cv_metrics", "folds", "input"])

        mse_scores = [i.cv_metrics["mse"] for i in inputs]
        r2_scores = [i.cv_metrics["r2"] for i in inputs]

        self.avg_cv_mse = np.mean(mse_scores)
        self.std_cv_mse = np.std(mse_scores)
        self.avg_cv_r2 = np.mean(r2_scores)
        self.std_cv_r2 = np.std(r2_scores)

        self.logger.info("Cross-validation Results:")
        self.logger.info(f"MSE: {self.avg_cv_mse:.4f} (+/- {self.std_cv_mse:.4f})")
        self.logger.info(f"R2: {self.avg_cv_r2:.4f} (+/- {self.std_cv_r2:.4f})")
        self.next(self.train)

    @card
    @environment(vars=environment_variables)
    @step
    def train(self):
        """Train the model"""
        import mlflow
        import joblib
        from sklearn.metrics import mean_squared_error, r2_score
        import gc

        self.logger.info("Training Final Model on full training dataset....")

        try:
            # 1. Databricks setup
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

            # 2. Set the experiment name
            mlflow.set_experiment(environment_variables["MLFLOW_EXPERIMENT_NAME"])
        except Exception as e:
            self.logger.error(f"Failed to set up MLflow: {str(e)}")
            raise

        # This assumes 'transform' step saved these files and set these path variables
        self.logger.info(f"Loading transformed data from {self.X_train_path}...")

        X_train = joblib.load(self.X_train_path)
        y_train = joblib.load(self.y_train_path)
        X_test = joblib.load(self.X_test_path)
        y_test = joblib.load(self.y_test_path)

        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)

            # Ensure build_model is imported or defined in your script
            self.model = build_model(n_estimators=self.training_epochs)

            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            # Evaluate
            y_pred = self.model.predict(X_test)
            self.mse = mean_squared_error(y_test, y_pred)
            self.r2 = r2_score(y_test, y_pred)

            self.logger.info(f"Test MSE: {self.mse}, Test R2: {self.r2}")

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
                    "training_epochs": self.training_epochs,
                    "accuracy_threshold": self.accuracy_threshold,
                    "mode": self.mode,
                }
            )

        # --- Cleanup to free memory before next step ---
        del X_train, y_train, X_test, y_test
        gc.collect()

        self.next(self.register)

    @environment(vars=environment_variables)
    @step
    def register(self):
        """Register the model in the model registery"""
        import tempfile
        import mlflow
        import joblib

        try:
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment(environment_variables["MLFLOW_EXPERIMENT_NAME"])
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}"
            raise RuntimeError(message) from e

        if self.r2 >= self.accuracy_threshold:
            self.registered = True
            self.logger.info(
                f"R2 ({self.r2:.4f}) >= accuracy threshold ({self.accuracy_threshold})",
                "Registering model....",
            )

            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                # Save model and artifacts
                model_path = (Path(directory) / "model.joblib").as_posix()
                joblib.dump(self.model, model_path)

                transformer_path = (Path(directory) / "features.joblib").as_posix()
                joblib.dump(self.features_transformer, transformer_path)

                mlflow.log_artifact(transformer_path, "preprocessing")

                # Log model
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    registered_model_name="nyc-taxi-model",
                )

                self.logger.info("Model registered successfully as 'nyc-taxi-model'")

        else:
            self.registered = False
            self.logger.info(
                f"R2 ({self.r2:.4f}) < accuracy threshold ({self.accuracy_threshold})",
                "Model performance below threshold. Skipping registration....",
            )

        self.next(self.end)

    @step
    def end(self):
        """End of the Training Pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("The pipeline finished successfully.")
        self.logger.info("=" * 60)
        self.logger.info(f"Mode: {self.model}")
        self.logger.info(f"Test R2: {self.r2:.4f}")
        self.logger.info(f"Model registered: {self.registered}")
        self.logger.info("=" * 60)


if __name__ == "__main__":
    Training()
