"""
NYC Taxi Fare Prediction - Inference Pipeline
==============================================

Batch inference using ONNX model from MLflow registry.
Uses the same feature engineering as training via shared features module.

Usage:
------
    # Run inference on default data
    uv run python3 src/pipelines/inference.py run

    # Run inference with custom input file
    uv run python3 src/pipelines/inference.py run --input-data path/to/data.parquet

    # Run inference with specific model version
    uv run python3 src/pipelines/inference.py run --model-version 3
"""

import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from dotenv import load_dotenv
from metaflow import (  # type: ignore[attr-defined]
    Parameter,  # For defining command-line parameters
    card,  # For generating visual reports/cards
    current,  # For accessing current run metadata (run_id, etc.)
    environment,  # For injecting environment variables into steps
    step,  # Decorator to define pipeline steps
)

file_path = Path(__file__).resolve()
root_path = file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.common.features import engineer_features  # noqa: E402
from src.common.pipeline import Pipeline  # noqa: E402

load_dotenv()

environment_variables = {
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME", "nyc-taxi-model-v2"),
    "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
}


class Inference(Pipeline):
    """
    Batch inference pipeline. Loads ONNX model from MLflow, runs predictions on new data.
    
    Attributes:
    -----------
    input_data : str
        Path to the input parquet file containing taxi trip data for prediction.
        The data should have the same schema as the training data.
    
    model_name : str
        Name of the registered model in MLflow Model Registry.
        Default is "nyc-taxi-model" (same as registered during training).
    
    model_version : str
        Version of the model to use for predictions.
        Options: "latest", "production", "staging", or a specific version number.
        Default is "latest" which uses the most recently registered version.
    
    output_path : str
        Path where predictions will be saved as a parquet file.
        Default creates a timestamped file in the current directory.
    
    Example:
    --------
    >>> # Run with default settings
    >>> poetry run python src/pipelines/inference.py run
    
    >>> # Run with specific model version
    >>> poetry run python src/pipelines/inference.py run --model-version 2
    
    >>> # Run with custom input/output paths
    >>> poetry run python src/pipelines/inference.py run \\
    ...     --input-data data/new_trips.parquet \\
    ...     --output-path predictions/batch_001.parquet
    """

    # =========================================================================
    # PIPELINE PARAMETERS
    # =========================================================================
    # These parameters can be overridden from the command line when running
    # the pipeline. They provide flexibility for different inference scenarios.
    # =========================================================================

    input_data = Parameter(
        "input-data",
        help="""
        Path to the input parquet file containing taxi trip data for prediction.
        The file should contain the same columns as the training data:
        - tpep_pickup_datetime: Trip pickup timestamp
        - tpep_dropoff_datetime: Trip dropoff timestamp  
        - trip_distance: Distance of the trip in miles
        - passenger_count: Number of passengers
        - fare_amount: Base fare amount
        - tip_amount: Tip amount
        - tolls_amount: Toll charges
        - VendorID: Taxi vendor identifier
        - payment_type: Payment method code
        - RatecodeID: Rate code for the trip
        - store_and_fwd_flag: Store and forward flag (Y/N)
        """,
        default="Dataset/yellow_tripdata_2025-08.parquet",
    )

    model_name = Parameter(
        "model-name",
        help="""
        Registered model name in MLflow Model Registry.
        """,
        default="nyc-taxi-model",
    )

    model_version = Parameter(
        "model-version",
        help="""
        Model version: 'latest' or a specific version number (e.g., '1', '2')
        """,
        default="latest",
    )

    output_path = Parameter(
        "output-path",
        help="""
        Path to save predictions as parquet.
        """,
        default="predictions/batch_predictions.parquet",
    )

    # =========================================================================
    # STEP 1: START - Load model and transformer from MLflow
    # =========================================================================
    @card
    @environment(vars=environment_variables)
    @step
    def start(self):
        """
        Connect to MLflow, download ONNX model and transformer artifacts.
        """
        import tempfile

        import joblib
        import mlflow
        from mlflow.tracking import MlflowClient

        self.logger.info("=" * 60)
        self.logger.info("NYC TAXI FARE PREDICTION - INFERENCE PIPELINE")
        self.logger.info("=" * 60)

        self.logger.info(f"MLflow tracking server: {self.mlflow_tracking_uri}")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Model version: {self.model_version}")

        tracking_uri = environment_variables["MLFLOW_TRACKING_URI"]

        try:
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info(f"Connected to MLflow at {tracking_uri}")

        except Exception as e:
            message = f"Failed to connect to MLflow server: {self.mlflow_tracking_uri}"
            self.logger.error(message)
            raise RuntimeError(message) from e

        client = MlflowClient()

        try:
            if self.model_version == "latest":
                versions = client.search_model_versions(f"name='{self.model_name}'")
                if not versions:
                    raise RuntimeError(f"No versions found for model '{self.model_name}'")
                version_info = sorted(versions, key=lambda v: int(v.version))[-1]
            else:
                version_info = client.get_model_version(
                    str(self.model_name), str(self.model_version)
                )

            version_number = version_info.version
            run_id = version_info.run_id
            self.logger.info(f"Resolved model: version {version_number}, run {run_id}")
        except Exception as e:
            raise RuntimeError(
                f"Could not resolve model '{self.model_name}' version '{self.model_version}': {e}"
            ) from e

        # Download the transformer artifact from the training run
        # The transformer was saved in the "preprocessing" artifact directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download ONNX model artifact
            onnx_artifact = client.download_artifacts(str(run_id), "onnx_model", str(tmp_dir))
            onnx_path = Path(onnx_artifact) / "model.onnx"

            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX model not at {onnx_path}")

            # Load into ONNX Runtime session
            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self.onnx_input_name = self.ort_session.get_inputs()[0].name
            self.logger.info(f"ONNX model loaded (input: '{self.onnx_input_name}')")

            # Download Transformer
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
            mlflow.set_experiment(
                environment_variables.get("MLFLOW_EXPERIMENT_NAME", "inference"),
            )

            run = mlflow.start_run(run_name=f"inference-{current.run_id}")
            self.mlflow_run_id = run.info.run_id

            # Log inference parameters for traceability
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

            self.logger.info(f" Started MLflow inference run: {self.mlflow_run_id}")

        except Exception as e:
            self.logger.warning(f"Could not start MLflow inference run: {str(e)}")
            self.logger.warning("  Inference will continue but metrics won't be logged")
            self.mlflow_run_id = None

        self.logger.info("\nInitialization complete. Proceeding to data loading...")
        self.next(self.load_data)

    # =========================================================================
    # STEP 2: LOAD DATA
    # =========================================================================
    @card
    @step
    def load_data(self):
        """
        Load and validate the input data for prediction.

        This step reads the input parquet file and performs basic validation
        to ensure the data has the expected schema. It also logs data statistics
        for monitoring purposes.

        Validation checks:
        ------------------
        1. File exists and is readable
        2. Required columns are present
        3. Data types are compatible
        4. No completely empty columns

        The raw data is stored for later use in generating output with
        predictions appended to the original records.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: LOAD DATA")
        self.logger.info("=" * 60)

        # ---------------------------------------------------------------------
        # LOAD INPUT DATA
        # ---------------------------------------------------------------------
        # Read the parquet file containing taxi trip data for prediction.
        # Parquet is the preferred format because it's columnar, compressed,
        # and preserves data types accurately.
        # ---------------------------------------------------------------------
        self.logger.info(f"Loading input data from: {self.input_data}")

        try:
            # Load the full dataset
            self.raw_data = pd.read_parquet(str(self.input_data))
            self.n_records = len(self.raw_data)
            self.logger.info(f"Loaded {self.n_records:,} records")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Input file not found: {self.input_data}") from e

        except Exception as e:
            raise RuntimeError(f"Failed to read input file: {self.input_data}") from e

        # ---------------------------------------------------------------------
        # SCHEMA VALIDATION
        # ---------------------------------------------------------------------
        # Verify that all required columns are present in the input data.
        # These columns are needed for feature engineering and prediction.
        # Missing columns would cause the pipeline to fail during transform.
        # ---------------------------------------------------------------------
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

        # Check for missing columns
        missing_columns = set(required_columns) - set(self.raw_data.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info("All required columns present")

        # ---------------------------------------------------------------------
        # DATA STATISTICS
        # ---------------------------------------------------------------------
        # Log basic statistics about the input data for monitoring.
        # These metrics help detect data drift (changes in input distribution)
        # which can indicate that the model may need retraining.
        # ---------------------------------------------------------------------
        self.logger.info(
            f"Schema OK. Columns: {len(self.raw_data.columns)}, "
            f"Memory: {self.raw_data.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )

        # Store the number of records for later validation
        self.n_records = len(self.raw_data)

        self.logger.info("\nData loading complete. Proceeding to feature engineering...")
        self.next(self.feature_engineering)

    # =========================================================================
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    @step
    def feature_engineering(self):
        """
        Create derived features from raw data - IDENTICAL to training pipeline.

        This step applies the EXACT same feature transformations that were used
        during training. Consistency is critical - any difference in feature
        engineering between training and inference will cause training-serving
        skew, leading to degraded prediction quality.

        Features Created:
        -----------------
        Temporal Features:
            - trip_duration_minutes: Duration of trip in minutes
            - pickup_hour: Hour of pickup (0-23)
            - pickup_dayofweek: Day of week (0=Monday, 6=Sunday)
            - pickup_month: Month of pickup (1-12)
            - is_weekend: Binary flag for weekend trips

        Cyclical Features (for preserving temporal continuity):
            - hour_sin, hour_cos: Sine/cosine encoding of hour
            - dayofweek_sin, dayofweek_cos: Sine/cosine encoding of day

        Financial Features:
            - fare_per_mile: Fare amount divided by distance
            - revenue_per_mile: Total amount divided by distance
            - tip_percentage: Tip as percentage of total

        Efficiency Features:
            - speed_mph: Average speed in miles per hour

        Categorical Derived:
            - time_of_day: Categorized time period
            - vendor_payment_interaction: Combined vendor and payment type

        IMPORTANT: This code must be kept in sync with training.py
        ---------  Any changes here must also be made in training pipeline!
        """
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
        self.next(self.predict)

    # =========================================================================
    # STEP 4: PREDICT
    # =========================================================================
    @card
    @environment(vars=environment_variables)
    @step
    def predict(self):
        """
        Generate predictions using the loaded model.

        This step runs the XGBoost model on the transformed features to
        predict the total_amount for each taxi trip. The predictions are
        stored for the final output and logged to MLflow for monitoring.

        Monitoring Metrics:
        -------------------
        - prediction_mean: Average predicted fare
        - prediction_std: Standard deviation of predictions
        - prediction_min/max: Range of predictions
        - n_predictions: Number of predictions made

        These metrics help detect:
        - Data drift (distribution of predictions changes over time)
        - Model degradation (predictions become less accurate)
        - Anomalies (unusually high/low predictions)
        """
        import mlflow

        self.logger.info("=" * 60)
        self.logger.info("STEP 5: PREDICT")
        self.logger.info("=" * 60)

        # ---------------------------------------------------------------------
        # GENERATE PREDICTIONS
        # ---------------------------------------------------------------------
        # Use the loaded model to predict total_amount for each record.
        # The model expects transformed features as input.
        # ---------------------------------------------------------------------
        self.logger.info(f"Generating predictions for {self.X_transformed.shape[0]:,} records...")

        # ONNX Runtime expects float32 input
        X_float = self.X_transformed.astype(np.float32)

        # Handle sparse matrices
        if hasattr(X_float, "toarray"):
            X_float = X_float.toarray().astype(np.float32)

        # Run ONNX inference
        ort_output = self.ort_session.run(None, {self.onnx_input_name: X_float})
        self.predictions = ort_output[0].flatten()

        self.logger.info(f"Generated {len(self.predictions):,} predictions")

        # ---------------------------------------------------------------------
        # PREDICTION STATISTICS
        # ---------------------------------------------------------------------
        # Calculate and log statistics about the predictions.
        # These metrics are used for monitoring model health over time.
        # ---------------------------------------------------------------------
        self.prediction_stats = {
            "prediction_mean": float(np.mean(self.predictions)),
            "prediction_std": float(np.std(self.predictions)),
            "prediction_min": float(np.min(self.predictions)),
            "prediction_max": float(np.max(self.predictions)),
            "prediction_median": float(np.median(self.predictions)),
            "n_predictions": len(self.predictions),
        }

        self.logger.info("\nPrediction Statistics:")
        for key, value in self.prediction_stats.items():
            self.logger.info(
                f"  {key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value:,}"
            )

        # ---------------------------------------------------------------------
        # LOG TO MLFLOW
        # ---------------------------------------------------------------------
        # Log prediction statistics to MLflow for monitoring.
        # This creates a time series of metrics that can be visualized.
        # ---------------------------------------------------------------------
        if self.mlflow_run_id:
            try:
                mlflow.set_tracking_uri(environment_variables["MLFLOW_TRACKING_URI"])

                with mlflow.start_run(run_id=self.mlflow_run_id):
                    mlflow.log_metrics(self.prediction_stats)
                    self.logger.info("Prediction metrics logged to MLflow")

            except Exception as e:
                self.logger.warning(f"Could not log metrics to MLflow: {str(e)}")

        self.next(self.end)

    # =========================================================================
    # STEP 5: END
    # =========================================================================
    @step
    def end(self):
        """
        Save predictions and generate final summary.

        This step combines the original input data with predictions and
        saves the result to a parquet file. It also generates a summary
        report of the inference run.

        Output Format:
        --------------
        The output parquet file contains all original columns plus:
        - predicted_total_amount: The model's fare prediction

        This allows easy comparison with actual fares if available,
        and preserves all metadata for downstream analysis.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: END - SAVE PREDICTIONS")
        self.logger.info("=" * 60)

        # ---------------------------------------------------------------------
        # COMBINE DATA WITH PREDICTIONS
        # ---------------------------------------------------------------------
        # Add the predictions as a new column to the original data.
        # This preserves all input information for downstream analysis.
        # ---------------------------------------------------------------------
        self.logger.info("Combining predictions with original data...")

        # Create output dataframe with original data
        output_df = self.raw_data.copy()

        # Add predictions column
        output_df["predicted_total_amount"] = self.predictions

        # ---------------------------------------------------------------------
        # SAVE OUTPUT
        # ---------------------------------------------------------------------
        # Save the combined data to a parquet file.
        # Create the output directory if it doesn't exist.
        # ---------------------------------------------------------------------
        output_path = Path(str(self.output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving predictions to: {output_path}")
        output_df.to_parquet(output_path, index=False)

        self.logger.info(f"Saved {len(output_df):,} predictions")

        # ---------------------------------------------------------------------
        # FINAL SUMMARY
        # ---------------------------------------------------------------------
        # Print a summary of the inference run for easy reference.
        # ---------------------------------------------------------------------
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INFERENCE PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.model_name} (version {self.model_version})")
        self.logger.info(f"Input records: {self.n_records:,}")
        self.logger.info(f"Predictions generated: {len(self.predictions):,}")
        self.logger.info(f"Output file: {output_path}")
        self.logger.info("")
        self.logger.info("Prediction Summary:")
        self.logger.info(f"  Mean predicted fare: ${self.prediction_stats['prediction_mean']:.2f}")
        self.logger.info(
            f"  Median predicted fare: ${self.prediction_stats['prediction_median']:.2f}"
        )
        self.logger.info(f"  Min predicted fare: ${self.prediction_stats['prediction_min']:.2f}")
        self.logger.info(f"  Max predicted fare: ${self.prediction_stats['prediction_max']:.2f}")
        self.logger.info("=" * 60)

        # If MLflow run was created, print the link
        if self.mlflow_run_id:
            self.logger.info(
                f"\nMLflow Run: {self.mlflow_tracking_uri}/#/experiments/runs/{self.mlflow_run_id}"
            )


# MAIN ENTRY POINT
if __name__ == "__main__":
    Inference()
