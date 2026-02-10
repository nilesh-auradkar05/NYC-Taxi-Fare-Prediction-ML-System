"""
NYC Taxi Fare Prediction - Inference Pipeline
==============================================

This module contains the inference pipeline for the NYC Taxi fare prediction system.
It loads a trained model from the MLflow Model Registry and runs batch predictions
on new data.

The inference pipeline follows the same architectural patterns as the training pipeline,
using Metaflow for orchestration and MLflow for model management. This ensures
consistency between training and inference environments, reducing the risk of
training-serving skew.

Pipeline Steps:
---------------
1. start: Initialize the pipeline, connect to MLflow, and load the registered model
2. load_data: Load and validate the input data for prediction
3. feature_engineering: Apply the same feature transformations as training
4. transform: Apply the saved preprocessing transformer (StandardScaler, OneHotEncoder)
5. predict: Run batch predictions using the loaded model
6. end: Save predictions and generate summary statistics

Usage:
------
    # Run inference on default data
    poetry run python src/pipelines/inference.py run
    
    # Run inference with custom input file
    poetry run python src/pipelines/inference.py run --input-data path/to/data.parquet
    
    # Run inference with specific model version
    poetry run python src/pipelines/inference.py run --model-version 3
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from metaflow import (
    Parameter,      # For defining command-line parameters
    card,           # For generating visual reports/cards
    current,        # For accessing current run metadata (run_id, etc.)
    environment,    # For injecting environment variables into steps
    step,           # Decorator to define pipeline steps
)
import onnxruntime as ort

file_path = Path(__file__).resolve()
root_path = file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.common.pipeline import Pipeline
from src.common.features import engineer_features

load_dotenv()

environment_variables = {
    # Databricks workspace URL (if using Databricks-hosted MLflow)
    "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"),
    # Personal Access Token for Databricks authentication
    "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN"),
    # Databricks workspace ID for MLflow tracking
    "DATABRICKS_WORKSPACE_ID": os.getenv("DATABRICKS_WORKSPACE_ID"),
    # MLflow tracking server URI (local: http://127.0.0.1:5000)
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
    # Name of the MLflow experiment for organizing runs
    "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME"),
    "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
}


class Inference(Pipeline):
    """
    Inference Pipeline for NYC Taxi Fare Prediction.
    
    This pipeline loads a trained model from the MLflow Model Registry and
    generates predictions for new taxi trip data. It applies the same feature
    engineering and preprocessing transformations that were used during training
    to ensure consistency between training and inference.
    
    The pipeline is designed for batch inference, processing multiple records
    at once. For real-time inference, consider deploying the model as a REST
    API using MLflow's model serving capabilities or a framework like FastAPI.
    
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
        Full Unity Catalog model path in the format: <catalog>.<schema>.<model_name>
        OR local model path in the format: <path>/<model_name>
        Unity Catalog uses a three-level namespace for model governance:
        - Catalog: Top-level container (e.g., 'ml_models')
        - Schema: Logical grouping within catalog (e.g., 'nyc-taxi')
        - Model: The registered model name (e.g., 'nyc-taxi-model')
        
        This should match the name used when registering the model during training.
        """,
        default="ml_models.nyc-taxi.nyc-taxi-model",
    )

    model_version = Parameter(
        "model-version",
        help="""
        Version or alias of the model to load from Unity Catalog.
        Unity Catalog Options:
        - 'champion': Use the model with @champion alias (default, recommended)
        - 'challenger': Use the model with @challenger alias (for A/B testing)
        - '<number>': Use a specific version number (e.g., '1', '2', '3')
        
        OR Model version number (e.g., '1', '2', 'latest') on local file system.

        NOTE: Unity Catalog uses ALIASES instead of the legacy stages 
        (None/Staging/Production/Archived). Common alias conventions:
        - @champion: Production-ready, validated model
        - @challenger: Model being tested against champion
        
        BEST PRACTICE: Production inference should always use 'champion' alias.
        Models should be promoted to 'champion' only after:
        - Performance validation on holdout data
        - Explainability analysis (SHAP, LIME)
        - Bias and fairness checks
        - Stakeholder approval
        """,
        default="latest",
    )

    output_path = Parameter(
        "output-path",
        help="""
        Path where the predictions will be saved as a parquet file.
        The output file will contain the original input data plus a new
        'predicted_total_amount' column with the model's predictions.
        """,
        default="predictions/batch_predictions.parquet",
    )

    # =========================================================================
    # STEP 1: START
    # =========================================================================
    @card
    @environment(vars=environment_variables)
    @step
    def start(self):
        """
        Initialize the inference pipeline and load the model from MLflow.
        
        This step performs the following operations:
        1. Connect to the Databricks-hosted MLflow tracking server
        2. Load the registered model from the Databricks Model Registry
        3. Load the preprocessing transformer (fitted during training)
        4. Validate that both artifacts are loaded correctly
        
        The model and transformer are stored as instance attributes so they
        can be accessed in subsequent steps. Metaflow automatically serializes
        these artifacts and passes them between steps.
        
        Databricks MLflow Authentication:
        ---------------------------------
        Authentication to Databricks MLflow is handled via environment variables:
        - DATABRICKS_HOST: The Databricks workspace URL
        - DATABRICKS_TOKEN: Personal Access Token for authentication
        - MLFLOW_TRACKING_URI: Set to 'databricks' or the workspace URI
        
        These are injected via the @environment decorator from environment_variables.
        
        Raises:
        -------
        RuntimeError
            If unable to connect to MLflow or load the model/transformer.
        
        Notes:
        ------
        - The model is loaded using MLflow's model URI format:
          models:/<model_name>/<version> or models:/<model_name>/<stage>
        - The transformer is loaded from the model's artifacts directory
        - Both artifacts were logged together during the training pipeline's
          register step to ensure they stay in sync
        """
        import mlflow
        from mlflow.tracking import MlflowClient
        import joblib
        import tempfile

        # ---------------------------------------------------------------------
        # MLFLOW CONNECTION (DATABRICKS)
        # ---------------------------------------------------------------------
        # Connect to the Databricks-hosted MLflow tracking server.
        # Databricks provides a managed MLflow service that integrates with:
        # - Unity Catalog for model governance
        # - Workspace Model Registry for model versioning
        # - Access control via workspace permissions
        #
        # The MLFLOW_TRACKING_URI should be set to 'databricks' or the
        # full workspace URL (e.g., https://<workspace>.cloud.databricks.com)
        # ---------------------------------------------------------------------
        self.logger.info("=" * 60)
        self.logger.info("NYC TAXI FARE PREDICTION - INFERENCE PIPELINE")
        self.logger.info("=" * 60)

        self.logger.info(f"MLflow tracking server: {self.mlflow_tracking_uri}")
        self.logger.info(f"Databricks host: {os.environ.get('DATABRICKS_HOST', 'Not set')}")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Model version/stage: {self.model_version}")

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

        try:
            # Set the MLflow tracking URI from environment variable
            # This can be local (127.0.0.1:5000) for experiment tracking
            mlflow.set_tracking_uri(tracking_uri)
            self.logger.info(f"Connected to MLflow at {tracking_uri}")

        except Exception as e:
            message = f"Failed to connect to MLflow server: {self.mlflow_tracking_uri}"
            self.logger.error(message)
            self.logger.error(f"  Error: {str(e)}")
            self.logger.error("  Check that DATABRICKS_HOST and DATABRICKS_TOKEN are set correctly")
            raise RuntimeError(message) from e

        client = MlflowClient()

        try:
            if self.model_version == "latest":
                versions = client.search_model_versions(f"name='{self.model_name}'")
                if not versions:
                    raise RuntimeError(f"No versions found for model '{self.model_name}'")
                version_info = sorted(versions, key=lambda v: int(v.version))[-1]
            else:
                version_info = client.get_model_version(str(self.model_name), str(self.model_version))

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
            transformer_artifact = client.download_artifacts(str(run_id), "preprocessing", str(tmp_dir))
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
            mlflow.log_params({
                "model_name": self.model_name,
                "model_version": str(version_number),
                "model_run_id": str(run_id),
                "input_data": self.input_data,
                "pipeline_type": "inference",
            })
            
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
            self.logger.info(f"Loaded {len(self.raw_data):,} records")
            
        except FileNotFoundError:
            message = f"Input file not found: {self.input_data}"
            self.logger.error(message)
            raise FileNotFoundError(message)
            
        except Exception as e:
            message = f"Failed to read input file: {self.input_data}"
            self.logger.error(message)
            raise RuntimeError(message) from e

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
            message = f"Missing required columns: {missing_columns}"
            self.logger.error(message)
            raise ValueError(message)

        self.logger.info("All required columns present")

        # ---------------------------------------------------------------------
        # DATA STATISTICS
        # ---------------------------------------------------------------------
        # Log basic statistics about the input data for monitoring.
        # These metrics help detect data drift (changes in input distribution)
        # which can indicate that the model may need retraining.
        # ---------------------------------------------------------------------
        self.logger.info("\nInput Data Statistics:")
        self.logger.info(f"  Total records: {len(self.raw_data):,}")
        self.logger.info(f"  Columns: {len(self.raw_data.columns)}")
        self.logger.info(f"  Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        # Log numerical column statistics
        self.logger.info("\nNumerical column ranges:")
        for col in ["trip_distance", "fare_amount", "passenger_count"]:
            if col in self.raw_data.columns:
                self.logger.info(
                    f"  {col}: min={self.raw_data[col].min():.2f}, "
                    f"max={self.raw_data[col].max():.2f}, "
                    f"mean={self.raw_data[col].mean():.2f}"
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
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value:,}")

        # ---------------------------------------------------------------------
        # LOG TO MLFLOW
        # ---------------------------------------------------------------------
        # Log prediction statistics to MLflow for monitoring.
        # This creates a time series of metrics that can be visualized.
        # ---------------------------------------------------------------------
        if self.mlflow_run_id:
            try:
                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
                
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
        self.logger.info(f"  Median predicted fare: ${self.prediction_stats['prediction_median']:.2f}")
        self.logger.info(f"  Min predicted fare: ${self.prediction_stats['prediction_min']:.2f}")
        self.logger.info(f"  Max predicted fare: ${self.prediction_stats['prediction_max']:.2f}")
        self.logger.info("=" * 60)

        # If MLflow run was created, print the link
        if self.mlflow_run_id:
            self.logger.info(f"\nMLflow Run: {self.mlflow_tracking_uri}/#/experiments/runs/{self.mlflow_run_id}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
# When this script is run directly, instantiate and execute the pipeline.
# The Pipeline base class handles CLI argument parsing and execution.
# =============================================================================
if __name__ == "__main__":
    Inference()