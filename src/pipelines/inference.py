"""
Inference Pipeline for the NYC Taxi Fare Prediction.

This module contains the inference pipeline for the NYC Taxi Fare Prediction system.
It loads a trained model from the MLflow model registry and executes batch predictions on new data.

Pipeline Steps:
-----------------
1. Start: Inititalize the pipeline, connect to MLflow, and load the registered model.
2. load_data: Load and validate the input data for prediction.
3. feature_engineering: Apply the same feature engineering steps as the training pipeline.
4. transform: Apply the saved preprocessing transformer
5. predict: Make predictions using the loaded model
6. end: Save predictions and generate summary statistics.

Usage:
------------
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

# Add the parent directory to sys.path to allow imports from common module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

from common.pipeline import Pipeline

load_dotenv()

# Environment variables
environment_variables = {
    "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"),
    "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN"),
    "DATABRICKS_WORKSPACE_ID": os.getenv("DATABRICKS_WORKSPACE_ID"),
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
    "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME"),
}

class Inference(Pipeline):
    """
    Inference pipeline for NYC Taxi Fare Prediction.

    This pipeline loads a trained model from the MLflow Model Registry and generates
    predictions for new taxi trip data. It applies the same feature engineering
    and preprocessing transformations as the training pipeline.

    Attributes:
    -----------------
    input_data: str
        Path to the input data file.
    model_name: str
        Name of the model to load from the MLflow Model Registry.
    model_version: int
        Version of the model to load from the MLflow Model Registry.
        Options: "latest", "production", "staging", or a specific version number.
        Default: "production".
    output_path: str
        Path where predictions will be saved as a parquet file.
        Default: creates a timestamped file in the current directory.

    Usage:
    -------------
        # Run inference on default settings
        poetry run python src/pipelines/inference.py run

        # Run inference with specific model version
        poetry run python src/pipelines/inference.py run --model-version 3

        # Run inference with custom input/output paths
        poetry run python src/pipelines/inference.py run /
        --input-data path/to/data.parquet --output-path path/to/predictions.parquet
    """

    # Pipeline parameters
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
        default="Dataset/yellow_tripdata_2025-09.parquet"
    )

    model_name = Parameter(
        "model-name",
        help="""
        Name of the registered model in MLflow Model Registry.
        This should match the name used when registering the model during training.
        """,
        default="nyc-taxi-model",
    )

    model_version = Parameter(
        "model-version",
        help="""
        Version of the model to load from the registry.
        Options:
        - "latest": Use the most recently registered version
        - "production": Use the version tagged as "production" (default)
        - "staging": Use the version in staging
        - "<number>": Use a specific version number
        """,
        default="production",
    )

    output_path = Parameter(
        "output-path",
        help="""
        Path where the predictions will be saved as a parquet file.
        The output file will contain the original input data plus a new
        'predicted_total_amount' column with the model's predictions.
        """,
        default="predictions/nyc-taxi-predictions.parquet",
    )

    # Step 1: Start - Initialize the pipeline, connect to MLflow, and load the registered model.
    @card
    @environment(vars=environment_variables)
    @step
    def start(self):
        """
        Initialize the inference pipeline and load the model from MLflow registry.

        The model and transformer are stored as instance attributes so they
        can be accessed in subsequent steps. Metaflow automatically serializes
        these artifacts and passes them between steps.

        Raises:
        -----------
        RuntimeError
            If unable to connect to MLflow or load the model/transformer.
        """
        import mlflow
        import joblib
        import tempfile

        # MLFLOW connection
        self.logger.info("="*60)
        self.logger.info("NYC TAXI FARE PREDICTION - INFERENCE PIPELINE")
        self.logger.info("="*60)

        self.logger.info(f"MLflow tracking server: {self.mlflow_tracking_uri}")
        self.logger.info(f"Model name: {self.model_name}")
        self.logger.info(f"Model version: {self.model_version}")

        try:
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

            inference_experiment_name = f"{environment_variables["MLFLOW_EXPERIMENT_NAME"]}-inference"
            mlflow.set_experiment(inference_experiment_name)

            self.logger.info("Succeessfully connected to MLflow tracking server")
            self.logger.info(f"    Inference experiment: {inference_experiment_name}")

        except Exception as e:
            message = f"Failed to connect to MLflow server: {self.mlflow_tracking_uri}"
            self.logger.error(message)
            raise RuntimeError(message) from e

        # Model Loading
        try:
            client = mlflow.tracking.MlflowClient()
            version_info = None  # Initialize to avoid unbound variable
            
            # construct the model URI based on version parameter
            if self.model_version in ["production", "staging"]:
                # Get the version number from the model registry
                stage = self.model_version.capitalize()

                versions_in_stage = client.get_latest_versions(
                    name=self.model_name,
                    stages=[stage]
                )

                if not versions_in_stage:
                    # No model in the requested stage
                    all_versions = client.get_latest_versions(
                        name=self.model_name,
                        stages=["None", "Staging", "Production", "Archived"]
                    )
                
                    if not all_versions:
                        raise RuntimeError(
                            f"No versions found for model '{self.model_name}'. "
                            f"Please run the training pipeline to create a model."
                        )

                    available_versions = [f"{v.version} ({v.current_stage})" for v in all_versions]

                    raise RuntimeError(
                        f"""No model found in '{stage}' stage for '{self.model_name}'.\n
                        Available versions: {available_versions}.\n
                        \n\nTo promote a model to {stage} in Databricks:\n
                            1. Go to MLflow UI > Models > {self.model_name}\n
                            2. Select the version to promote\n
                            3. Click 'Stage' dropdown > 'Transition to {stage}'\n
                            or use the API:
                            client.transition_model_version_stage(name='{self.model_name}', version=<version_number>, stage='{stage}'\n)
                            """
                    )

                version_info = versions_in_stage[0]
                version_number = version_info.version
                model_uri = f"models:/{self.model_name}/{stage}"

                self.logger.info(f"Loading {stage} model (version {version_number})")
                self.logger.info(f"  Run ID: {version_info.run_id}")
                self.logger.info(f"  Description: {version_info.description or 'N/A'}")

            elif self.model_version == "latest":
                # Load from specific stage
                self.logger.info("Loading latest model version")
                all_versions = client.get_latest_versions(
                    name=self.model_name,
                    stages=["None", "Staging", "Production", "Archived"]
                )

                if not all_versions:
                    raise RuntimeError(f"No versions found for model: '{self.model_name}'")

                version_number = max(int(v.version) for v in all_versions)
                version_info = [v for v in all_versions if int(v.version) == version_number][0]
                model_uri = f"models:/{self.model_name}/{version_number}"

                self.logger.info(f"Using latest model version: {version_number}")
                self.logger.info(f"   Current stage: {version_info.current_stage}")
                self.logger.info(f"   Run ID: {version_info.run_id}")

            else:
                # Use specific version number
                version_number = self.model_version
                model_uri = f"models:/{self.model_name}/{version_number}"

                # Get version info for logging
                version_info = client.get_model_version(
                    self.model_name,
                    version_number,
                )
                self.logger.info(f"Loading specific model version: {version_number}")
                self.logger.info(f"   Current stage: {version_info.current_stage}")
                self.logger.info(f"   Run ID: {version_info.run_id}")

            self.logger.info(f"Loading model from: {model_uri}")

            # Load the model using MLflow's sklearn
            self.model = mlflow.sklearn.load_model(model_uri)

            self.logger.info("Model loaded successfully")
            self.logger.info(f"Model type: {type(self.model).__name__}")

        except Exception as e:
            message = f"Failed to load model '{self.model_name}' version '{self.model_version}'"
            self.logger.error(message)
            self.logger.error(f"Error details: {str(e)}")
            raise RuntimeError(message) from e

        # Transformer Loading
        try:
            # Get the run ID from version_info (already retrieved above)
            run_id = version_info.run_id
            
            self.logger.info(f"Downloading transformer from run: {run_id}")

            # Download the transformer artifact from the run
            # The transformer is saved in "preprocessing" artifact directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download the preprocessing artifacts
                artifact_path = client.download_artifacts(
                    run_id,
                    "preprocessing",
                    tmp_dir,
                )

                # Load the transformer using joblib
                transformer_path = Path(artifact_path) / "features.joblib"
                self.features_transformer = joblib.load(transformer_path)

                if not transformer_path.exists():
                    raise FileNotFoundError(f"Transformer artifact not found at expected path: {transformer_path}. Contents of downloaded artifacts: {list(Path(artifact_path).iterdir())}")

                self.features_transformer = joblib.load(transformer_path)

            self.logger.info("Preprocessing transformer loaded successfully")
            self.logger.info(f"Transformer type: {type(self.features_transformer).__name__}")

        except Exception as e:
            message = "Failed to load preprocessing tranformer"
            self.logger.error(message)
            self.logger.error(f"Error details: {str(e)}")
            raise RuntimeError(message) from e

        # Starting Inference run
        try:
            run = mlflow.start_run(run_name=f"inference-{current.run_id}")
            self.mlflow_run_id = run.info.run_id

            # Log inference parameters
            mlflow.log_params({
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_version_number": str(version_number),
                "model_run_id": run_id,
                "input_data": self.input_data,
                "metaflow_run_id": self.mlflow_run_id,
                "pipeline_type": "inference",
            })

            mlflow.end_run()

            self.logger.info(f"Started MLflow inference run: {self.mlflow_run_id}")

        except Exception as e:
            self.logger.warning(f"Could not start MLflow run: {str(e)}")
            self.mlflow_run_id = None
        
        self.logger.info("Initialization complete. Proceeding to data loading...")
        self.next(self.load_data)

    @card
    @step
    def load_data(self):
        """
        Load and validate the input data for prediction.

        This step reads the input parquet file and performs basic validation
        to ensure the data has the expected schema. It also logs data statistics
        for monitoring purposes.
        
        The raw data is stored for later use in generating output with
        predictions appended to the original records
        """
        self.logger.info("="*60)
        self.logger.info("Step 2: Loading Input Data")
        self.logger.info("="*60)

        self.logger.info(f"Loading input data from: {self.input_data}")
        # Load input data
        try:
            self.raw_data = pd.read_parquet(self.input_data)
            self.logger.info(f"Loaded {len(self.raw_data):,} records")

        except FileNotFoundError:
            message = f"Input file not found: {self.input_data}"
            self.logger.error(message)
            raise FileNotFoundError(message)

        except Exception as e:
            message = f"Failed to read input file: {self.input_data}"
            self.logger.error(message)
            raise RuntimeError(message) from e

        # Schema validation
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

        # check for missing columns
        missing_columns = set(required_columns) - set(self.raw_data.columns)

        if missing_columns:
            message = f"Missing required columns: {missing_columns}"
            self.logger.error(message)
            raise ValueError(message)

        self.logger.info("All required columns present")
        
        # Data Statistics
        self.logger.info(f"""\nInput Data Statistics:\n\t
                        Total records: {len(self.raw_data):,}\n\t
                        Columns: {self.raw_data.columns}\n\t
                        Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1e6:.2f} MB""")

        # Log numerical column statistics
        self.logger.info("\nNumerical Column Statistics:")
        for col in ["trip_distance", "fare_amount", "passenger_count"]:
            if col in self.raw_data.columns:
                self.logger.info(
                    f"""    {col}: min={self.raw_data[col].min():.2f},
                    max={self.raw_data[col].max():.2f},
                    mean={self.raw_data[col].mean():.2f}"""
                )

        # Store the number of records for later validation
        self.n_records = len(self.raw_data)

        self.logger.info("\nData loading complete. Proceeding to feature engineering...")
        self.next(self.feature_engineering)

    # Step 3: Feature Engineering
    @step
    def feature_engineering(self):
        """
        Create derived features from raw data.

        This step applies the EXACT same feature transformation that was used during training.

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
        """

        self.logger.info("="*60)
        self.logger.info("Step 3: Feature Engineering")
        self.logger.info("="*60)

        # Make a copy of the raw data to avoid modifying the original df
        df = self.raw_data.copy()

        # Datetime conversions
        self.logger.info("Converting timestamps to datetime...")
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

        # Trip duration
        self.logger.info("Calculating trip duration...")
        df["trip_duration_minutes"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60

        # Temporal Features
        self.logger.info("Creating temporal features...")
        
        # Basic temporal extraction
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
        
        # Binary flags for specific time periods
        df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)

        # Cyclical Encoding
        # Encode hours and day-of-week as cyclical features using sine/cosine.
        # This preserves the circular nature of time, ensuring that time-based features
        self.logger.info("Creating cyclical features...")
        
        # Hour encoding (24-hour cycle)
        df["hour_sin"] = np.sin(df["pickup_hour"] * (2 * np.pi / 24))
        df["hour_cos"] = np.cos(df["pickup_hour"] * (2 * np.pi / 24))
        
        # Day of week encoding (7-day cycle)
        df["dayofweek_sin"] = np.sin(df["pickup_dayofweek"] * (2 * np.pi / 7))
        df["dayofweek_cos"] = np.cos(df["pickup_dayofweek"] * (2 * np.pi / 7))

        # Financial Features
        self.logger.info("Creating financial features...")
        
        # Revenue per mile - how much money generated per unit distance
        df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"].replace(0, np.nan)
        
        # Need total_amount for revenue calculation, but it's our target!
        # For inference, we'll use fare_amount as a proxy
        df["revenue_per_mile"] = df["fare_amount"] / df["trip_distance"].replace(0, np.nan)
        
        # Tip as percentage of fare (for tipped transactions)
        df["tip_percentage"] = df["tip_amount"] / df["fare_amount"].replace(0, np.nan)

        # Efficiency Features
        self.logger.info("Creating efficiency features...")
        
        # Speed in miles per hour
        df["speed_mph"] = df["trip_distance"] / (
            df["trip_duration_minutes"] / 60
        ).replace(0, np.nan)

        # Flag Features
        # Binary flags for special conditions
        # Similar to training pipeline
        self.logger.info("Creating flag features...")
        
        df["is_rush_hour"] = 0
        df["is_night"] = 0
        df["refund_amount"] = 0
        df["has_negative_fare"] = (df["fare_amount"] < 0).astype(int)
        df["is_full_refund"] = 0

        # Categorical Derived Features
        self.logger.info("Creating categorical derived features...")
        
        # Time of day categorization (placeholder - simplified)
        df["time_of_day"] = "unknown"
        
        # Negative fare categorization
        df["negative_fare_category"] = "none"
        
        # Vendor-Payment interaction
        # This captures patterns like "Vendor 1 + Credit Card has different pricing"
        df["vendor_payment_interaction"] = (
            df["VendorID"].astype(str) + "_" + df["payment_type"].astype(str)
        )

        # Missing value handling
        self.logger.info("Handling missing values...")
        df.fillna(0, inplace=True)

        # Data type conversions
        self.logger.info("Converting object columns to string...")
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)

        # Store the engineered features dataframe
        self.engineered_data = df
        
        self.logger.info(f"Feature engineering complete. Created {len(df.columns)} columns")
        self.logger.info(f"Records: {len(df):,}")

        self.next(self.transform)

    # Step 4: Transform
    @step
    def transform(self):
        """
        Apply the saved preprocessing transformer to prepare the data for prediction.
        """
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: TRANSFORM")
        self.logger.info("=" * 60)

        # Prepare Features
        df = self.engineered_data.copy()
        
        # Drop target column if it exists (for inference on labeled data)
        if "total_amount" in df.columns:
            self.logger.info("Removing target column 'total_amount' from features")
            df = df.drop(columns=["total_amount"])

        # Apply Transformation
        self.logger.info("Applying preprocessing transformer...")
        self.logger.info(f"Input shape: {df.shape}")

        try:
            # Apply the transformation
            # Here the call is to transform() and not fit_transform as in traning pipeline.
            self.X_transformed = self.features_transformer.transform(df)
            
            self.logger.info(f"Output shape: {self.X_transformed.shape}")
            self.logger.info("Transformation successful")
            
        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            self.logger.error("This may indicate a schema mismatch between training and inference data")
            raise

        # Validate that we have the same number of records
        if self.X_transformed.shape[0] != self.n_records:
            self.logger.warning(
                f"Record count mismatch: input={self.n_records}, "
                f"transformed={self.X_transformed.shape[0]}"
            )

        self.next(self.predict)

    # Step 5: Predict
    @card
    @environment(vars=environment_variables)
    @step
    def predict(self):
        """
        Generate predictions using the loaded model.

        This step runs the XGBoost model on the transformed features to
        predict the total_amount for each taxi trip. The predictions are
        stored for the final output and logged to mlflow for monitoring.

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

        # Generate predictions
        self.logger.info(f"Generating predictions for {self.X_transformed.shape[0]:,} records...")

        try:
            # Run batch prediction
            self.predictions = self.model.predict(self.X_transformed)

            self.logger.info("Predictions generated successfully")
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

        # Prediction Statistics
        pred_stats = {
            "prediction_mean": float(np.mean(self.predictions)),
            "prediction_std": float(np.std(self.predictions)),
            "prediction_min": float(np.min(self.predictions)),
            "prediction_max": float(np.max(self.predictions)),
            "prediction_median": float(np.median(self.predictions)),
            "n_predictions": len(self.predictions),
        }

        self.logger.info("\nPrediction Statistics:")
        for key, value in pred_stats.items():
            if isinstance(value, float):
                self.logger.info(f"    {key}: {value:.4f}")
            else:
                self.logger.info(f"    {key}: {value:,}")

        # Log to MLflow
        if self.mlflow_run_id:
            try:
                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

                with mlflow.start_run(run_id=self.mlflow_run_id):
                    mlflow.log_metrics(pred_stats)
                    self.logger.info("Prediction metrics logged to MLflow")

            except Exception as e:
                self.logger.error(f"Could not log metrics to MLflow: {str(e)}")

        # store prediction stats for the final summary
        self.prediction_stats = pred_stats

        self.next(self.end)

    # Step 6: End
    @step
    def end(self):
        """
        Save predictions and generate summary statistics.

        Thsi step combines the original input data with predictions and
        saves the result to a parquet file. It also generates a summary report of the inference run.

        Output Format:
        ----------------
        The output parquet file contains all original columns plus:
            - predicted_total_amount: The model's fare prediction
        """
        import os

        self.logger.info("=" * 60)
        self.logger.info("STEP 6: END - SAVE PREDICTIONS")
        self.logger.info("=" * 60)

        # COMBINE Data with Predictions
        self.logger.info("Combining predictions with original data...")

        # create output dataframe with original data
        output_df = self.raw_data.copy()

        # Add predictions column
        output_df["predicted_total_amount"] = self.predictions

        # Save output
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving predictions to: {output_path}")
        output_df.to_parquet(output_path, index=False)

        self.logger.info(f"Saved {len(output_df):,} predictions")

        # Final Summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INFERENCE PIPELINE COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.model_name} (version {self.model_version})")
        self.logger.info(f"Input records: {self.n_records:,}")
        self.logger.info(f"Predictions generated: {len(self.predictions):,}")
        self.logger.info(f"Output file: {output_path}")
        self.logger.info("Prediction Summary:")
        self.logger.info(f"  Mean predicted fare: ${self.prediction_stats['prediction_mean']:.2f}")
        self.logger.info(f"  Median predicted fare: ${self.prediction_stats['prediction_median']:.2f}")
        self.logger.info(f"  Min predicted fare: ${self.prediction_stats['prediction_min']:.2f}")
        self.logger.info(f"  Max predicted fare: ${self.prediction_stats['prediction_max']:.2f}")
        self.logger.info("=" * 60)

        # I mlflow run was created, print the link
        if self.mlflow_run_id:
            self.logger.info(f"View the run in MLflow: {self.mlflow_tracking_uri}/#/experiments/runs/{self.mlflow_run_id}")

# Main Execution
if __name__ == "__main__":
    Inference()