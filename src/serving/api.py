"""
FastAPI Service for NYC Taxi Fare Prediction
=============================================

This module provide a REST API for real-time taxi fare predictions.
The API loads the cached model and transformer, applies feature engineering,
and returns fare predictions.

API Endpoints:
--------------

- GET  /health:      Health check endpoint
- GET  /model/info:  Model metadata and version info
- POST /predict:     Single trip fare prediction

Usage:
-------
    # Start the server
    uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

    # Start server (production)
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --workers 4

    # Test predictions
    curl -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"pickup_datetime": "2024-01-15 08:30:00", ....}'
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from loguru import logger

# Config
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models/cache")
MODEL_PATH = Path(MODEL_CACHE_DIR) / "model.joblib"
TRANSFORMER_PATH = Path(MODEL_CACHE_DIR) / "transformer.joblib"
METADATA_PATH = Path(MODEL_CACHE_DIR) / "metadata.json"

# Global Model Storage
class ModelArtifacts:
    """Container for loaded model artifacts"""
    model = None
    transformer = None
    metadata = None
    is_loaded = False

artifacts = ModelArtifacts()

# Request/Response Schemas
class TripInput(BaseModel):
    """
    Input schema for a single taxi trip prediction.

    This schema defines all the fields a client can provide to get a
    fare prediction. Some fields are required (like pickup/dropoff times),
    while others have sensible defaults.
    
    The API performs feature engineering on these inputs to create the
    features expected by the ML model.
    """
    # Required Fields
    pickup_datetime: str = Field(
        ...,
        description="Trip pickup date and time (format: 'YYYY-MM-DD HH:MM:SS')(required)",
        examples=["2024-01-15 08:30:00"],
    )

    dropoff_datetime: str = Field(
        ...,
        description="Dropoff date and time (format: 'YYYY-MM-DD HH:MM:SS')(required)",
        examples=["2024-01-15 09:15:00"],
    )

    trip_distance: float = Field(
        ...,
        description="Trip distance in miles (required)",
        examples=[5.2],
    )

    # Optional Fields
    passenger_count: int = Field(
        default=1,
        ge=0,
        le=4,
        description="Number of passengers (0-4)",
        examples=[2],
    )

    VendorID: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Vendor ID (1 or 2)",
        examples=[1],
    )

    RatecodeID: int = Field(
        default=1,
        ge=1,
        le=6,
        description="Rate code (1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group)",
        examples=[1],
    )

    store_and_fwd_flag: str = Field(
        default="N",
        description="Store and forward flag ('Y' or 'N')",
        examples=["N"],
    )

    payment_type: int = Field(
        default=1,
        ge=1,
        le=6,
        description="Payment type (1=Credit card, 2=Cash, 3=No Charge, 4=Dispute, 5=Unknown, 6=Voided)",
        examples=[1],
    )

    # Financial Fields
    fare_amount: float = Field(
        default=0.0,
        description="Base Fare amount (if known, otherwise estimated)",
        examples=[14.50],
    )

    tip_amount: float = Field(
        default=0.0,
        ge=0,
        description="Tip amount (if known)",
        examples=[3.00]
    )
    
    tolls_amount: float = Field(
        default=0.0,
        ge=0,
        description="Toll charges (if known)",
        examples=[0.0]
    )

    # Validators
    @field_validator("pickup_datetime", "dropoff_datetime")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        """Validate datetime format"""
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            return v
        except ValueError:
            raise ValueError(
                f"Invalid datetime format: '{v}'. "
                f"Expected format: 'YYYY-MM-DD HH:MM:SS'"
            )
    
    @field_validator("store_and_fwd_flag")
    @classmethod
    def validate_store_fwd(cls, v: str) -> str:
        """Validate store_and_fwd_flag is Y or N"""
        if v.upper() not in ["Y", "N"]:
            raise ValueError("store_and_fwd_flag must be 'Y' or 'N'")

        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "pickup_datetime": "2024-01-15 08:30:00",
                "dropoff_datetime": "2024-01-15 09:15:00",
                "trip_distance": 5.2,
                "passenger_count": 2,
                "VendorID": 1,
                "RatecodeID": 1,
                "store_and_fwd_flag": "N",
                "payment_type": 1,
                "fare_amount": 0.0,
                "tip_amount": 0.0,
                "tolls_amount": 0.0
            }
        }

class PredictionResponse(BaseModel):
    """
    Response schema for fare predictions.

    Contains the predicted fare along with metadata about the prediction.
    """

    predicted_fare: float = Field(
        description="Predicted total fare amount in USD",
    )

    trip_duration_minutes: float = Field(
        description="Calculated trip duration in minutes"
    )

    model_version: str = Field(
        description="Version of the model for prediction"
    )

    prediction_timestamp: str = Field(
        description="Timestamp when prediction was made"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_fare": 25.50,
                "trip_duration_minutes": 45.0,
                "model_version": "1",
                "prediction_timestamp": "2024-01-15T10:30:00"
            }
        }

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: Optional[str]
    model_type: str
    transformer_type: str
    cache_path: str

# Feature Engineering
def engineer_features(trip: TripInput) -> pd.DataFrame:
    """
    Perform feature engineering on the input data.

    Parameters:
    -----------
        trip_input: TripInput
            Raw user input data from the API request
    
    Returns:
    --------
        pd.DataFrame: DataFrame with engineered features
    """
    # Parse DateTime
    pickup_dt = pd.to_datetime(trip.pickup_datetime)
    dropoff_dt = pd.to_datetime(trip.dropoff_datetime)

    # Create Base DateTime
    data = {
        "tpep_pickup_datetime": pickup_dt,
        "tpep_dropoff_datetime": dropoff_dt,
        "trip_distance": [trip.trip_distance],
        "passenger_count": [trip.passenger_count],
        "VendorID": [trip.VendorID],
        "RatecodeID": [trip.RatecodeID],
        "store_and_fwd_flag": [trip.store_and_fwd_flag],
        "payment_type": [trip.payment_type],
        "fare_amount": [trip.fare_amount],
        "tip_amount": [trip.tip_amount],
        "tolls_amount": [trip.tolls_amount],
        # These are typically in the raw data but we'll set defaults
        "extra": [0.0],
        "mta_tax": [0.5],
        "improvement_surcharge": [0.3],
        "congestion_surcharge": [2.5],
        "Airport_fee": [0.0],
        "total_amount": [0.0],  # Target variable
    }

    df = pd.DataFrame(data)

    # Trip Duration
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Temporal Features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)

    # Cyclical Encoding
    df["hour_sin"] = np.sin(df["pickup_hour"] * (2 * np.pi / 24))
    df["hour_cos"] = np.cos(df["pickup_hour"] * (2 * np.pi / 24))
    df["dayofweek_sin"] = np.sin(df["pickup_dayofweek"] * (2 * np.pi / 7))
    df["dayofweek_cos"] = np.cos(df["pickup_dayofweek"] * (2 * np.pi / 7))

    # Financial Features
    if trip.fare_amount == 0 and trip.tip_amount > 0:
        estimated_fare = 2.50 * (trip.trip_distance *2.50)
        df["fare_amount"] = estimated_fare

    df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"].replace(0, np.nan)
    df["revenue_per_mile"] = df["total_amount"] / df["trip_distance"].replace(0, np.nan)
    df["tip_percentage"] = df["tip_amount"] / df["fare_amount"].replace(0, np.nan)

    # Efficiency Features
    df["speed_mph"] = df["trip_distance"] / (
        df["trip_duration_minutes"] / 60
    ).replace(0, np.nan)

    # Flag Features
    df["is_rush_hour"] = 0
    df["is_night"] = 0
    df["refund_amount"] = 0
    df["has_negative_fare"] = (df["fare_amount"] < 0).astype(int)
    df["is_full_refund"] = 0

    # Categorical Derived Features
    df["time_of_day"] = "unknown"
    df["negative_fare_category"] = "none"
    df["vendor_payment_interaction"] = (
        df["VendorID"].astype(str) + "_" + df["payment_type"].astype(str)
    )

    # Fill Missing Values
    df.fillna(0, inplace=True)

    # Convert object columns to string
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    # Drop Target and datetime columns
    df = df.drop(columns=["total_amount", "tpep_pickup_datetime", "tpep_dropoff_datetime"])

    return df

# Application lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Loads model and transformer artifacts at startup, ensuring they're ready
    before accepting requests.
    """

    # Startup
    # Load Model artifacts
    logger.info("Starting NYC Taxi Fare Prediction API...")

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}"
                f"Run 'python or python3 src/serving/download_model.py' first"
            )

        if not TRANSFORMER_PATH.exists():
            raise FileNotFoundError(
                f"Transformer not found at {TRANSFORMER_PATH}"
                f"Run 'python or python3 src/serving/download_transformer.py' first"
            )

        # load model
        logger.info(f"Loading model from {MODEL_PATH}...")
        artifacts.model = joblib.load(MODEL_PATH)
        logger.info(f"    Model Loaded successfully! {type(artifacts.model).__name__}")
        
        # load transformer
        logger.info(f"Loading transformer from {TRANSFORMER_PATH}")
        artifacts.transformer = joblib.load(TRANSFORMER_PATH)
        logger.info(f"    Transformer Loaded successfully! {type(artifacts.transformer).__name__}")

        # load metadata
        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                artifacts.metadata = json.load(f)
            logger.info(f"    Metadata loaded: version {artifacts.metadata["model_version"]}")
        else:
            artifacts.metadata = {
                "model_version": "unknown"
            }

        artifacts.is_loaded = True

        logger.info("")
        logger.info("    API ready to serve perdictions!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down NYC Taxi Fare Prediction API...")
    artifacts.model = None
    artifacts.transformer = None
    artifacts.is_loaded = False


# FastAPI Application
app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="""
    Real-time taxi fare prediction service for NYC yellow taxis.
    
    This API uses an XGBoost model trained on historical NYC taxi trip data
    to predict fare amounts based on trip characteristics.
    
    ## Features
    - Single trip fare predictions
    - Sub-second response times
    - Automatic feature engineering
    
    ## Model Information
    - Algorithm: XGBoost Regressor
    - Target: Total fare amount (USD)
    - Features: Temporal, distance, and categorical features
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the current status of the API, including whether the model
    is loaded and ready for predictions.

    Intended Usage:
    - Kubernetes health checks
    - Load balancer health checks
    - Monitoring and alerting
    """
    return HealthResponse(
        status="On" if artifacts.is_loaded else "Off",
        model_loaded=artifacts.is_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.

    Returns metadata about the model currently being used for predictions,
    including version, type, and cache location.
    """
    if not artifacts.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfoResponse(
        model_name=artifacts.metadata.get("model_name", "unknown"),
        model_version=artifacts.metadata.get("model_version", "unknown"),
        model_alias=artifacts.metadata.get("model_alias"),
        model_type=type(artifacts.model).__name__,
        transformer_type=type(artifacts.transformer).__name__,
        cache_path=str(MODEL_CACHE_DIR),
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(trip: TripInput):
    """
    Predict the total fare for a taxi trip.

    This endpoint accepts trip details and returns a fare prediction.
    The prediction includes:
        - Predicted total fare amount
        - Calculated trip duration
        - Model version used
        - Prediction timestamp

    ## Example Request
    ```json
    {
        "pickup_datetime": "2024-01-15 08:30:00",
        "dropoff_datetime": "2024-01-15 09:15:00",
        "trip_distance": 5.2,
        "passenger_count": 2,
        "VendorID": 1,
        "payment_type": 1
    }
    ```
    
    ## Example Response
    ```json
    {
        "predicted_fare": 25.50,
        "trip_duration_minutes": 45.0,
        "model_version": "1",
        "prediction_timestamp": "2024-01-15T10:30:00"
    }
    ```
    """
    # Validate Model is loaded
    if not artifacts.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )

    try:
        # Feature Engineering
        logger.debug(f"Processing prediction request: {trip.pickup_datetime}")

        features_df = engineer_features(trip)

        # Calculate trip duration
        pickup_dt = pd.to_datetime(trip.pickup_datetime)
        dropoff_dt = pd.to_datetime(trip.dropoff_datetime)
        trip_duration = (dropoff_dt - pickup_dt).total_seconds() / 60

        # Apply Transformer
        X_transformed = artifacts.transformer.transform(features_df)

        # Make Prediction
        prediction = artifacts.model.predict(X_transformed)[0]

        # Ensure prediction is non-negative
        predicted_fare = max(0.0, float(prediction))

        logger.debug(f"Prediction: ${predicted_fare:.2f}")

        # Return Response
        return PredictionResponse(
            predicted_fare=round(predicted_fare, 2),
            trip_duration_minutes=round(trip_duration, 2),
            model_version=artifacts.metadata.get("model_version", "unknown"),
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Main Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
