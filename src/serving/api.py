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
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import onnxruntime as ort

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from loguru import logger

file_path = Path(__file__).resolve()
root_path = file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.common.features import engineer_features

# Config
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models/cache")

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

    store_and_fwd_flag: Literal["Y", "N"] = Field(default="N")

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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pickup_datetime": "2025-01-01T08:30:00",
                "dropoff_datetime": "2024-01-15T09:15:00",
                "trip_distance": 5.2,
                "passenger_count": 2,
            }
        }
    )

class PredictionResponse(BaseModel):
    """
    Response schema for fare predictions.

    Contains the predicted fare along with metadata about the prediction.
    """

    predicted_fare: float
    trip_duration_minutes: float
    model_version: str
    prediction_timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: Optional[str] = None
    model_type: str
    transformer_type: str

def trip_to_dataframe(trip: TripInput) -> pd.DataFrame:
    """Convert API request into a single-row DataFrame matching training schema."""
    return pd.DataFrame([{
        "tpep_pickup_datetime": pd.to_datetime(trip.pickup_datetime),
        "tpep_dropoff_datetime": pd.to_datetime(trip.dropoff_datetime),
        "trip_distance": trip.trip_distance,
        "passenger_count": trip.passenger_count,
        "VendorID": trip.VendorID,
        "RatecodeID": trip.RatecodeID,
        "store_and_fwd_flag": trip.store_and_fwd_flag,
        "payment_type": trip.payment_type,
        "fare_amount": trip.fare_amount,
        "tip_amount": trip.tip_amount,
        "tolls_amount": trip.tolls_amount,
        # Columns present in training data but not user-provided
        "extra": 0.0,
        "mta_tax": 0.5,
        "improvement_surcharge": 0.3,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
        "total_amount": 0.0,
    }])

# Application lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Loads ONNX model and transformer artifacts at startup, ensuring they're ready
    before accepting requests.
    """

    # Startup
    # Load Model artifacts
    logger.info("Starting NYC Taxi Fare Prediction API...")

    cache = Path(MODEL_CACHE_DIR)

    MODEL_PATH = cache / "model.onnx"
    TRANSFORMER_PATH = cache / "transformer.joblib"
    METADATA_PATH = cache / "metadata.json"

    if not MODEL_PATH.exists() or not TRANSFORMER_PATH.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {MODEL_CACHE_DIR}. "
            "Run 'python or python3 src/serving/download_model.py' first."
        )

    # load model
    logger.info(f"Loading ONNX model from {MODEL_PATH}...")
    app.state.model = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    app.state.onnx_input_name = app.state.model.get_inputs()[0].name
    logger.info(f" Model loaded: (input: '{app.state.onnx_input_name}')")
    
    # load transformer
    logger.info(f"Loading transformer from {TRANSFORMER_PATH}")
    app.state.transformer = joblib.load(TRANSFORMER_PATH)
    logger.info(f" Transformer loaded: {type(app.state.transformer).__name__}")

    app.state.metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}

    logger.info("")
    logger.info("    API ready to serve perdictions!")
    logger.info("="*60)

    yield

    logger.info("Shutting down...")


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
    version="0.2.0",
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
        status="On" if hasattr(app.state, "model") else "Off",
        model_loaded=hasattr(app.state, "model"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model.

    Returns metadata about the model currently being used for predictions,
    including version, type, and cache location.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfoResponse(
        model_name=app.state.metadata.get("model_name", "unknown"),
        model_version=app.state.metadata.get("model_version", "unknown"),
        model_alias=app.state.metadata.get("model_alias"),
        model_type="ONNX (XGBoost)",
        transformer_type=type(app.state.transformer).__name__,
    )

@app.post("/predict", response_model=PredictionResponse)
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
    import numpy as np
    # Validate Model is loaded
    if not hasattr(app.state, "model"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Feature Engineering
        raw_df = trip_to_dataframe(trip)
        df = engineer_features(raw_df)
        df = df.drop(columns=["total_amount"], errors="ignore")
        logger.debug(f"Processing prediction request: {trip.pickup_datetime}")

        # Transform and predict
        X = app.state.transformer.transform(df)
        
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = X.astype(np.float32)
        
        output = app.state.model.run(None, {app.state.onnx_input_name: X})
        prediction = max(0.0, float(output[0].flatten()[0]))

        # Calculate trip duration
        pickup_dt = pd.to_datetime(trip.pickup_datetime)
        dropoff_dt = pd.to_datetime(trip.dropoff_datetime)
        trip_duration = (dropoff_dt - pickup_dt).total_seconds() / 60

        logger.debug(f"Prediction: ${prediction:.2f}")

        # Return Response
        return PredictionResponse(
            predicted_fare=round(prediction, 2),
            trip_duration_minutes=round(trip_duration, 2),
            model_version=app.state.metadata.get("model_version", "unknown"),
            prediction_timestamp=datetime.now(timezone.utc),
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
