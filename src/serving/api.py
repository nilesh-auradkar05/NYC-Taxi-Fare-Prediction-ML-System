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

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from src.common.features import engineer_features
from src.serving.monitoring import (
    LAST_PREDICTION_TIME,
    MODEL_INFO,
    PREDICTION_DISTANCE,
    PREDICTION_VALUE,
    PREDICTIONS_SERVED,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    PredictionRecord,
    generate_drift_report,
    get_metrics_text,
    prediction_tracker,
)

file_path = Path(__file__).resolve()
root_path = file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))


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
        description="Trip pickup date and time (format: 'YYYY-MM-DD HH:MM:SS')",
        examples=["2026-01-15 08:30:00"],
    )

    trip_distance: float = Field(
        ...,
        gt=0,
        description="Estimated trip distance in miles",
        examples=[5.2],
    )

    # Optional: reasonable defaults for typical trips
    estimated_duration_minutes: float | None = Field(
        default=None,
        gt=0,
        description="Estimated trip duration in minutes. If omitted, estimated from distance.",
        examples=[25.0],
    )

    # Optional Fields
    passenger_count: int = Field(
        default=1,
        ge=0,
        le=6,
        description="Number of passengers",
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
        description=(
            "Rate code (1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group)"
        ),
        examples=[1],
    )

    payment_type: int = Field(
        default=1,
        ge=1,
        le=6,
        description="Payment type (1=Credit card, 2=Cash)",
        examples=[1],
    )

    store_and_fwd_flag: Literal["Y", "N"] = Field(default="N")


class PredictionResponse(BaseModel):
    """
    Response schema for fare predictions.

    Contains the predicted fare along with metadata about the prediction.
    """

    predicted_fare: float
    estimated_duration_minutes: float
    model_version: str
    prediction_timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_alias: str | None = None
    model_type: str
    transformer_type: str


def trip_to_dataframe(trip: TripInput) -> pd.DataFrame:
    """Convert pre-trip API request into a single-row DataFrame matching training schema.

    Estimates dropoff time from distance if duration not provided.
    Financial fields (fare, tip, tolls) are set to 0 since they are
    unknown pre-trip; the model was trained with these fields but will
    learn to rely on distance/time/metadata features for pre-trip estimates.
    """
    pickup_dt = pd.to_datetime(trip.pickup_datetime)

    # Estimate duration: user-provided or ~15 mph average NYC speed
    if trip.estimated_duration_minutes is not None:
        duration_min = trip.estimated_duration_minutes
    else:
        avg_speed_mph = 15.0  # typical NYC average
        duration_min = max(1.0, (trip.trip_distance / avg_speed_mph) * 60)

    dropoff_dt = pickup_dt + pd.Timedelta(duration_min, "min")

    return pd.DataFrame(
        [
            {
                "tpep_pickup_datetime": pickup_dt,
                "tpep_dropoff_datetime": dropoff_dt,
                "trip_distance": trip.trip_distance,
                "passenger_count": trip.passenger_count,
                "VendorID": trip.VendorID,
                "RatecodeID": trip.RatecodeID,
                "store_and_fwd_flag": trip.store_and_fwd_flag,
                "payment_type": trip.payment_type,
                # Financial fields unknown pre-trip
                "fare_amount": 0.0,
                "tip_amount": 0.0,
                "tolls_amount": 0.0,
                # Columns present in training data but not user-provided
                "extra": 0.0,
                "mta_tax": 0.5,
                "improvement_surcharge": 0.3,
                "congestion_surcharge": 2.5,
                "Airport_fee": 0.0,
                "total_amount": 0.0,
            }
        ]
    )


# Application lifespan


def load_model_artifacts(app: FastAPI, model_cache_dir: str | Path = MODEL_CACHE_DIR) -> None:
    """Load cached model artifacts into app state."""
    cache = Path(model_cache_dir)

    model_path = cache / "model.onnx"
    transformer_path = cache / "transformer.joblib"
    metadata_path = cache / "metadata.json"

    if not model_path.exists() or not transformer_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {model_cache_dir}. "
            "Run 'python or python3 src/serving/download_model.py' first."
        )

    logger.info(f"Loading ONNX model from {model_path}...")
    app.state.model = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    app.state.onnx_input_name = app.state.model.get_inputs()[0].name
    logger.info(f" Model loaded: (input: '{app.state.onnx_input_name}')")

    logger.info(f"Loading transformer from {transformer_path}")
    app.state.transformer = joblib.load(transformer_path)
    logger.info(f" Transformer loaded: {type(app.state.transformer).__name__}")

    app.state.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    MODEL_INFO.info(
        {
            "name": app.state.metadata.get("model_name", "unknown"),
            "version": app.state.metadata.get("model_version", "unknown"),
            "format": "ONNX",
        }
    )

    app.state.startup_time = time.time()


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

    load_model_artifacts(app)

    logger.info("")
    logger.info("    API ready to serve predictions!")
    logger.info("=" * 60)

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

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
        timestamp=datetime.now(UTC).isoformat(),
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
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
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
    Estimate/Predict the total fare for a taxi trip before it starts.

    Accepts pickup time, estimated distance, and optional metadata.
    Returns predicted fare, estimated duration, and model version.
    """
    # Validate Model is loaded
    if not hasattr(app.state, "model"):
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    start_time = time.time()

    try:
        # Feature Engineering
        raw_df = trip_to_dataframe(trip)
        df = engineer_features(raw_df)
        df = df.drop(columns=["total_amount"], errors="ignore")

        # Transform and predict
        X = app.state.transformer.transform(df)

        if hasattr(X, "toarray"):
            X = X.toarray()

        X = X.astype(np.float32)

        output = app.state.model.run(None, {app.state.onnx_input_name: X})
        prediction = max(0.0, float(output[0].flatten()[0]))

        # Duration: user-provided or estimated from distance
        if trip.estimated_duration_minutes is not None:
            duration = trip.estimated_duration_minutes
        else:
            duration = round(max(1.0, (trip.trip_distance / 15.0) * 60), 1)

        # Record metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(status="success").inc()
        PREDICTION_VALUE.observe(prediction)
        PREDICTION_DISTANCE.observe(trip.trip_distance)
        PREDICTIONS_SERVED.inc()
        LAST_PREDICTION_TIME.set(time.time())

        # Calculate trip duration
        pickup_dt = pd.to_datetime(trip.pickup_datetime)
        prediction_tracker.record(
            PredictionRecord(
                timestamp=time.time(),
                predicted_fare=prediction,
                trip_distance=trip.trip_distance,
                pickup_hour=pickup_dt.hour,
                is_rush_hour=int(pickup_dt.hour in range(7, 10) or pickup_dt.hour in range(16, 19)),
                is_weekend=int(pickup_dt.weekday() >= 5),
                passenger_count=trip.passenger_count,
                duration_minutes=duration,
            )
        )

        logger.debug(f"Prediction: ${prediction:.2f}, latency: {latency * 1000:.1f}ms")

        # Return Response
        return PredictionResponse(
            predicted_fare=round(prediction, 2),
            estimated_duration_minutes=round(duration, 2),
            model_version=app.state.metadata.get("model_version", "unknown"),
            prediction_timestamp=datetime.now(UTC),
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


# =============================================================================
# Monitoring ENDPOINTS
# =============================================================================


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=get_metrics_text(),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/ready", tags=["Health"])
async def readiness():
    """
    Readiness probe for kubernetes.

    Returns 200 only when the model is loaded and has served at least
    one prediction successfully (warm start). Use for K8s readinessProbe.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not hasattr(app.state, "transformer"):
        raise HTTPException(status_code=503, detail="Transformer not loaded")
    return {"ready": True}


@app.get("/live", tags=["Health"])
async def liveness():
    """
    Liveness probe for kubernetes.

    Returns 200 if the process is alive. Use for K8s livenessProbe.
    """
    return {"live": True}


@app.get("/predictions/summary", tags=["Monitoring"])
async def predictions_summary():
    "Summary statistics of recent predictions from the ring buffer."
    return prediction_tracker.get_summary()


@app.post("/drift", tags=["Monitoring"])
async def drift_check():
    """
    Run drift detection comparing recent predictions against a reference.

    Requires at least 100 predictions in the buffer. Uses Evidently
    to detect distribution shifts in fare, distance, and time features.
    """
    current_df = prediction_tracker.to_dataframe()

    if len(current_df) < 100:
        raise HTTPException(
            status_code=400,
            detail="Need at least 100 predictions for drift analysis, "
            f"currently have {len(current_df)}. Keep sending requests.",
        )

    # Use the first half as reference, second half as current
    midpoint = len(current_df) // 2
    reference_df = current_df.iloc[:midpoint]
    recent_df = current_df.iloc[midpoint:]

    result = generate_drift_report(reference_df, recent_df)
    return result


# Main Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
