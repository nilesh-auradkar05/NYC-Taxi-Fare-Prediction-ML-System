"""
End-to-end smoke test for the NYC Taxi Fare Prediction API.

Trains a tiny XGBoost model on synthetic data, converts to ONNX,
and exercises the full prediction pipeline through FastAPI's TestClient.
No external services required (no MLflow, no MinIO).
"""

import asyncio
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.features import (
    TARGET_COLUMN,
    build_model,
    build_transformer,
    convert_to_onnx,
    engineer_features,
)


class SyncASGIClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, url, **kwargs):
        async def send_request():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send_request())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


def _make_training_data(n=200):
    """Create synthetic taxi data for training a fixture model."""
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2025-01-01")

    distances = rng.uniform(0.5, 20.0, size=n)
    durations = distances * rng.uniform(2.0, 5.0, size=n)  # minutes
    fares = 2.50 + distances * 2.50 + durations * 0.50  # rough NYC formula

    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                base + timedelta(hours=int(rng.integers(0, 720))) for _ in range(n)
            ],
            "tpep_dropoff_datetime": [
                base + timedelta(hours=int(rng.integers(0, 720)), minutes=int(d))
                for d in durations
            ],
            "trip_distance": distances,
            "passenger_count": rng.integers(1, 5, size=n),
            "fare_amount": fares + rng.normal(0, 2, size=n),
            "tip_amount": fares * rng.uniform(0, 0.25, size=n),
            "tolls_amount": rng.choice([0.0, 0.0, 0.0, 5.76, 6.55], size=n),
            "VendorID": rng.integers(1, 3, size=n),
            "payment_type": rng.integers(1, 5, size=n),
            "RatecodeID": rng.integers(1, 7, size=n),
            "store_and_fwd_flag": rng.choice(["Y", "N"], size=n),
            "total_amount": fares + rng.normal(0, 3, size=n),
        }
    )
    return df


@pytest.fixture(scope="module")
def model_cache_dir():
    """
    Train a tiny model on synthetic data, save ONNX + transformer to a temp dir.
    This fixture runs once per test module.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Generate data and engineer features
        raw_df = _make_training_data(200)
        df = engineer_features(raw_df)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # 2. Fit transformer and model
        transformer = build_transformer()
        X_transformed = transformer.fit_transform(X)

        model = build_model(n_estimators=10, max_depth=3)
        model.fit(X_transformed, y, verbose=False)

        # 3. Convert to ONNX
        onnx_model = convert_to_onnx(model, transformer)

        # 4. Save artifacts
        onnx_path = Path(tmpdir) / "model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        transformer_path = Path(tmpdir) / "transformer.joblib"
        joblib.dump(transformer, transformer_path)

        metadata = {
            "model_name": "test-model",
            "model_version": "test",
            "model_format": "ONNX",
        }
        metadata_path = Path(tmpdir) / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        yield tmpdir


@pytest.fixture(scope="module")
def client(model_cache_dir):
    """Create a FastAPI TestClient with the fixture model loaded."""
    # Override the cache dir before importing the app
    os.environ["MODEL_CACHE_DIR"] = model_cache_dir

    # Force re-import to pick up the new env var
    import importlib

    import src.serving.api as api_module

    importlib.reload(api_module)
    api_module.load_model_artifacts(api_module.app, model_cache_dir)
    yield SyncASGIClient(api_module.app)


class TestE2EPredict:
    """Full pipeline: API request → feature engineering → transform → ONNX → response."""

    def test_minimal_predict(self, client):
        """Minimal request: just pickup_datetime and trip_distance."""
        response = client.post(
            "/predict",
            json={
                "pickup_datetime": "2025-06-15 08:30:00",
                "trip_distance": 5.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_fare" in data
        assert data["predicted_fare"] > 0
        assert "estimated_duration_minutes" in data
        assert "model_version" in data
        assert "prediction_timestamp" in data

    def test_predict_with_all_options(self, client):
        """Full request with all optional fields."""
        response = client.post(
            "/predict",
            json={
                "pickup_datetime": "2025-01-15 17:30:00",
                "trip_distance": 12.5,
                "estimated_duration_minutes": 35.0,
                "passenger_count": 3,
                "RatecodeID": 2,
                "VendorID": 2,
                "payment_type": 2,
                "store_and_fwd_flag": "Y",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_fare"] > 0
        assert data["estimated_duration_minutes"] == 35.0

    def test_predict_fare_scales_with_distance(self, client):
        """Longer trips should generally predict higher fares."""
        short = client.post(
            "/predict",
            json={
                "pickup_datetime": "2025-06-15 10:00:00",
                "trip_distance": 1.0,
            },
        ).json()

        long = client.post(
            "/predict",
            json={
                "pickup_datetime": "2025-06-15 10:00:00",
                "trip_distance": 15.0,
            },
        ).json()

        assert long["predicted_fare"] > short["predicted_fare"], (
            f"15-mile trip (${long['predicted_fare']}) should cost more "
            f"than 1-mile (${short['predicted_fare']})"
        )

    def test_predict_returns_positive_fare(self, client):
        """Fare should never be negative."""
        response = client.post(
            "/predict",
            json={
                "pickup_datetime": "2025-03-01 03:00:00",
                "trip_distance": 0.5,
            },
        )
        assert response.json()["predicted_fare"] >= 0

    def test_auto_duration_estimation(self, client):
        """When no duration provided, should estimate from distance."""
        response = client.post(
            "/predict",
            json={
                "pickup_datetime": "2025-06-15 12:00:00",
                "trip_distance": 15.0,
                # no estimated_duration_minutes
            },
        )
        data = response.json()
        # 15 miles at 15 mph avg = ~60 min
        assert 50 <= data["estimated_duration_minutes"] <= 70


class TestE2EHealth:
    """Health and operational endpoints with model loaded."""

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_ready(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_live(self, client):
        response = client.get("/live")
        assert response.status_code == 200

    def test_model_info(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "test"
        assert data["model_type"] == "ONNX (XGBoost)"

    def test_metrics_endpoint(self, client):
        """Prometheus metrics should be served as text."""
        response = client.get("/metrics")
        assert response.status_code == 200
        text = response.text
        assert "prediction_requests_total" in text
        assert "prediction_latency_seconds" in text

    def test_predictions_summary(self, client):
        """After predictions, summary should have data."""
        # Fire a prediction first
        client.post(
            "/predict",
            json={
                "pickup_datetime": "2025-06-15 08:30:00",
                "trip_distance": 5.0,
            },
        )
        response = client.get("/predictions/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1
        assert "fare_mean" in data

    def test_drift_requires_minimum_predictions(self, client):
        """Drift endpoint should reject when < 100 predictions buffered."""
        response = client.post("/drift")
        assert response.status_code == 400
        assert "100" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
