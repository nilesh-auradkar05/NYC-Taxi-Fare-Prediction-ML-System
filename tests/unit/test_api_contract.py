"""
Contract tests for the NYC Taxi Fare Prediction API.

Tests the request/response schemas, defaults, and data conversion
without requiring a running model or MLflow server.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.serving.api import PredictionResponse, TripInput, trip_to_dataframe


class TestTripInput:
    """Tests for the pre-trip estimator input schema."""

    def test_minimal_valid_request(self):
        """Only pickup_datetime and trip_distance are required."""
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=5.2,
        )
        assert trip.trip_distance == 5.2
        assert trip.passenger_count == 1  # default
        assert trip.VendorID == 1  # default
        assert trip.payment_type == 1  # default
        assert trip.estimated_duration_minutes is None  # optional

    def test_full_request(self):
        """All optional fields provided."""
        trip = TripInput(
            pickup_datetime="2025-06-01 17:00:00",
            trip_distance=12.5,
            estimated_duration_minutes=35.0,
            passenger_count=3,
            RatecodeID=2,
            VendorID=2,
            payment_type=2,
            store_and_fwd_flag="Y",
        )
        assert trip.estimated_duration_minutes == 35.0
        assert trip.passenger_count == 3
        assert trip.store_and_fwd_flag == "Y"

    def test_rejects_zero_distance(self):
        """Distance must be positive."""
        with pytest.raises(ValidationError):
            TripInput(pickup_datetime="2025-01-15 08:30:00", trip_distance=0.0)

    def test_rejects_negative_distance(self):
        with pytest.raises(ValidationError):
            TripInput(pickup_datetime="2025-01-15 08:30:00", trip_distance=-1.0)

    def test_rejects_missing_pickup(self):
        with pytest.raises(ValidationError):
            TripInput(trip_distance=5.0)

    def test_no_dropoff_datetime_field(self):
        """Pre-trip estimator should NOT have dropoff_datetime."""
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=5.0,
        )
        assert not hasattr(trip, "dropoff_datetime")

    def test_no_fare_amount_field(self):
        """Pre-trip estimator should NOT expose fare_amount."""
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=5.0,
        )
        assert not hasattr(trip, "fare_amount")

    def test_no_tip_amount_field(self):
        """Pre-trip estimator should NOT expose tip_amount."""
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=5.0,
        )
        assert not hasattr(trip, "tip_amount")


class TestTripToDataframe:
    """Tests for the API-to-model data conversion."""

    def test_returns_single_row_dataframe(self):
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=5.0,
        )
        df = trip_to_dataframe(trip)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_has_required_training_columns(self):
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=5.0,
        )
        df = trip_to_dataframe(trip)
        required = [
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "trip_distance", "passenger_count",
            "fare_amount", "tip_amount", "tolls_amount",
            "VendorID", "payment_type", "RatecodeID",
            "store_and_fwd_flag", "total_amount",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_financial_fields_zeroed_pretip(self):
        """fare_amount, tip_amount, tolls_amount should be 0 for pre-trip."""
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=5.0,
        )
        df = trip_to_dataframe(trip)
        assert df["fare_amount"].iloc[0] == 0.0
        assert df["tip_amount"].iloc[0] == 0.0
        assert df["tolls_amount"].iloc[0] == 0.0

    def test_duration_estimated_from_distance(self):
        """When no duration provided, estimate from distance at ~15 mph."""
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=15.0,  # 15 miles at 15 mph = 60 min
        )
        df = trip_to_dataframe(trip)
        pickup = df["tpep_pickup_datetime"].iloc[0]
        dropoff = df["tpep_dropoff_datetime"].iloc[0]
        duration_min = (dropoff - pickup).total_seconds() / 60
        assert 55 <= duration_min <= 65  # ~60 min with some tolerance

    def test_duration_uses_provided_estimate(self):
        """When user provides duration, use it instead of distance estimate."""
        trip = TripInput(
            pickup_datetime="2025-01-15 08:30:00",
            trip_distance=5.0,
            estimated_duration_minutes=45.0,
        )
        df = trip_to_dataframe(trip)
        pickup = df["tpep_pickup_datetime"].iloc[0]
        dropoff = df["tpep_dropoff_datetime"].iloc[0]
        duration_min = (dropoff - pickup).total_seconds() / 60
        assert duration_min == 45.0


class TestPredictionResponse:
    """Tests for the response schema."""

    def test_valid_response(self):
        resp = PredictionResponse(
            predicted_fare=25.50,
            estimated_duration_minutes=30.0,
            model_version="3",
            prediction_timestamp=datetime.now(),
        )
        assert resp.predicted_fare == 25.50
        assert resp.estimated_duration_minutes == 30.0

    def test_response_has_no_trip_duration_field(self):
        """Should use estimated_duration_minutes, not trip_duration_minutes."""
        resp = PredictionResponse(
            predicted_fare=10.0,
            estimated_duration_minutes=15.0,
            model_version="1",
            prediction_timestamp=datetime.now(),
        )
        assert hasattr(resp, "estimated_duration_minutes")
        assert not hasattr(resp, "trip_duration_minutes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
