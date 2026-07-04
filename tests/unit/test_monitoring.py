"""
Tests for the monitoring module.

Verifies Prometheus metrics, prediction tracking, and drift detection.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.serving.monitoring import (
    PredictionRecord,
    PredictionTracker,
    generate_drift_report,
    get_metrics_text,
)


class TestPredictionTracker:
    """Tests for the prediction ring buffer."""

    def test_empty_tracker(self):
        tracker = PredictionTracker(max_size=100)
        assert tracker.count == 0
        summary = tracker.get_summary()
        assert summary["count"] == 0

    def test_record_and_count(self):
        tracker = PredictionTracker(max_size=100)
        rec = PredictionRecord(
            timestamp=time.time(),
            predicted_fare=25.50,
            trip_distance=5.2,
            pickup_hour=8,
            is_rush_hour=1,
            is_weekend=0,
            passenger_count=2,
            duration_minutes=20.0,
        )
        tracker.record(rec)
        assert tracker.count == 1

    def test_ring_buffer_evicts_old_records(self):
        tracker = PredictionTracker(max_size=5)
        for i in range(10):
            tracker.record(PredictionRecord(
                timestamp=time.time(),
                predicted_fare=float(i * 10),
                trip_distance=float(i),
                pickup_hour=i % 24,
                is_rush_hour=0,
                is_weekend=0,
                passenger_count=1,
                duration_minutes=15.0,
            ))
        assert tracker.count == 5  # max_size enforced

    def test_to_dataframe(self):
        tracker = PredictionTracker(max_size=100)
        for i in range(5):
            tracker.record(PredictionRecord(
                timestamp=time.time(),
                predicted_fare=20.0 + i,
                trip_distance=3.0 + i,
                pickup_hour=8,
                is_rush_hour=1,
                is_weekend=0,
                passenger_count=1,
                duration_minutes=15.0,
            ))
        df = tracker.to_dataframe()
        assert len(df) == 5
        assert "predicted_fare" in df.columns
        assert "trip_distance" in df.columns

    def test_empty_to_dataframe(self):
        tracker = PredictionTracker(max_size=100)
        df = tracker.to_dataframe()
        assert len(df) == 0

    def test_summary_stats(self):
        tracker = PredictionTracker(max_size=100)
        fares = [10.0, 20.0, 30.0, 40.0, 50.0]
        for fare in fares:
            tracker.record(PredictionRecord(
                timestamp=time.time(),
                predicted_fare=fare,
                trip_distance=5.0,
                pickup_hour=12,
                is_rush_hour=0,
                is_weekend=0,
                passenger_count=1,
                duration_minutes=20.0,
            ))
        summary = tracker.get_summary()
        assert summary["count"] == 5
        assert summary["fare_mean"] == 30.0
        assert summary["fare_p50"] == 30.0


class TestPrometheusMetrics:
    """Tests for Prometheus metrics output."""

    def test_metrics_text_is_bytes(self):
        text = get_metrics_text()
        assert isinstance(text, bytes)

    def test_metrics_text_contains_custom_metrics(self):
        text = get_metrics_text().decode("utf-8")
        assert "prediction_requests_total" in text
        assert "prediction_latency_seconds" in text
        assert "prediction_value_dollars" in text


class TestDriftDetection:
    """Tests for drift report generation."""

    def test_drift_report_without_evidently(self):
        """Should return gracefully if evidently is not installed."""
        import pandas as pd
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "predicted_fare": rng.normal(25, 5, 200),
            "trip_distance": rng.normal(5, 2, 200),
        })
        cur = pd.DataFrame({
            "predicted_fare": rng.normal(25, 5, 200),
            "trip_distance": rng.normal(5, 2, 200),
        })
        result = generate_drift_report(ref, cur)
        # Either returns drift results (if evidently installed) or error dict
        assert isinstance(result, dict)
        assert "dataset_drift" in result or "error" in result

    def test_drift_report_detects_shift(self):
        """If distributions are very different, drift should be flagged."""
        import pandas as pd
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "predicted_fare": rng.normal(25, 5, 500),
            "trip_distance": rng.normal(5, 2, 500),
        })
        # Dramatically shifted distribution
        cur = pd.DataFrame({
            "predicted_fare": rng.normal(100, 5, 500),
            "trip_distance": rng.normal(50, 2, 500),
        })
        result = generate_drift_report(ref, cur)
        # If evidently is installed, it should detect drift
        if "error" not in result:
            assert result.get("dataset_drift") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
