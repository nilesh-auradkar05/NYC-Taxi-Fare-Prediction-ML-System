"""
Monitoring module for the NYC Taxi Fare Prediction API.

Provides:
- Prometheus metrics (request count, latency, prediction distribution)
- Prediction ring buffer for drift detection
- Drift analysis using Evidently
"""

import time
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest


# PROMETHEUS METRICS

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status"], # "success" or "error"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

PREDICTION_VALUE = Histogram(
    "prediction_value_dollars",
    "Distribution of predicted fare values",
    buckets=[5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200],
)

PREDICTION_DISTANCE = Histogram(
    "prediction_distance_miles",
    "Distribution of requested trip distances",
    buckets=[0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50],
)

MODEL_INFO = Info(
    "model",
    "Information about the loaded model",
)

PREDICTIONS_SERVED = Counter(
    "predictions_served_total",
    "Total predictions served since startup",
)

LAST_PREDICTION_TIME = Gauge(
    "last_prediction_timestamp",
    "Unix timestamp of the last prediction served",
)

def get_metrics_text() -> bytes:
    """Generate Prometheus metrics text for the /metrics endpoint."""
    return generate_latest()

# PREDICTION Tracker - ring buffer for drift detection

@dataclass
class PredictionRecord:
    """A single prediction record for drift tracking."""
    timestamp: float
    predicted_fare: float
    trip_distance: float
    pickup_hour: int
    is_rush_hour: int
    is_weekend: int
    passenger_count: int
    duration_minutes: float

class PredictionTracker:
    """
    Thread-safe ring buffer that stores recent predictions for drift analysis.

    Keeps the last `max_size` predictions in memory. Used by the /drift
    endpoint to compare recent prediction distributions against a reference.
    """

    def __init__(self, max_size: int = 5000):
        self._buffer: deque[PredictionRecord] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def record(self, rec: PredictionRecord):
        with self._lock:
            self._buffer.append(rec)

    @property
    def count(self) -> int:
        return len(self._buffer)

    def to_dataframe(self):
        """Export recent predictions as a DataFrame for Evidently."""
        import pandas as pd

        with self._lock:
            records = list(self._buffer)

        if not records:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "predicted_fare": r.predicted_fare,
                "trip_distance": r.trip_distance,
                "pickup_hour": r.pickup_hour,
                "is_rush_hour": r.is_rush_hour,
                "is_weekend": r.is_weekend,
                "passenger_count": r.passenger_count,
                "duration_minutes": r.duration_minutes,
            }
            for r in records
        ])

    def get_summary(self) -> dict:
        """Quick stats without building a full DataFrame."""
        with self._lock:
            if not self._buffer:
                return {"count": 0}

            fares = [r.predicted_fare for r in self._buffer]
            distances = [r.trip_distance for r in self._buffer]

        return {
            "count": len(fares),
            "fare_mean": round(float(np.mean(fares)), 2),
            "fare_std": round(float(np.std(fares)), 2),
            "fare_p50": round(float(np.median(fares)), 2),
            "fare_p95": round(float(np.percentile(fares, 95)), 2),
            "distance_mean": round(float(np.mean(distances)), 2),
            "oldest_record_age_seconds": round(time.time() - self._buffer[0].timestamp, 1),
        }

# Global tracker instance
prediction_tracker = PredictionTracker(max_size=5000)

# DRIFT DETECTION (Evidently)

def generate_drift_report(
    reference_df,
    current_df,
) -> dict:
    """
    Run Evidently drift detection comparing reference vs current predictions.

    Parameters:
        reference_df: DataFrame of reference (training-time) prediction distribution.
        current_df: DataFrame of recent predictions from the tracker.

    Returns:
        Dict with drift results per feature and overall drift flag.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)

        result = report.as_dict()

        # Extract per-column drift results
        metrics = result.get("metrics", [])
        drift_results = {}

        for metric in metrics:
            metric_result = metric.get("result", {})

            # Dataset-level drift
            if "share_of_drifted_columns" in metric_result:
                drift_results["dataset_drift"] = metric_result.get("dataset_drift", False)
                drift_results["share_drifted"] = metric_result.get("share_of_drifted_columns", 0)
                drift_results["n_drifted"] = metric_result.get("number_of_drifted_columns", 0)

            # Per-column drift
            if "drift_by_columns" in metric_result:
                columns = {}
                for col_name, col_data in metric_result["drift_by_columns"].items():
                    columns[col_name] = {
                        "drifted": col_data.get("drift_detected", False),
                        "p_value": round(col_data.get("drift_score", 0), 4),
                        "stattest": col_data.get("stattest_name", "unknown"),
                    }
                drift_results["columns"] = columns

        return drift_results
    except ImportError:
        return {
            "error": "evidently not installed.",
            "dataset_drift": None,
        }
    except Exception as e:
        return {
            "error": str(e),
            "dataset_drift": None,
        }