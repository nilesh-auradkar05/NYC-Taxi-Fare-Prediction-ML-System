"""
Tests for the feature engineering module.

Verifies that engineered features have real logic (not placeholders),
imputation is intentional, and feature lists are consistent.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_transformer,
    engineer_features,
)


def _make_sample_df(n=100):
    """Create a minimal DataFrame mimicking NYC taxi data."""
    rng = np.random.default_rng(42)
    hours = rng.integers(0, 24, size=n)
    base = pd.Timestamp("2025-01-01")

    df = pd.DataFrame({
        "tpep_pickup_datetime": [base + pd.Timedelta(hours=int(h)) for h in hours],
        "tpep_dropoff_datetime": [
            base + pd.Timedelta(hours=int(h), minutes=int(rng.integers(5, 60)))
            for h in hours
        ],
        "trip_distance": rng.uniform(0.5, 20.0, size=n),
        "passenger_count": rng.integers(1, 5, size=n),
        "fare_amount": rng.uniform(3.0, 80.0, size=n),
        "tip_amount": rng.uniform(0, 20.0, size=n),
        "tolls_amount": rng.choice([0, 5.76, 6.55], size=n),
        "VendorID": rng.integers(1, 3, size=n),
        "payment_type": rng.integers(1, 5, size=n),
        "RatecodeID": rng.integers(1, 7, size=n),
        "store_and_fwd_flag": rng.choice(["Y", "N"], size=n),
        "total_amount": rng.uniform(5.0, 100.0, size=n),
    })
    return df


class TestFeatureEngineering:
    """Tests for engineer_features()."""

    def test_output_has_all_numerical_features(self):
        df = engineer_features(_make_sample_df())
        for feat in NUMERICAL_FEATURES:
            assert feat in df.columns, f"Missing numerical feature: {feat}"

    def test_output_has_all_categorical_features(self):
        df = engineer_features(_make_sample_df())
        for feat in CATEGORICAL_FEATURES:
            assert feat in df.columns, f"Missing categorical feature: {feat}"

    def test_is_rush_hour_has_real_values(self):
        """is_rush_hour should have both 0 and 1, not all zeros."""
        df = engineer_features(_make_sample_df(500))
        assert set(df["is_rush_hour"].unique()) == {0, 1}, \
            "is_rush_hour should contain both 0 and 1"

    def test_is_night_has_real_values(self):
        """is_night should have both 0 and 1."""
        df = engineer_features(_make_sample_df(500))
        assert set(df["is_night"].unique()) == {0, 1}, \
            "is_night should contain both 0 and 1"

    def test_time_of_day_has_real_categories(self):
        """time_of_day should not be 'unknown' everywhere."""
        df = engineer_features(_make_sample_df(500))
        unique = set(df["time_of_day"].unique())
        assert "unknown" not in unique, "time_of_day should not contain 'unknown'"
        assert len(unique) >= 3, "time_of_day should have multiple categories"

    def test_no_placeholder_features_exist(self):
        """Removed placeholders should NOT appear in the output."""
        df = engineer_features(_make_sample_df())
        removed = ["refund_amount", "is_full_refund", "negative_fare_category"]
        for feat in removed:
            assert feat not in df.columns, f"Placeholder feature still present: {feat}"

    def test_rush_hour_logic_correct(self):
        """7-9 AM and 4-6 PM should be rush hour."""
        base = pd.Timestamp("2025-01-06")  # Monday
        rows = []
        for h in range(24):
            rows.append({
                "tpep_pickup_datetime": base + pd.Timedelta(hours=h),
                "tpep_dropoff_datetime": base + pd.Timedelta(hours=h, minutes=20),
                "trip_distance": 5.0, "passenger_count": 1,
                "fare_amount": 15.0, "tip_amount": 3.0, "tolls_amount": 0,
                "VendorID": 1, "payment_type": 1, "RatecodeID": 1,
                "store_and_fwd_flag": "N", "total_amount": 18.0,
            })
        df = engineer_features(pd.DataFrame(rows))
        rush = df.set_index("pickup_hour")["is_rush_hour"]
        # Hours 7,8,9 and 16,17,18 should be rush hour
        for h in [7, 8, 9, 16, 17, 18]:
            assert rush.loc[h] == 1, f"Hour {h} should be rush hour"
        for h in [0, 3, 10, 14, 22]:
            assert rush.loc[h] == 0, f"Hour {h} should NOT be rush hour"


class TestImputation:
    """Tests for intentional NaN handling."""

    def test_no_nans_in_output(self):
        """After engineering, there should be no NaN values."""
        df = engineer_features(_make_sample_df())
        # Only check feature columns, not datetime columns
        feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        for col in feature_cols:
            if col in df.columns:
                assert df[col].isna().sum() == 0, f"NaN found in {col}"

    def test_zero_distance_doesnt_produce_inf(self):
        """fare_per_mile with zero distance should be median-imputed, not inf."""
        df = _make_sample_df(10)
        df.loc[0, "trip_distance"] = 0.0
        result = engineer_features(df)
        assert np.isfinite(result["fare_per_mile"].iloc[0]), \
            "fare_per_mile should be finite for zero-distance trips"

    def test_zero_duration_doesnt_produce_inf(self):
        """speed_mph with zero duration should be median-imputed."""
        df = _make_sample_df(10)
        df.loc[0, "tpep_dropoff_datetime"] = df.loc[0, "tpep_pickup_datetime"]
        result = engineer_features(df)
        assert np.isfinite(result["speed_mph"].iloc[0]), \
            "speed_mph should be finite for zero-duration trips"


class TestTransformer:
    """Tests for build_transformer()."""

    def test_transformer_fits_on_engineered_data(self):
        """Transformer should fit without errors on engineered features."""
        df = engineer_features(_make_sample_df())
        transformer = build_transformer()
        X = df.drop(columns=["total_amount"], errors="ignore")
        transformed = transformer.fit_transform(X)
        assert transformed.shape[0] == len(df)
        assert transformed.shape[1] > 0

    def test_feature_lists_match_transformer_columns(self):
        """All features in NUMERICAL/CATEGORICAL lists must exist after engineering."""
        df = engineer_features(_make_sample_df())
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        for feat in all_features:
            assert feat in df.columns, \
                f"Feature '{feat}' in list but not in engineered DataFrame"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
