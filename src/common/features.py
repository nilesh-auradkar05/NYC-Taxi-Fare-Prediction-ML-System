"""
NYC Feature Engineering

This module contains the feature engineering pipeline for the NYC Taxi dataset.
Removing the redundant feature engineering steps from the training and inference pipelines.

Usage:
    from common.features import engineer_features, build_transformer, build_model

    df = engineer_features(df, target_column="total_amount")

    # No target column
    df = engineer_features(df)
"""

import numpy as np
import pandas as pd

NUMERICAL_FEATURES = [
    "trip_distance",
    "trip_duration_minutes",
    "passenger_count",
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "fare_per_mile",
    "tip_percentage",
    "speed_mph",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "has_negative_fare",
]

CATEGORICAL_FEATURES = [
    "VendorID",
    "payment_type",
    "RatecodeID",
    "store_and_fwd_flag",
    "time_of_day",
    "vendor_payment_interaction",
]

TARGET_COLUMN = "total_amount"

def _safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)

def convert_to_onnx(model, transformer):
    """
    Convert the model to ONNX format.
    """
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    # input dim = numeric features + one-hot encoded categories
    n_numeric = len(NUMERICAL_FEATURES)

    cat_transformer = transformer.named_transformers_["categorical"]
    encoder = cat_transformer.named_steps["onehotencoder"]
    n_categorical = sum(len(cats) for cats in encoder.categories_)

    n_features = n_numeric + n_categorical

    onnx_model = convert_xgboost(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=15,
    )

    return onnx_model

def engineer_features(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """
    Perform feature engineering on the input data.

    Parameters:
    -----------
    df: pd.DataFrame
        Raw taxi trip data with columns like tpep_pickup_datetime,
        trip_distance, fare_amount, etc.

    target_column: str | None
        If provided, features like revenue_per_mile use the target.
        If None, fare_amount is used as a proxy.

    Returns:
    --------
    pd.DataFrame:
        DataFrame with all engineered features added.
    """
    
    df = df.copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(df["pickup_hour"] * (2 * np.pi / 24))
    df["hour_cos"] = np.cos(df["pickup_hour"] * (2 * np.pi / 24))
    df["dayofweek_sin"] = np.sin(df["pickup_dayofweek"] * (2 * np.pi / 7))
    df["dayofweek_cos"] = np.cos(df["pickup_dayofweek"] * (2 * np.pi / 7))

    df["fare_per_mile"] = _safe_divide(df["fare_amount"], df["trip_distance"])
    df["tip_percentage"] = _safe_divide(df["tip_amount"], df['fare_amount'])

    hours = (df["trip_duration_minutes"] / 60).replace(0, np.nan)
    df["speed_mph"] = df["trip_distance"] / hours

    # Temporal classification
    df["is_rush_hour"] = (
        df["pickup_hour"].between(7, 9) | df["pickup_hour"].between(16, 18)
    ).astype(int)

    df["is_night"] = (
        (df["pickup_hour"] >= 22 | df["pickup_hour"] <= 5)
    ).astype(int)

    df["has_negative_fare"] = (df["fare_amount"] < 0).astype(int)


    df["time_of_day"] = pd.cut(
        df["pickup_hour"],
        bins=[-1, 5, 11, 16, 21, 24],
        labels=["night", "morning", "afternoon", "evening", 'late_night'],
    ).astype(str)

    df["vendor_payment_interaction"] = (
        df["VendorID"].astype(str) + "_" + df["payment_type"].astype(str)
    )

    # # Intentional imputation: computed ratios
    ratio_cols = ["fare_per_mile", "tip_percentage", "speed_mph"]
    for col in ratio_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Remaining NaNs (e.g. passenger_count) - fill with 0 only for non-ratio columns
    df.fillna(0)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str)

    return df

def build_transformer():
    """
    Build a Scikit-learn ColumnTransformer for preprocessing.

    Uses the NUMERICAL_FEATURES and CATEGORICAL_FEATURES lists to define the columns to preprocess.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERICAL_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
    )

def build_model(learning_rate=0.1, n_estimators=100, max_depth=6):
    from xgboost import XGBRegressor

    return XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        objective="reg:squarederror",
        random_state=42,
    )
