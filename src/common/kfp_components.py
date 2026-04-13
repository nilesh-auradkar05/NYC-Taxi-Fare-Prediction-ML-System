"""
NYC Taxi Fare Prediction - KFP Components
=========================================

Each component is a self-contained function that runs inside the
nyc-taxing-training Docker image.
"""
from collections import namedtuple

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output

TRAINING_IMAGE = "nyc-taxi-training:latest"

CVOutput = namedtuple("CVOutput", ["mse", "r2"])
AggOutput = namedtuple("AggOutput", ["avg_mse", "std_mse", "avg_r2", "std_r2"])
TrainOutput = namedtuple("TrainOutput", ["test_mse", "test_r2", "mlflow_run_id"])


# Component 1: Load Data + Feature Engineering

@dsl.component(base_image=TRAINING_IMAGE)
def load_and_engineer_features(
    dataset_path: str,
    engineered_data: Output[Dataset],
):
    """Load raw parquet file and engineer features."""

    import pandas as pd
    from .features import engineer_features

    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df):,} rows from {dataset_path}")

    df = engineer_features(df)
    print(f"Engineered {len(df.columns)} columns")

    df.to_parquet(engineered_data.path)

# Component 2: Transform + Train-Test Split

@dsl.component(base_image=TRAINING_IMAGE)
def transform_and_split_data(
    engineered_data: Input[Dataset],
    x_train: Output[Dataset],
    y_train: Output[Dataset],
    x_test: Output[Dataset],
    y_test: Output[Dataset],
    fitted_transformer: Output[Artifact],
):
    """Fit the preprocessing transformer and split into train/test (time-based)."""

    import joblib
    import pandas as pd
    
    from .features import build_transformer, TARGET_COLUMN

    df = pd.read_parquet(engineered_data.path)

    # Time-based split: train on earlier trips, test on later ones
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    transformer = build_transformer()
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    print(f"Train: {X_train_transformed.shape}, Test: {X_test_transformed.shape}")

    # Save transformed data
    joblib.dump(X_train_transformed, x_train.path)
    joblib.dump(y_train, y_train.path)
    
    joblib.dump(X_test_transformed, x_test.path)
    joblib.dump(y_test, y_test.path)
    
    joblib.dump(transformer, fitted_transformer.path)

# Component 3: Cross-Validation Fold

@dsl.component(base_image=TRAINING_IMAGE)
def cross_validate_fold(
    x_train: Input[Dataset],
    y_train: Input[Dataset],
    fold_index: int,
    n_splits: int,
    n_estimators: int,
) -> CVOutput:
    """Train and evaluate on a single CV fold."""

    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import KFold
    from .features import build_model

    X = joblib.load(x_train.path)
    y = joblib.load(y_train.path)

    # Reconstruct the same fold split in every parallel container
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=47)
    folds = list(kf.split(X))
    train_idx, val_idx = folds[fold_index]

    model = build_model(n_estimators=n_estimators)
    model.fit(
        X[train_idx], y.iloc[train_idx],
        eval_set=[(X[val_idx], y.iloc[val_idx])],
        verbose=False,
    )

    y_pred = model.predict(X[val_idx])
    mse = float(mean_squared_error(y.iloc[val_idx], y_pred))
    r2 = float(r2_score(y.iloc[val_idx], y_pred))

    print(f"Fold {fold_index + 1}/{n_splits} - MSE: {mse:.4f}, R2: {r2:.4f}")

    return CVOutput(mse=mse, r2=r2)


# Component 4: Aggregate CV Results

@dsl.component(base_image=TRAINING_IMAGE)
def aggregate_cv_results(
    mse_scores: list,
    r2_scores: list,
    cv_metrics: Output[Metrics],
) -> AggOutput:
    """Combine cross-validation metrics from all folds."""
    import numpy as np

    avg_mse = float(np.mean(mse_scores))
    std_mse = float(np.std(mse_scores))
    avg_r2 = float(np.mean(r2_scores))
    std_r2 = float(np.std(r2_scores))

    print(f"CV Results - MSE: {avg_mse:.4f} (+/- {std_mse:.4f})")
    print(f"CV Results - R2: {avg_r2:.4f} (+/- {std_r2:.4f})")

    # Log to KFP's built-in metrics UI
    cv_metrics.log_metric("cv_avg_mse", avg_mse)
    cv_metrics.log_metric("cv_std_mse", std_mse)
    cv_metrics.log_metric("cv_avg_r2", avg_r2)
    cv_metrics.log_metric("cv_std_r2", std_r2)

    return AggOutput(avg_mse=avg_mse, std_mse=std_mse, avg_r2=avg_r2, std_r2=std_r2)


# Component 5: Train Final Model

@dsl.component(base_image=TRAINING_IMAGE)
def train_final_model(
    x_train: Input[Dataset],
    y_train: Input[Dataset],
    x_test: Input[Dataset],
    y_test: Input[Dataset],
    n_estimators: int,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    avg_cv_mse: float,
    std_cv_mse: float,
    avg_cv_r2: float,
    std_cv_r2: float,
    trained_model: Output[Model],
    test_metrics: Output[Metrics],
) -> TrainOutput:
    """Train the final model on the full train set and log to MLFlow"""
    import joblib
    import mlflow
    from sklearn.metrics import mean_squared_error, r2_score
    from .features import build_model

    test_mse = 0.0
    test_r2 = 0.0
    mlflow_run_id = ""
    model = None

    X_train = joblib.load(x_train.path)
    y_train_data = joblib.load(y_train.path)
    X_test = joblib.load(x_test.path)
    y_test_data = joblib.load(y_test.path)

    # Connect to local MLFlow server
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name="nyc-taxi-training-kfp") as run:
        mlflow.autolog(log_models=False)

        model = build_model(n_estimators=n_estimators)
        model.fit(
            X_train, y_train_data,
            eval_set=[(X_test, y_test_data)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        test_mse = float(mean_squared_error(y_test_data, y_pred))
        test_r2 = float(r2_score(y_test_data, y_pred))

        mlflow.log_metrics({
            "test_mse": test_mse,
            "test_r2": test_r2,
            "cv_avg_mse": avg_cv_mse,
            "cv_std_mse": std_cv_mse,
            "cv_avg_r2": avg_cv_r2,
            "cv_std_r2": std_cv_r2,
        })
        mlflow.log_params({"n_estimators": n_estimators})

        mlflow_run_id = run.info.run_id

    print(f"Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")
    print(f"MLFlow Run: {mlflow_run_id}")

    # Save model as artifact for the register step
    joblib.dump(model, trained_model.path)

    # Log to KFP UI
    test_metrics.log_metric("test_mse", test_mse)
    test_metrics.log_metric("test_r2", test_r2)

    return TrainOutput(test_mse=test_mse, test_r2=test_r2, mlflow_run_id=mlflow_run_id)

# Component 6: ONNX Conversion + MLFlow Registration

@dsl.component(base_image=TRAINING_IMAGE)
def register_onnx_model(
    trained_model: Input[Model],
    fitted_transformer: Input[Artifact],
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_run_id: str,
    model_name: str,
):
    """Convert XGBoost to ONNX and register in MLflow Model Registry"""

    import joblib
    import mlflow
    from .features import convert_to_onnx

    model = joblib.load(trained_model.path)
    transformer = joblib.load(fitted_transformer.path)

    print(f"Converting {type(model).__name__} to ONNX....")
    onnx_model = convert_to_onnx(model, transformer)
    print("ONNX conversion successful")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_id=mlflow_run_id):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Log transformer for serving layer
            transformer_path = str(Path(tmp_dir) / "features.joblib")
            joblib.dump(transformer, transformer_path)
            mlflow.log_artifact(transformer_path, "preprocessing")

            # Log raw ONNX file for direct download
            onnx_path = str(Path(tmp_dir) / "model.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            mlflow.log_artifact(onnx_path, "onnx_model")

        # Register the MLflow Model Registry
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="model",
            registered_model_name=model_name,
        )

    print(f"Model registered as '{model_name}' (ONNX)")