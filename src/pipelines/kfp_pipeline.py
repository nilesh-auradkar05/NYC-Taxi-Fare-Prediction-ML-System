"""
KFP Training Pipeline for NYC Taxi Fare Prediction

Pipeline DAG:
    load_and_engineer_features
            │
    transform_and_split
            │
    ┌───────┼───────┐
  fold_0  fold_1  fold_2  ...  (ParallelFor)
    └───────┼───────┘
    aggregate_cv_results
            │
    train_final_model
            │
    register_onnx_model

Usage:
    # Compile to YAML (upload to Kubeflow UI or submit via SDK)
    python src/pipelines/kfp_pipeline.py

    # Submit directly to a running Kubeflow cluster
    python src/pipelines/kfp_pipeline.py --submit --host http://kubeflow.local
"""

import argparse
from pathlib import Path
import sys

from kfp import compiler, dsl

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.common.kfp_components import (
    load_and_engineer_features,
    transform_and_split_data,
    cross_validate_fold,
    aggregate_cv_results,
    train_final_model,
    register_onnx_model,
)

@dsl.pipeline(
    name="nyc-taxi-training",
    description="Train XGBoost model with cross-validation. convert to ONNX and register in MLflow"
)
def training_pipeline(
    dataset_path: str = "Dataset/yellow_tripdata_2025-09.parquet",
    n_estimators: int = 100,
    n_cv_folds: int = 5,
    accuracy_threshold: float = 0.7,
    model_name: str = "nyc-taxi-model",
    mlflow_tracking_uri: str = "http://127.0.0.1:5000",
    mlflow_experiment_name: str = "nyc-taxi-kubeflow-experiment",
):
    # 1. Load raw data and create features
    load_step = load_and_engineer_features(dataset_path=dataset_path)

    # 2. Fit Transformer, split data into train/test
    split_step = transform_and_split_data(
        engineered_data=load_step.outputs["engineered_data"],
    )

    # 3. Parallel cross-validation
    # ParallelFor deploys one container per fold
    with dsl.ParallelFor(
        items=list(range(n_cv_folds)),
        parallelism=n_cv_folds,
    ) as fold_index:

        cv_step = cross_validate_fold(
            x_train=split_step.outputs["x_train"],
            y_train=split_step.outputs["y_train"],
            fold_index=fold_index,
            n_splits=n_cv_folds,
            n_estimators=n_estimators,
        )

    # 4. Aggregate cross-validation results
    # dsl.Collected gathers outputs from all parallel iterations into lists
    agg_step = aggregate_cv_results(
        mse_scores=dsl.Collected(cv_step.outputs["mse"]),
        r2_scores=dsl.Collected(cv_step.outputs["r2"]),
    )

    # 5. Train final model on full training set
    train_step = train_final_model(
        x_train=split_step.outputs["x_train"],
        y_train=split_step.outputs["y_train"],
        x_test=split_step.outputs["x_test"],
        y_test=split_step.outputs["y_test"],
        n_estimators=n_estimators,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        avg_cv_mse=agg_step.outputs["avg_mse"],
        std_cv_mse=agg_step.outputs["std_mse"],
        avg_cv_r2=agg_step.outputs["avg_r2"],
        std_cv_r2=agg_step.outputs["std_r2"],
    )

    # 6. Convert to ONNX and register
    with dsl.Condition(
        train_step.outputs["test_r2"] >= accuracy_threshold,
        name="model-quality-gate",
    ):
        register_onnx_model(
            trained_model=train_step.outputs["trained_model"],
            fitted_transformer=split_step.outputs["fitted_transformer"],
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_run_id=train_step.outputs["mlflow_run_id"],
            model_name=model_name,
        )


def main():
    parser = argparse.ArgumentParser(description="Compile or submit KFP pipeline")
    parser.add_argument(
        "--output", default="pipeline.yaml",
        help="Path for compiled yaml (default: pipeline.yaml)",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Submit to a running kubeflow cluster instead of compiling",
    )
    parser.add_argument(
        "--host", default="http://localhost:8080",
        help="Kubeflow pipelines endpoint (used with --submit)",
    )
    args = parser.parse_args()

    if args.submit:
        from kfp.client import Client

        client = Client(host=args.host)
        client.create_run_from_pipeline_func(
            training_pipeline,
            arguments={},
            experiment_name="nyc-taxi-training-kubeflow-experiment",
        )
        print(f"Pipeline submitted to {args.host}")
    else:
        compiler.Compiler().compile(
            pipeline_func=training_pipeline,
            package_path=args.output,
        )
        print(f"Pipeline compiled to {args.output}")

if __name__ == "__main__":
    main()