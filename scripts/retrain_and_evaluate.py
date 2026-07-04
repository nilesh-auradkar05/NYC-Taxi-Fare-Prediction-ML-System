"""
Standalone training evaluation script.

Runs the full fixed pipeline (time-based split, fold-local CV, ONNX export)
and produces and honest metrics report. Works with real or synthetic data.

Usage:
    # With real data
    uv run python3 scripts/retrain_and_evaluate.py --data Dataset/yellow_tripdata_2025-09.parquet

    # With synthetic data (for testing the pipeline)
    uv run python3 scripts/retrain_and_evaluate.py --synthetic --samples 5000

    # Output saved to metrics/evaluation_report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_model,
    build_transformer,
    clean_training_data,
    convert_to_onnx,
    engineer_features,
)


def generate_synthetic_data(n: int = 5000, seed: int = 47) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Generate pickup times spread over 30 days
    base = pd.Timestamp("2026-02-01")
    pickup_offsets = np.sort(rng.integers(0, 30 * 24 * 60, size=n))
    pickups = [base + pd.Timedelta(minutes=int(m)) for m in pickup_offsets]

    distances = rng.lognormal(mean=1.0, sigma=0.8, size=n).clip(0.1, 50)
    durations_min = distances * rng.uniform(2.0, 6.0, size=n) + rng.normal(0, 2, size=n)
    durations_min = np.maximum(durations_min, 1.0)

    dropoffs = [
        p + pd.Timedelta(minutes=float(d)) for p, d in zip(pickups, durations_min, strict=True)
    ]

    # NYC fare structure: $3 base + $2.50/mile + surcharges + noise
    fares = 3.0 + distances * 2.50 + rng.normal(0, 2, size=n)
    fares = np.maximum(fares, 2.50)

    tips = np.where(
        rng.integers(1, 5, size=n) == 1,  # ~25% cash (no recorded tip)
        0.0,
        fares * rng.uniform(0.10, 0.25, size=n),
    )
    tolls = rng.choice([0.0, 0.0, 0.0, 0.0, 5.76, 6.55, 17.34], size=n)
    surcharges = 0.50 + 0.30 + 2.50  # MTA + improvement + congestion

    totals = fares + tips + tolls + surcharges + rng.normal(0, 1, size=n)
    totals = np.maximum(totals, 3.0)

    # Injecting some negative fares
    neg_mask = rng.random(n) < 0.02
    fares[neg_mask] = -fares[neg_mask]
    totals[neg_mask] = -totals[neg_mask]

    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pickups,
            "tpep_dropoff_datetime": dropoffs,
            "trip_distance": distances,
            "passenger_count": rng.integers(0, 7, size=n),
            "fare_amount": fares,
            "tip_amount": tips,
            "tolls_amount": tolls,
            "VendorID": rng.integers(1, 3, size=n),
            "payment_type": rng.integers(1, 5, size=n),
            "RatecodeID": rng.choice([1, 1, 1, 1, 2, 3, 4, 5, 6], size=n),
            "store_and_fwd_flag": rng.choice(["N", "N", "N", "Y"], size=n),
            "extra": rng.choice([0, 0.5, 1.0, 2.5], size=n),
            "mta_tax": np.full(n, 0.5),
            "improvement_surcharge": np.full(n, 0.3),
            "congestion_surcharge": rng.choice([0, 2.5, 2.5, 2.5], size=n),
            "Airport_fee": rng.choice([0, 0, 0, 1.75], size=n),
            "total_amount": totals,
        }
    )


def run_evaluation(df: pd.DataFrame, n_cv_folds: int = 5, n_estimators: int = 100) -> dict:
    print("=" * 60)
    print("NYC Taxi - RETRAIN & EVALUATE V2")
    print("=" * 60)

    # 1. Feature Engineering
    t0 = time.time()
    print(f"\n[1/6] Engineering features on {len(df):,} rows....")
    df = engineer_features(df)
    print(f"\t->{len(df.columns)} columns, {time.time() - t0:.1f}s")

    # 1b. Data Cleaning
    print("\n[2/7] Cleaning outliers (training data only)....")
    df, clean_summary = clean_training_data(df)
    print(f"\tRemoved {clean_summary['rows_removed']:,} rows ({clean_summary['pct_removed']:.1f}%)")
    for reason, count in clean_summary["reasons"].items():
        if count > 0:
            print(f"\t{reason}: {count:,}")
    print(f"\t->{clean_summary['rows_after']:,} rows remaining")

    # 2. Time-based split
    print("\n[3/7] Time-based train/test split....")
    df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    train_end = X_train["tpep_pickup_datetime"].max()
    test_start = X_test["tpep_pickup_datetime"].min()
    print(f"\tTrain: {len(X_train):,} rows (up to {train_end})")
    print(f"\tTest: {len(X_test):,} rows (from {test_start})")

    # 3. Fold-local cross-validation
    print(f"\n[4/7] {n_cv_folds}-fold cross-validation (fold-local preprocessing)....")
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=47)
    cv_mses, cv_r2s, cv_maes = [], [], []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        fold_transformer = build_transformer()
        X_fold_train = fold_transformer.fit_transform(X_train.iloc[train_idx])
        X_fold_val = fold_transformer.transform(X_train.iloc[val_idx])

        fold_model = build_model(n_estimators=n_estimators)
        fold_model.fit(
            X_fold_train,
            y_train.iloc[train_idx],
            eval_set=[(X_fold_val, y_train.iloc[val_idx])],
            verbose=False,
        )

        y_pred = fold_model.predict(X_fold_val)
        mse = mean_squared_error(y_train.iloc[val_idx], y_pred)
        r2 = r2_score(y_train.iloc[val_idx], y_pred)
        mae = mean_absolute_error(y_train.iloc[val_idx], y_pred)
        cv_mses.append(mse)
        cv_r2s.append(r2)
        cv_maes.append(mae)
        print(f"\tFold {fold_num + 1}: R2={r2:.4f}, RMSE=${mse**0.5:.2f}, MAE=${mae:.2f}")

    cv_results = {
        "cv_r2_mean": float(np.mean(cv_r2s)),
        "cv_r2_std": float(np.std(cv_r2s)),
        "cv_rmse_mean": float(np.mean([m**0.5 for m in cv_mses])),
        "cv_rmse_std": float(np.std([m**0.5 for m in cv_mses])),
        "cv_mae_mean": float(np.mean(cv_maes)),
        "cv_mae_std": float(np.std(cv_maes)),
    }

    print(f"\n\tCV Average: R2={cv_results['cv_r2_mean']:.4f} (±{cv_results['cv_r2_std']:.4f})")
    print(
        f"\tCV Average: RMSE=${cv_results['cv_rmse_mean']:.2f} (±{cv_results['cv_rmse_std']:.2f})"
    )

    # 4. Final model on full training set
    print(f"\n[5/7] Training final model on full training set({len(X_train):,} rows)....")
    t0 = time.time()
    final_transformer = build_transformer()
    X_train_transformed = final_transformer.fit_transform(X_train)
    X_test_transformed = final_transformer.transform(X_test)

    final_model = build_model(n_estimators=n_estimators)
    final_model.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_test_transformed, y_test)],
        verbose=False,
    )
    train_time = time.time() - t0
    print(f"\t-> {train_time:.1f}s")

    # 5. Evaluate on held-out test set
    print("\n[6/7] Evaluating on held-out test set....")
    y_pred = final_model.predict(X_test_transformed)

    test_results = {
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_mse": float(mean_squared_error(y_test, y_pred)),
    }

    # Error analysis: percentile breakdown
    errors = np.abs(y_test.values - y_pred)
    test_results["error_p50"] = float(np.percentile(errors, 50))
    test_results["error_p90"] = float(np.percentile(errors, 90))
    test_results["error_p95"] = float(np.percentile(errors, 95))
    test_results["error_p99"] = float(np.percentile(errors, 99))

    print(f"\tTest R²:   {test_results['test_r2']:.4f}")
    print(f"\tTest RMSE: ${test_results['test_rmse']:.2f}")
    print(f"\tTest MAE:  ${test_results['test_mae']:.2f}")
    print(f"\tError p50: ${test_results['error_p50']:.2f}")
    print(f"\tError p90: ${test_results['error_p90']:.2f}")
    print(f"\tError p95: ${test_results['error_p95']:.2f}")

    # 6. ONNX conversion
    print("\n[7/7] Converting to ONNX....")
    onnx_model = convert_to_onnx(final_model, final_transformer)
    import tempfile

    import onnxruntime as ort

    with tempfile.NamedTemporaryFile(suffix=".onxx", delete=False) as f:
        f.write(onnx_model.SerializeToString())
        onnx_path = f.name

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    X_test_float = X_test_transformed.astype(np.float32)
    if hasattr(X_test_float, "toarray"):
        X_test_float = X_test_float.toarray().astype(np.float32)

    onnx_preds = session.run(None, {input_name: X_test_float})[0].flatten()
    onnx_diff = np.max(np.abs(y_pred - onnx_preds))
    print(f"\tONNX max prediction diff: {onnx_diff:.6f}")
    assert onnx_diff < 0.01, f"ONNX predictions diverge by {onnx_diff}"
    print("\tONNX verification: PASS")

    Path(onnx_path).unlink()

    # Build Report
    report = {
        "pipeline_version": "v2-fixed",
        "data_rows_raw": clean_summary["rows_before"],
        "data_rows_after_cleaning": clean_summary["rows_after"],
        "rows_removed": clean_summary["rows_removed"],
        "pct_removed": clean_summary["pct_removed"],
        "data_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "n_features": len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES),
        "n_estimators": n_estimators,
        "split_method": "time-based (chronological 80/20)",
        "cv_method": "5-fold with fold-local preprocessing",
        "train_time_seconds": round(train_time, 1),
        **cv_results,
        **test_results,
    }

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Retrain NYC Taxi model with the fixed pipeline and produce honest metrics."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to parquet file (e.g., Dataset/yellow_tripdata_2025-09.parquet)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real data (for pipeline testing)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of synthetic samples to generate (default: 5000)",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=100,
        help="Number of XGBoost estimators (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics/evaluation_report.json",
        help="Output path for metrics report (default: metrics/evaluation_report.json)",
    )
    args = parser.parse_args()

    if args.data:
        print(f"Loading data from {args.data}...")
        df = pd.read_parquet(args.data)
        data_source = args.data
    elif args.synthetic:
        print(f"Generating {args.samples:,} synthetic samples...")
        df = generate_synthetic_data(n=args.samples)
        data_source = f"synthetic ({args.samples:,} rows)"
    else:
        parser.error("Specify --data <path> for real data or --synthetic for test data")

    report = run_evaluation(df, n_estimators=args.estimators)
    report["data_source"] = data_source

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
