from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_SERVICES = {"yellow", "hvfhv"}


@dataclass(frozen=True)
class DQRunConfig:
    service: str
    year_month: str
    catalog: str
    silver_db: str
    dq_rules_path: str
    dq_config_path: str
    crz_zones_path: str
    dq_results_s3_prefix: str
    allow_insufficient_history: bool = False


def validate_year_month(year_month: str) -> str:
    if not YEAR_MONTH_RE.match(year_month):
        raise ValueError(f"Invalid year_month={year_month!r}; expected YYYY-MM")
    month = int(year_month[-2:])
    if month < 1 or month > 12:
        raise ValueError(f"Invalid year_month={year_month!r}; month must be 01..12")
    return year_month


def validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {label}={value!r}; refusing unsafe Spark SQL identifier")
    return value


def previous_months(year_month: str, n: int) -> list[str]:
    validate_year_month(year_month)
    year = int(year_month[:4])
    month = int(year_month[5:7])

    result: list[str] = []
    for _ in range(n):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        result.append(f"{year:04d}-{month:02d}")
    return result


def read_text(path: str) -> str:
    if path.startswith("s3://"):
        import boto3

        bucket_key = path.removeprefix("s3://")
        bucket, key = bucket_key.split("/", 1)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")

    return Path(path).read_text(encoding="utf-8")


def read_json(path: str) -> dict[str, Any]:
    return json.loads(read_text(path))


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def load_crz_zone_ids(spark: SparkSession, crz_zones_path: str) -> set[int]:
    if crz_zones_path.startswith("s3://"):
        rows = (
            spark.read.option("header", "true")
            .csv(crz_zones_path)
            .select(F.col("location_id").cast("int").alias("location_id"))
            .where(F.col("location_id").isNotNull())
            .collect()
        )
        return {int(row.location_id) for row in rows}

    with Path(crz_zones_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {int(row["location_id"]) for row in reader}


def read_silver_trips(
    spark: SparkSession,
    *,
    catalog: str,
    silver_db: str,
    service: str,
    months: list[str],
) -> DataFrame:
    validate_identifier(catalog, "catalog")
    validate_identifier(silver_db, "silver_db")
    if service not in SUPPORTED_SERVICES:
        raise ValueError(f"Unsupported service={service!r}")

    table = f"{catalog}.{silver_db}.trips"
    return (
        spark.table(table)
        .where(F.col("service") == F.lit(service))
        .where(F.col("year_month").isin(months))
    )


def count_where(df: DataFrame, condition: F.Column) -> int:
    return int(df.where(condition).count())


def rate(numerator: int, denominator: int, empty_value: float = 1.0) -> float:
    if denominator == 0:
        return empty_value
    return float(numerator) / float(denominator)


def build_dq_metrics(
    spark: SparkSession,
    *,
    current_df: DataFrame,
    prior_df: DataFrame,
    service: str,
    year_month: str,
    config: dict[str, Any],
    crz_zone_ids: set[int],
    allow_insufficient_history_override: bool = False,
) -> DataFrame:
    validate_year_month(year_month)
    current_df = current_df.cache()
    prior_df = prior_df.cache()

    current_row_count = int(current_df.count())

    prior_counts = {
        row["year_month"]: int(row["count"])
        for row in prior_df.groupBy("year_month").count().collect()
    }
    prior_month_count = len(prior_counts)
    trailing_avg = (
        sum(prior_counts.values()) / prior_month_count
        if prior_month_count > 0
        else 0.0
    )

    dq01_cfg = config["dq_01"]
    min_prior_months = int(dq01_cfg["min_prior_months"])
    band_pct = float(dq01_cfg["band_pct"])
    allow_insufficient_history = bool(dq01_cfg["allow_insufficient_history"]) or allow_insufficient_history_override

    lower = trailing_avg * (1.0 - band_pct)
    upper = trailing_avg * (1.0 + band_pct)

    if prior_month_count < min_prior_months:
        dq_01_ok = int(allow_insufficient_history and current_row_count > 0)
    else:
        dq_01_ok = int(lower <= current_row_count <= upper)

    pickup_nulls = count_where(current_df, F.col("pickup_ts").isNull())
    pu_zone_nulls = count_where(current_df, F.col("pu_zone_id").isNull())
    fare_nulls = count_where(current_df, F.col("fare_amount").isNull())

    dq02_cfg = config["dq_02"]
    max_null_rate = float(dq02_cfg["max_null_rate"])
    pickup_ts_null_rate = rate(pickup_nulls, current_row_count)
    pu_zone_id_null_rate = rate(pu_zone_nulls, current_row_count)
    fare_amount_null_rate = rate(fare_nulls, current_row_count)
    dq_02_ok = int(
        pickup_ts_null_rate < max_null_rate
        and pu_zone_id_null_rate < max_null_rate
        and fare_amount_null_rate < max_null_rate
    )

    dq03_cfg = config["dq_03"]
    bad_domain_count = count_where(
        current_df,
        ~(
            (F.col("fare_amount") > float(dq03_cfg["fare_min_exclusive"]))
            & (F.col("fare_amount") <= float(dq03_cfg["fare_max_inclusive"]))
            & (F.col("trip_distance") > float(dq03_cfg["distance_min_exclusive"]))
            & (F.col("trip_distance") <= float(dq03_cfg["distance_max_inclusive"]))
            & (F.col("duration_s") > int(dq03_cfg["duration_s_min_exclusive"]))
            & (F.col("duration_s") <= int(dq03_cfg["duration_s_max_inclusive"]))
        ),
    )
    dq_03_ok = int(current_row_count > 0 and bad_domain_count == 0)

    bad_timestamp_count = count_where(
        current_df,
        ~(
            (F.col("pickup_ts") < F.col("dropoff_ts"))
            & (F.date_format(F.col("pickup_ts"), "yyyy-MM") == F.col("year_month"))
            & (F.col("year_month") == F.lit(year_month))
        ),
    )
    dq_04_ok = int(current_row_count > 0 and bad_timestamp_count == 0)

    distinct_trip_ids = int(current_df.select("trip_id").distinct().count())
    duplicate_key_rate = rate(current_row_count - distinct_trip_ids, current_row_count)
    dq_05_ok = int(duplicate_key_rate < float(config["dq_05"]["max_duplicate_rate"]))

    fee_col = F.coalesce(F.col("cbd_congestion_fee").cast("double"), F.lit(0.0))
    zero_tol = float(config["dq_06"]["fee_zero_tolerance"])
    pre_2025_cutoff = str(config["dq_06"]["pre_2025_cutoff_year_month"])
    pre_2025 = year_month < pre_2025_cutoff

    pre_2025_nonzero_fee_count = count_where(current_df, F.abs(fee_col) > zero_tol)
    dq_06_ok = int((not pre_2025) or pre_2025_nonzero_fee_count == 0)

    dq07_cfg = config["dq_07"]
    post_2025 = year_month >= str(dq07_cfg["post_2025_start_year_month"])
    expected_fee = float(dq07_cfg["expected_fee_amounts"][service])
    fee_tol = float(dq07_cfg["fee_amount_tolerance"])

    allowed_fee_condition = (F.abs(fee_col - F.lit(0.0)) <= fee_tol) | (
        F.abs(fee_col - F.lit(expected_fee)) <= fee_tol
    )
    invalid_post_fee_amount_count = count_where(current_df, ~allowed_fee_condition)

    crz_ids = sorted(crz_zone_ids)
    crz_condition = F.col("pu_zone_id").isin(crz_ids) if crz_ids else F.lit(False)
    crz_trip_count = count_where(current_df, crz_condition)
    crz_positive_fee_count = count_where(current_df, crz_condition & (fee_col > fee_tol))
    crz_positive_fee_rate = rate(crz_positive_fee_count, crz_trip_count, empty_value=0.0)

    min_crz_rows = int(dq07_cfg["min_crz_rows_for_rate_check"])
    rate_min = float(dq07_cfg["crz_positive_fee_rate_min"])
    rate_max = float(dq07_cfg["crz_positive_fee_rate_max"])

    if not post_2025:
        dq_07_ok = 1
    elif invalid_post_fee_amount_count > 0:
        dq_07_ok = 0
    elif crz_trip_count < min_crz_rows:
        dq_07_ok = 1
    else:
        dq_07_ok = int(rate_min <= crz_positive_fee_rate <= rate_max)

    payload = {
        "service": service,
        "year_month": year_month,
        "current_row_count": current_row_count,
        "prior_month_count": prior_month_count,
        "trailing_3_month_avg_row_count": float(trailing_avg),
        "row_count_lower_bound": float(lower),
        "row_count_upper_bound": float(upper),
        "pickup_ts_null_rate": float(pickup_ts_null_rate),
        "pu_zone_id_null_rate": float(pu_zone_id_null_rate),
        "fare_amount_null_rate": float(fare_amount_null_rate),
        "bad_domain_count": int(bad_domain_count),
        "bad_timestamp_count": int(bad_timestamp_count),
        "duplicate_key_rate": float(duplicate_key_rate),
        "pre_2025_nonzero_fee_count": int(pre_2025_nonzero_fee_count),
        "invalid_post_fee_amount_count": int(invalid_post_fee_amount_count),
        "crz_trip_count": int(crz_trip_count),
        "crz_positive_fee_count": int(crz_positive_fee_count),
        "crz_positive_fee_rate": float(crz_positive_fee_rate),
        "dq_01_row_count_within_trailing_3_band": dq_01_ok,
        "dq_02_required_null_rates_ok": dq_02_ok,
        "dq_03_domain_values_ok": dq_03_ok,
        "dq_04_timestamp_sanity_ok": dq_04_ok,
        "dq_05_duplicate_key_rate_ok": dq_05_ok,
        "dq_06_pre_2025_cbd_fee_zero_ok": dq_06_ok,
        "dq_07_post_2025_cbd_fee_plausibility_ok": dq_07_ok,
    }

    return spark.createDataFrame([payload])


def evaluate_with_glue_dq(
    spark: SparkSession,
    *,
    metrics_df: DataFrame,
    ruleset: str,
    dq_results_s3_prefix: str,
    service: str,
    year_month: str,
) -> DataFrame:
    try:
        from awsglue.context import GlueContext
        from awsglue.dynamicframe import DynamicFrame
        from awsgluedq.transforms import EvaluateDataQuality
    except ImportError as exc:
        raise RuntimeError(
            "AWS Glue libraries are required for DQ evaluation. "
            "Unit tests should call build_dq_metrics directly."
        ) from exc

    glue_context = GlueContext(spark.sparkContext)
    metrics_dyf = DynamicFrame.fromDF(metrics_df, glue_context, "silver_trips_dq_metrics")

    dq_results = EvaluateDataQuality.apply(
        frame=metrics_dyf,
        ruleset=ruleset,
        publishing_options={
            "dataQualityEvaluationContext": f"silver_trips_{service}_{year_month}",
            "enableDataQualityCloudWatchMetrics": True,
            "enableDataQualityResultsPublishing": True,
            "resultsS3Prefix": dq_results_s3_prefix,
        },
    )

    return dq_results.toDF()


def fail_if_dq_failed(dq_results_df: DataFrame) -> None:
    failures = dq_results_df.where(F.col("Outcome") != F.lit("Passed")).collect()
    if not failures:
        return

    rendered = [
        {
            "Rule": row["Rule"],
            "Outcome": row["Outcome"],
            "FailureReason": row["FailureReason"],
        }
        for row in failures
    ]
    raise RuntimeError(f"Glue DQ gate failed: {json.dumps(rendered, default=str)}")


def parse_args(argv: list[str]) -> DQRunConfig:
    arg_names = [
        "JOB_NAME",
        "service",
        "year_month",
        "catalog",
        "silver_db",
        "dq_rules_path",
        "dq_config_path",
        "crz_zones_path",
        "dq_results_s3_prefix",
        "allow_insufficient_history",
    ]

    try:
        from awsglue.utils import getResolvedOptions

        parsed = getResolvedOptions(argv, arg_names)
    except ImportError:
        parser = argparse.ArgumentParser()
        for arg in arg_names:
            if arg == "JOB_NAME":
                continue
            required = arg != "allow_insufficient_history"
            parser.add_argument(f"--{arg}", required=required)
        parsed = vars(parser.parse_args(argv[1:]))

    service = parsed["service"].lower()
    if service not in SUPPORTED_SERVICES:
        raise ValueError(f"Unsupported service={service!r}; expected one of {sorted(SUPPORTED_SERVICES)}")

    return DQRunConfig(
        service=service,
        year_month=validate_year_month(parsed["year_month"]),
        catalog=validate_identifier(parsed["catalog"], "catalog"),
        silver_db=validate_identifier(parsed["silver_db"], "silver_db"),
        dq_rules_path=parsed["dq_rules_path"],
        dq_config_path=parsed["dq_config_path"],
        crz_zones_path=parsed["crz_zones_path"],
        dq_results_s3_prefix=parsed["dq_results_s3_prefix"],
        allow_insufficient_history=parse_bool(parsed.get("allow_insufficient_history")),
    )


def main(argv: list[str]) -> None:
    run_config = parse_args(argv)
    spark = SparkSession.builder.appName(
        f"nyc-mobility-dq-gate-{run_config.service}-{run_config.year_month}"
    ).getOrCreate()

    dq_config = read_json(run_config.dq_config_path)
    ruleset = read_text(run_config.dq_rules_path)
    crz_zone_ids = load_crz_zone_ids(spark, run_config.crz_zones_path)

    prior_months = previous_months(
        run_config.year_month,
        int(dq_config["dq_01"]["trailing_months"]),
    )
    all_months = [run_config.year_month, *prior_months]

    trips_df = read_silver_trips(
        spark,
        catalog=run_config.catalog,
        silver_db=run_config.silver_db,
        service=run_config.service,
        months=all_months,
    )

    current_df = trips_df.where(F.col("year_month") == F.lit(run_config.year_month))
    prior_df = trips_df.where(F.col("year_month").isin(prior_months))

    metrics_df = build_dq_metrics(
        spark,
        current_df=current_df,
        prior_df=prior_df,
        service=run_config.service,
        year_month=run_config.year_month,
        config=dq_config,
        crz_zone_ids=crz_zone_ids,
        allow_insufficient_history_override=run_config.allow_insufficient_history,
    )

    metrics_df.show(truncate=False)

    dq_results_df = evaluate_with_glue_dq(
        spark,
        metrics_df=metrics_df,
        ruleset=ruleset,
        dq_results_s3_prefix=run_config.dq_results_s3_prefix,
        service=run_config.service,
        year_month=run_config.year_month,
    )

    dq_results_df.show(truncate=False)
    fail_if_dq_failed(dq_results_df)


if __name__ == "__main__":
    main(sys.argv)
