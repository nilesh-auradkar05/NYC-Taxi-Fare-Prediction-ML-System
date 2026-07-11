from __future__ import annotations

import json
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from ingestion.jobs.dq_gate import (
    PLATFORM_START_YEAR_MONTH,
    build_dq_metrics,
)

DQ_RULES_PATH = Path("ingestion/dq/rules/silver_trips_dq.dqdl")
DQ_CONFIG_PATH = Path("ingestion/dq/rules/silver_trips_dq_config.json")


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-dq-rules")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture()
def dq_config() -> dict:
    return json.loads(DQ_CONFIG_PATH.read_text(encoding="utf-8"))


def valid_trip(trip_id: str, year_month: str = "2024-04", fare: float = 20.0) -> dict:
    return {
        "trip_id": trip_id,
        "service": "yellow",
        "year_month": year_month,
        "pickup_ts": f"{year_month}-15 10:00:00",
        "dropoff_ts": f"{year_month}-15 10:20:00",
        "pu_zone_id": 100,
        "do_zone_id": 161,
        "fare_amount": fare,
        "trip_distance": 4.0,
        "duration_s": 1200,
        "cbd_congestion_fee": 0.0,
    }


def test_dqdl_contains_all_dq_ids_and_fail_closed_metric_flags():
    text = DQ_RULES_PATH.read_text(encoding="utf-8")

    for dq_id in ["DQ-01", "DQ-02", "DQ-03", "DQ-04", "DQ-05", "DQ-06", "DQ-07"]:
        assert dq_id in text

    expected_metric_flags = [
        "dq_01_row_count_within_trailing_3_band",
        "dq_02_required_null_rates_ok",
        "dq_03_domain_values_ok",
        "dq_04_timestamp_sanity_ok",
        "dq_05_duplicate_key_rate_ok",
        "dq_06_pre_2025_cbd_fee_zero_ok",
        "dq_07_post_2025_cbd_fee_plausibility_ok",
    ]
    for flag in expected_metric_flags:
        assert f'ColumnValues "{flag}" = 1' in text


def test_dq_config_preserves_test_plan_thresholds(dq_config: dict):
    assert dq_config["dq_01"]["band_pct"] == 0.40
    assert dq_config["dq_01"]["trailing_months"] == 3
    assert dq_config["dq_02"]["max_null_rate"] == 0.005
    assert dq_config["dq_03"]["fare_min_exclusive"] == 0.0
    assert dq_config["dq_03"]["fare_max_inclusive"] == 1000.0
    assert dq_config["dq_03"]["distance_min_exclusive"] == 0.0
    assert dq_config["dq_03"]["distance_max_inclusive"] == 200.0
    assert dq_config["dq_03"]["duration_s_min_exclusive"] == 0
    assert dq_config["dq_03"]["duration_s_max_inclusive"] == 21600
    assert dq_config["dq_05"]["max_duplicate_rate"] == 0.0001
    assert dq_config["dq_06"]["pre_2025_cutoff_year_month"] == "2025-01"
    assert dq_config["dq_07"]["expected_fee_amounts"]["yellow"] == 0.75
    assert dq_config["dq_07"]["expected_fee_amounts"]["hvfhv"] == 1.50


def test_dq_metrics_pass_for_clean_month_with_three_month_history(
    spark: SparkSession,
    dq_config: dict,
):
    current = spark.createDataFrame(
        [valid_trip(f"cur-{i}", "2024-04") for i in range(10)]
    )
    prior = spark.createDataFrame(
        [
            valid_trip(f"prior-{month}-{i}", month)
            for month in ["2024-01", "2024-02", "2024-03"]
            for i in range(10)
        ]
    )

    metrics = build_dq_metrics(
        spark,
        current_df=current,
        prior_df=prior,
        service="yellow",
        year_month="2024-04",
        config=dq_config,
        crz_zone_ids={100, 161},
    ).collect()[0].asDict()

    assert metrics["dq_01_row_count_within_trailing_3_band"] == 1
    assert metrics["dq_02_required_null_rates_ok"] == 1
    assert metrics["dq_03_domain_values_ok"] == 1
    assert metrics["dq_04_timestamp_sanity_ok"] == 1
    assert metrics["dq_05_duplicate_key_rate_ok"] == 1
    assert metrics["dq_06_pre_2025_cbd_fee_zero_ok"] == 1
    assert metrics["dq_07_post_2025_cbd_fee_plausibility_ok"] == 1


def test_dq_metrics_fail_for_poisoned_rows(spark: SparkSession, dq_config: dict):
    current = spark.createDataFrame(
        [
            valid_trip("ok-1", "2024-04"),
            valid_trip("bad-domain", "2024-04", fare=2001.0),
            {
                **valid_trip("bad-null", "2024-04"),
                "pickup_ts": None,
            },
            {
                **valid_trip("bad-time", "2024-04"),
                "dropoff_ts": "2024-04-15 09:59:00",
            },
            valid_trip("dupe", "2024-04"),
            valid_trip("dupe", "2024-04"),
        ]
    )
    prior = spark.createDataFrame(
        [
            valid_trip(f"prior-{month}-{i}", month)
            for month in ["2024-01", "2024-02", "2024-03"]
            for i in range(6)
        ]
    )

    metrics = build_dq_metrics(
        spark,
        current_df=current,
        prior_df=prior,
        service="yellow",
        year_month="2024-04",
        config=dq_config,
        crz_zone_ids={100, 161},
    ).collect()[0].asDict()

    assert metrics["dq_02_required_null_rates_ok"] == 0
    assert metrics["dq_03_domain_values_ok"] == 0
    assert metrics["dq_04_timestamp_sanity_ok"] == 0
    assert metrics["dq_05_duplicate_key_rate_ok"] == 0


def build_metrics_for_month(
    spark: SparkSession,
    dq_config: dict,
    *,
    year_month: str,
    prior_months: list[str],
    current_rows: list[dict] | None = None,
) -> dict:
    current = spark.createDataFrame(
        current_rows or [valid_trip(f"cur-{i}", year_month) for i in range(10)]
    )
    prior_rows = [
        valid_trip(f"prior-{month}-{i}", month)
        for month in prior_months
        for i in range(10)
    ]
    prior = spark.createDataFrame(prior_rows) if prior_rows else current.limit(0)

    return build_dq_metrics(
        spark,
        current_df=current,
        prior_df=prior,
        service="yellow",
        year_month=year_month,
        config=dq_config,
        crz_zone_ids={100, 161},
    ).collect()[0].asDict()


@pytest.mark.parametrize(
    (
        "year_month",
        "prior_months",
        "applicable",
        "history_complete",
        "mode",
        "missing",
        "dq01_ok",
    ),
    [
        ("2023-01", [], 0, 0, "bootstrap_not_applicable", None, 1),
        ("2023-02", ["2023-01"], 0, 0, "bootstrap_not_applicable", None, 1),
        (
            "2023-03",
            ["2023-01", "2023-02"],
            0,
            0,
            "bootstrap_not_applicable",
            None,
            1,
        ),
        (
            "2023-04",
            ["2023-01", "2023-02"],
            1,
            0,
            "missing_required_history",
            "2023-03",
            0,
        ),
        (
            "2025-06",
            [],
            1,
            0,
            "missing_required_history",
            "2025-05,2025-04,2025-03",
            0,
        ),
        (
            "2025-06",
            ["2025-03", "2025-04", "2025-05"],
            1,
            1,
            "evaluated",
            "",
            1,
        ),
    ],
)
def test_dq01_bootstrap_and_history_contract(
    spark: SparkSession,
    dq_config: dict,
    year_month: str,
    prior_months: list[str],
    applicable: int,
    history_complete: int,
    mode: str,
    missing: str | None,
    dq01_ok: int,
):
    metrics = build_metrics_for_month(
        spark,
        dq_config,
        year_month=year_month,
        prior_months=prior_months,
    )

    assert metrics["dq_01_platform_start_year_month"] == PLATFORM_START_YEAR_MONTH
    assert metrics["dq_01_applicable"] == applicable
    assert metrics["dq_01_history_complete"] == history_complete
    assert metrics["dq_01_evaluation_mode"] == mode
    assert metrics["dq_01_row_count_within_trailing_3_band"] == dq01_ok
    if missing is not None:
        assert metrics["dq_01_missing_prior_months"] == missing


def test_bootstrap_does_not_hide_other_dq_failures(
    spark: SparkSession,
    dq_config: dict,
):
    metrics = build_metrics_for_month(
        spark,
        dq_config,
        year_month="2023-01",
        prior_months=[],
        current_rows=[
            valid_trip("ok", "2023-01"),
            {**valid_trip("bad-null", "2023-01"), "pickup_ts": None},
            valid_trip("bad-domain", "2023-01", fare=2001.0),
        ],
    )

    assert metrics["dq_01_row_count_within_trailing_3_band"] == 1
    assert metrics["dq_02_required_null_rates_ok"] == 0
    assert metrics["dq_03_domain_values_ok"] == 0


def test_generic_insufficient_history_bypass_is_rejected(
    spark: SparkSession,
    dq_config: dict,
):
    unsafe_config = json.loads(json.dumps(dq_config))
    unsafe_config["dq_01"]["allow_insufficient_history"] = True
    current = spark.createDataFrame(
        [valid_trip(f"cur-{i}", "2025-06") for i in range(10)]
    )

    with pytest.raises(ValueError, match="Generic DQ-01 insufficient-history bypass"):
        build_dq_metrics(
            spark,
            current_df=current,
            prior_df=current.limit(0),
            service="yellow",
            year_month="2025-06",
            config=unsafe_config,
            crz_zone_ids={100, 161},
        )
