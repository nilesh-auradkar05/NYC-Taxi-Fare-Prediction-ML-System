from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from ingestion.jobs.conform import _quote_identifier, conform_trips


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("test-conform")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture()
def zones(spark: SparkSession):
    return spark.createDataFrame(
        [
            {"LocationID": "1", "Borough": "Manhattan", "Zone": "East Village", 
             "service_zone": "Yellow Zone"},
            {"LocationID": "2", "Borough": "Queens", "Zone": "Astoria", "service_zone": 
             "Boro Zone"},
        ]
    )


def yellow_row(**overrides):
    row = {
        "VendorID": "1",
        "tpep_pickup_datetime": "2024-01-15 10:00:00",
        "tpep_dropoff_datetime": "2024-01-15 10:20:00",
        "passenger_count": "1",
        "trip_distance": "4.0",
        "PULocationID": "1",
        "DOLocationID": "2",
        "fare_amount": "20.00",
        "extra": "1.00",
        "mta_tax": "0.50",
        "tip_amount": "4.00",
        "tolls_amount": "0.00",
        "improvement_surcharge": "1.00",
        "congestion_surcharge": "2.50",
        "Airport_fee": "0.00",
        "total_amount": "29.00",
        "source_file": "s3://test/yellow_tripdata_2024-01.parquet",
    }
    row.update(overrides)
    return row


def test_u01_zone_join_invalid_zone_goes_to_quarantine_without_drop(spark: SparkSession, zones):
    raw = spark.createDataFrame(
        [
            yellow_row(),
            yellow_row(DOLocationID="999"),
        ]
    )

    valid, quarantine = conform_trips(raw, zones, service="yellow", year_month="2024-01")

    assert valid.count() == 1
    assert quarantine.count() == 1
    assert valid.count() + quarantine.count() == raw.count()

    rejected = quarantine.select("reason_code", "service", "year_month", "source_file").collect()[0]
    assert "BAD_DO_ZONE" in rejected.reason_code
    assert rejected.service == "yellow"
    assert rejected.year_month == "2024-01"
    assert rejected.source_file.endswith(".parquet")


def test_u03_dedupe_duplicate_trips_collapse_to_one(spark: SparkSession, zones):
    duplicate = yellow_row()
    raw = spark.createDataFrame([duplicate, duplicate])

    valid, quarantine = conform_trips(raw, zones, service="yellow", year_month="2024-01")

    assert quarantine.count() == 0
    assert valid.count() == 1
    assert valid.select("trip_id").distinct().count() == 1


def test_u04_malformed_fare_and_distance_go_to_quarantine_with_reason(spark: SparkSession, zones):
    raw = spark.createDataFrame(
        [
            yellow_row(fare_amount="not-a-fare"),
            yellow_row(trip_distance="not-a-distance"),
        ]
    )

    valid, quarantine = conform_trips(raw, zones, service="yellow", year_month="2024-01")

    assert valid.count() == 0
    reasons = {row.reason_code for row in quarantine.select("reason_code").collect()}

    assert any("BAD_FARE" in reason for reason in reasons)
    assert any("BAD_DISTANCE" in reason for reason in reasons)


def test_merge_identifier_quoting():
    assert _quote_identifier("trip_id") == "`trip_id`"
    assert _quote_identifier("source_file") == "`source_file`"
