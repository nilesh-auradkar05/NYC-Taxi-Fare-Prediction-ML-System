from ingestion.lambdas.shared.tlc_manifest import (
    bronze_keys,
    build_tlc_url,
    should_skip_existing,
    validate_event,
)


def test_same_service_month_etag_skips_when_fetched() -> None:
    existing = {
        "service": "yellow",
        "year_month": "2025-01",
        "etag": "abc123",
        "status": "fetched",
    }

    assert should_skip_existing(existing, "abc123") is True


def test_changed_etag_does_not_skip() -> None:
    existing = {
        "service": "yellow",
        "year_month": "2025-01",
        "etag": "old-etag",
        "status": "fetched",
    }

    assert should_skip_existing(existing, "new-etag") is False


def test_force_does_not_skip_even_when_etag_same() -> None:
    existing = {
        "service": "yellow",
        "year_month": "2025-01",
        "etag": "abc123",
        "status": "fetched",
    }

    assert should_skip_existing(existing, "abc123", force=True) is False


def test_no_existing_manifest_does_not_skip() -> None:
    assert should_skip_existing(None, "abc123") is False


def test_build_tlc_url_for_yellow() -> None:
    assert (
        build_tlc_url("yellow", "2025-01")
        == "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
    )


def test_build_tlc_url_for_hvfhv_uses_tlc_file_prefix() -> None:
    assert (
        build_tlc_url("hvfhv", "2025-01")
        == "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2025-01.parquet"
    )


def test_bronze_keys() -> None:
    data_key, meta_key = bronze_keys("yellow", "2025-01")

    assert data_key == "bronze/service=yellow/year_month=2025-01/yellow_tripdata_2025-01.parquet"
    assert meta_key == "bronze/service=yellow/year_month=2025-01/_meta/manifest.json"


def test_validate_event_accepts_valid_yellow_event() -> None:
    service, year_month, source_url, force = validate_event(
        {"service": "yellow", "year_month": "2025-01"}
    )

    assert service == "yellow"
    assert year_month == "2025-01"
    assert source_url is None
    assert force is False
