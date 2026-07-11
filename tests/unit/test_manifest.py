from ingestion.lambdas.shared.tlc_manifest import (
    MANIFEST_STATUS_COMPLETE,
    PIPELINE_ACTION_FETCH,
    PIPELINE_ACTION_RESUME,
    PIPELINE_ACTION_SKIP,
    bronze_keys,
    build_tlc_url,
    determine_pipeline_action,
    should_skip_existing,
    validate_event,
)


def _manifest(status: str, *, etag: str = "abc123", include_bronze: bool = True):
    item = {
        "service": "yellow",
        "year_month": "2025-01",
        "etag": etag,
        "status": status,
    }
    if include_bronze:
        item["bronze_s3_uri"] = "s3://bucket/path/file.parquet"
    return item


def test_same_etag_skips_only_when_pipeline_complete() -> None:
    existing = _manifest(MANIFEST_STATUS_COMPLETE)

    assert determine_pipeline_action(existing, "abc123") == PIPELINE_ACTION_SKIP
    assert should_skip_existing(existing, "abc123") is True


def test_same_etag_fetched_manifest_resumes_from_bronze() -> None:
    existing = _manifest("fetched")

    assert determine_pipeline_action(existing, "abc123") == PIPELINE_ACTION_RESUME
    assert should_skip_existing(existing, "abc123") is False


def test_same_etag_processing_manifest_resumes_from_bronze() -> None:
    existing = _manifest("processing")

    assert determine_pipeline_action(existing, "abc123") == PIPELINE_ACTION_RESUME


def test_same_etag_failed_manifest_resumes_from_bronze() -> None:
    existing = _manifest("failed")

    assert determine_pipeline_action(existing, "abc123") == PIPELINE_ACTION_RESUME


def test_incomplete_manifest_without_bronze_uri_refetches() -> None:
    existing = _manifest("failed", include_bronze=False)

    assert determine_pipeline_action(existing, "abc123") == PIPELINE_ACTION_FETCH


def test_changed_etag_fetches() -> None:
    existing = _manifest(MANIFEST_STATUS_COMPLETE, etag="old-etag")

    assert determine_pipeline_action(existing, "new-etag") == PIPELINE_ACTION_FETCH
    assert should_skip_existing(existing, "new-etag") is False


def test_force_fetches_even_when_etag_same_and_complete() -> None:
    existing = _manifest(MANIFEST_STATUS_COMPLETE)

    assert (
        determine_pipeline_action(existing, "abc123", force=True)
        == PIPELINE_ACTION_FETCH
    )
    assert should_skip_existing(existing, "abc123", force=True) is False


def test_no_existing_manifest_fetches() -> None:
    assert determine_pipeline_action(None, "abc123") == PIPELINE_ACTION_FETCH
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

    assert (
        data_key
        == "bronze/service=yellow/year_month=2025-01/yellow_tripdata_2025-01.parquet"
    )
    assert meta_key == "bronze/service=yellow/year_month=2025-01/_meta/manifest.json"


def test_validate_event_accepts_valid_yellow_event() -> None:
    service, year_month, source_url, force = validate_event(
        {"service": "yellow", "year_month": "2025-01"}
    )

    assert service == "yellow"
    assert year_month == "2025-01"
    assert source_url is None
    assert force is False
