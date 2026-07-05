from ingestion.lambdas.check_manifest import handler


class FakeManifestTable:
    def __init__(self, item=None):
        self.item = item

    def get_item(self, Key, ConsistentRead=False):  # noqa: N803
        assert Key == {"service": "yellow", "year_month": "2025-01"}
        assert ConsistentRead is True

        if self.item is None:
            return {}

        return {"Item": self.item}


class FakeDynamoResource:
    def __init__(self, table):
        self.table = table

    def Table(self, table_name):  # noqa: N802
        assert table_name == "manifest-test"
        return self.table


def test_check_manifest_returns_is_new_when_no_existing_item(monkeypatch) -> None:
    import ingestion.lambdas.check_manifest as module

    monkeypatch.setenv("MANIFEST_TABLE", "manifest-test")
    monkeypatch.setenv("TLC_SOURCE_BASE_URL", "https://example.com")

    monkeypatch.setattr(
        module,
        "get_source_metadata",
        lambda source_url: {
            "etag": "etag-1",
            "last_modified": "today",
            "content_length": "123",
        },
    )

    monkeypatch.setattr(
        module.boto3,
        "resource",
        lambda service_name: FakeDynamoResource(FakeManifestTable()),
    )

    result = handler({"service": "yellow", "year_month": "2025-01"}, None)

    assert result["is_new"] is True
    assert result["etag"] == "etag-1"
    assert result["skip_reason"] is None


def test_check_manifest_returns_skip_when_etag_unchanged(monkeypatch) -> None:
    import ingestion.lambdas.check_manifest as module

    monkeypatch.setenv("MANIFEST_TABLE", "manifest-test")
    monkeypatch.setenv("TLC_SOURCE_BASE_URL", "https://example.com")

    monkeypatch.setattr(
        module,
        "get_source_metadata",
        lambda source_url: {
            "etag": "etag-1",
            "last_modified": "today",
            "content_length": "123",
        },
    )

    existing_item = {
        "service": "yellow",
        "year_month": "2025-01",
        "etag": "etag-1",
        "status": "fetched",
        "bronze_s3_uri": "s3://bucket/path/file.parquet",
    }

    monkeypatch.setattr(
        module.boto3,
        "resource",
        lambda service_name: FakeDynamoResource(FakeManifestTable(existing_item)),
    )

    result = handler({"service": "yellow", "year_month": "2025-01"}, None)

    assert result["is_new"] is False
    assert result["skip_reason"] == "unchanged_etag"
    assert result["existing_bronze_s3_uri"] == "s3://bucket/path/file.parquet"
