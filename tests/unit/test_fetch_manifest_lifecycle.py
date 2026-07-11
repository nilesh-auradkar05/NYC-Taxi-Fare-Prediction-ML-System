from botocore.exceptions import ClientError

from ingestion.lambdas.fetch_tlc_to_bronze import write_manifest_item


class RecordingTable:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def put_item(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {}


def _item() -> dict:
    return {
        "service": "yellow",
        "year_month": "2025-07",
        "etag": "etag-1",
        "status": "fetched",
        "bronze_s3_uri": "s3://bucket/yellow-2025-07.parquet",
    }


def test_fetch_can_replace_same_etag_when_manifest_is_incomplete() -> None:
    table = RecordingTable()

    write_manifest_item(table, _item())

    call = table.calls[0]
    assert "#status <> :complete" in call["ConditionExpression"]
    assert call["ExpressionAttributeValues"][":etag"] == "etag-1"
    assert call["ExpressionAttributeValues"][":complete"] == "complete"


def test_completed_same_etag_conditional_failure_is_idempotent() -> None:
    error = ClientError(
        {
            "Error": {
                "Code": "ConditionalCheckFailedException",
                "Message": "same completed etag",
            }
        },
        "PutItem",
    )
    table = RecordingTable(error)

    write_manifest_item(table, _item())

    assert len(table.calls) == 1


def test_force_write_is_unconditional() -> None:
    table = RecordingTable()

    write_manifest_item(table, _item(), force=True)

    assert table.calls == [{"Item": _item()}]
