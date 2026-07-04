from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import ClientError

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")

# TLC's public page labels this as "High Volume For-Hire Vehicle",
# but the parquet file prefix is commonly "fhvhv".
SERVICE_FILE_PREFIX = {
    "yellow": "yellow_tripdata",
    "hvfhv": "fhvhv_tripdata",
}

DEFAULT_TLC_SOURCE_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def validate_event(event: dict[str, Any]) -> tuple[str, str, str | None, bool]:
    service = str(event.get("service", "")).lower().strip()
    year_month = str(event.get("year_month", "")).strip()
    source_url = event.get("source_url")
    force = bool(event.get("force", False))

    if service not in SERVICE_FILE_PREFIX:
        expected_services = sorted(SERVICE_FILE_PREFIX)
        raise ValueError(f"Unsupported service={service!r}; expected one of {expected_services}")

    if not YEAR_MONTH_RE.match(year_month):
        raise ValueError(f"Invalid year_month={year_month!r}; expected YYYY-MM")

    if source_url is not None:
        source_url = str(source_url).strip()
        if not source_url.startswith("https://"):
            raise ValueError("source_url must be https://")

    return service, year_month, source_url, force


def build_tlc_url(
    service: str,
    year_month: str,
    base_url: str = DEFAULT_TLC_SOURCE_BASE_URL,
) -> str:
    prefix = SERVICE_FILE_PREFIX[service]
    return f"{base_url.rstrip('/')}/{prefix}_{year_month}.parquet"


def get_source_metadata(source_url: str) -> dict[str, str]:
    request = Request(
        source_url,
        method="HEAD",
        headers={"User-Agent": "nyc-mobility-platform-v3/0.1"},
    )

    with urlopen(request, timeout=30) as response:
        headers = response.headers
        etag = headers.get("ETag", "").strip('"')
        last_modified = headers.get("Last-Modified", "")
        content_length = headers.get("Content-Length", "")

    # Some public endpoints are inconsistent with ETags.
    # Use a deterministic fallback so idempotency still has a source version.
    source_version = etag or f"last_modified={last_modified};content_length={content_length}"

    if not source_version or source_version == "last_modified=;content_length=":
        raise ValueError(f"Could not derive source version from HEAD metadata for {source_url}")

    return {
        "etag": source_version,
        "last_modified": last_modified,
        "content_length": content_length,
    }


def should_skip_existing(
    existing_item: dict[str, Any] | None,
    source_etag: str,
    force: bool = False,
) -> bool:
    if force:
        return False

    if not existing_item:
        return False

    return (
        existing_item.get("etag") == source_etag
        and existing_item.get("status") == "fetched"
    )


def bronze_keys(service: str, year_month: str) -> tuple[str, str]:
    prefix = SERVICE_FILE_PREFIX[service]
    data_key = f"bronze/service={service}/year_month={year_month}/{prefix}_{year_month}.parquet"
    meta_key = f"bronze/service={service}/year_month={year_month}/_meta/manifest.json"
    return data_key, meta_key


def fetch_to_s3(source_url: str, bucket: str, key: str, s3_client: Any) -> None:
    request = Request(
        source_url,
        method="GET",
        headers={"User-Agent": "nyc-mobility-platform-v3/0.1"},
    )

    with urlopen(request, timeout=900) as response:
        s3_client.upload_fileobj(
            Fileobj=response,
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": "application/octet-stream"},
        )


def put_manifest_metadata(bucket: str, key: str, metadata: dict[str, Any], s3_client: Any) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )


def get_existing_manifest(table: Any, service: str, year_month: str) -> dict[str, Any] | None:
    response = table.get_item(
        Key={"service": service, "year_month": year_month},
        ConsistentRead=True,
    )
    return response.get("Item")


def write_manifest_item(
    table: Any,
    item: dict[str, Any],
    force: bool = False,
) -> None:
    if force:
        table.put_item(Item=item)
        return

    try:
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(#service) OR etag <> :etag",
            ExpressionAttributeNames={"#service": "service"},
            ExpressionAttributeValues={":etag": item["etag"]},
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code == "ConditionalCheckFailedException":
            LOGGER.info("Manifest already contains same etag; treating as idempotent skip.")
            return
        raise


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    del context

    bronze_bucket = os.environ["BRONZE_BUCKET"]
    manifest_table_name = os.environ["MANIFEST_TABLE"]
    base_url = os.environ.get("TLC_SOURCE_BASE_URL", DEFAULT_TLC_SOURCE_BASE_URL)

    service, year_month, source_url, force = validate_event(event)
    source_url = source_url or build_tlc_url(service, year_month, base_url)

    dynamodb = boto3.resource("dynamodb")
    s3_client = boto3.client("s3")
    manifest_table = dynamodb.Table(manifest_table_name)

    try:
        source_metadata = get_source_metadata(source_url)
    except HTTPError as exc:
        message = f"Could not read source metadata for {source_url}: HTTP {exc.code}"
        raise RuntimeError(message) from exc

    source_etag = source_metadata["etag"]
    existing = get_existing_manifest(manifest_table, service, year_month)

    if should_skip_existing(existing, source_etag, force=force):
        return {
            "status": "skipped",
            "reason": "unchanged_etag",
            "service": service,
            "year_month": year_month,
            "etag": source_etag,
            "bronze_s3_uri": existing.get("bronze_s3_uri"),
        }

    data_key, meta_key = bronze_keys(service, year_month)
    started_at = now_utc_iso()

    fetch_to_s3(source_url, bronze_bucket, data_key, s3_client)

    bronze_s3_uri = f"s3://{bronze_bucket}/{data_key}"
    meta_s3_uri = f"s3://{bronze_bucket}/{meta_key}"

    manifest_metadata = {
        "service": service,
        "year_month": year_month,
        "source_url": source_url,
        "etag": source_etag,
        "source_last_modified": source_metadata.get("last_modified"),
        "source_content_length": source_metadata.get("content_length"),
        "bronze_s3_uri": bronze_s3_uri,
        "fetched_at": started_at,
    }

    put_manifest_metadata(bronze_bucket, meta_key, manifest_metadata, s3_client)

    manifest_item = {
        "service": service,
        "year_month": year_month,
        "etag": source_etag,
        "status": "fetched",
        "source_url": source_url,
        "bronze_s3_uri": bronze_s3_uri,
        "meta_s3_uri": meta_s3_uri,
        "source_last_modified": source_metadata.get("last_modified", ""),
        "source_content_length": source_metadata.get("content_length", ""),
        "fetched_at": started_at,
        "updated_at": now_utc_iso(),
    }

    write_manifest_item(manifest_table, manifest_item, force=force)

    return {
        "status": "fetched",
        "service": service,
        "year_month": year_month,
        "etag": source_etag,
        "bronze_s3_uri": bronze_s3_uri,
        "meta_s3_uri": meta_s3_uri,
    }
