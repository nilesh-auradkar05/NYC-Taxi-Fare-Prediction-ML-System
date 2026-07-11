from __future__ import annotations

import os
from typing import Any

import boto3
from shared.tlc_manifest import (
    PIPELINE_ACTION_RESUME,
    PIPELINE_ACTION_SKIP,
    build_tlc_url,
    determine_pipeline_action,
    get_source_metadata,
    validate_event,
)

DEFAULT_TLC_SOURCE_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def get_existing_manifest(table: Any, service: str, year_month: str) -> dict[str, Any] | None:
    response = table.get_item(
        Key={"service": service, "year_month": year_month},
        ConsistentRead=True,
    )
    return response.get("Item")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    del context

    manifest_table_name = os.environ["MANIFEST_TABLE"]
    base_url = os.environ.get("TLC_SOURCE_BASE_URL", DEFAULT_TLC_SOURCE_BASE_URL)

    service, year_month, source_url, force = validate_event(event)
    source_url = source_url or build_tlc_url(service, year_month, base_url)

    dynamodb = boto3.resource("dynamodb")
    manifest_table = dynamodb.Table(manifest_table_name)

    source_metadata = get_source_metadata(source_url)
    existing = get_existing_manifest(manifest_table, service, year_month)
    pipeline_action = determine_pipeline_action(
        existing,
        source_metadata["etag"],
        force=force,
    )

    return {
        **event,
        "service": service,
        "year_month": year_month,
        "source_url": source_url,
        "etag": source_metadata["etag"],
        "source_last_modified": source_metadata.get("last_modified", ""),
        "source_content_length": source_metadata.get("content_length", ""),
        "pipeline_action": pipeline_action,
        # Retained for compatibility with existing execution inspection and tests.
        "is_new": pipeline_action != PIPELINE_ACTION_SKIP,
        "skip_reason": (
            "unchanged_etag_complete"
            if pipeline_action == PIPELINE_ACTION_SKIP
            else None
        ),
        "resume_reason": (
            "unchanged_etag_incomplete"
            if pipeline_action == PIPELINE_ACTION_RESUME
            else None
        ),
        "existing_manifest_status": existing.get("status") if existing else None,
        "existing_bronze_s3_uri": existing.get("bronze_s3_uri") if existing else None,
    }
