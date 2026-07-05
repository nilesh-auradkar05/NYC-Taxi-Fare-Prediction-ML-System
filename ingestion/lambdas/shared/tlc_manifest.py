from __future__ import annotations

import re
from typing import Any
from urllib.request import Request, urlopen

YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")

SERVICE_FILE_PREFIX = {
    "yellow": "yellow_tripdata",
    "hvfhv": "fhvhv_tripdata",
}

DEFAULT_TLC_SOURCE_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


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

    return existing_item.get("etag") == source_etag and existing_item.get("status") == "fetched"


def bronze_keys(service: str, year_month: str) -> tuple[str, str]:
    prefix = SERVICE_FILE_PREFIX[service]
    data_key = f"bronze/service={service}/year_month={year_month}/{prefix}_{year_month}.parquet"
    meta_key = f"bronze/service={service}/year_month={year_month}/_meta/manifest.json"
    return data_key, meta_key
