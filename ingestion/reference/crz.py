"""CRZ and buffer-ring zone reference helpers for T-102.

The CSVs are drafted from official TLC taxi-zone IDs and the MTA CRZ definition.
They are intentionally small committed reference seeds, not downloaded raw
shapefiles. Boundary-zone correctness is human-reviewed in T-102.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS_DIR = REPO_ROOT / "seeds"
CRZ_ZONES_PATH = SEEDS_DIR / "crz_zones.csv"
BUFFER_RING_PATH = SEEDS_DIR / "buffer_ring.csv"

REQUIRED_COLUMNS = {
    "location_id",
    "borough",
    "zone",
    "source",
    "review_status",
    "notes",
}

ALLOWED_REVIEW_STATUSES = {
    "draft_needs_human_review",
    "human_verified",
}


@dataclass(frozen=True, slots=True)
class ZoneSeed:
    location_id: int
    borough: str
    zone: str
    source: str
    review_status: str
    notes: str


def _read_seed_csv(path: Path) -> tuple[ZoneSeed, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Missing zone seed file: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        rows: list[ZoneSeed] = []
        seen_ids: set[int] = set()

        for row_number, row in enumerate(reader, start=2):
            raw_location_id = (row.get("location_id") or "").strip()
            if not raw_location_id.isdigit():
                raise ValueError(f"{path}:{row_number} invalid location_id={raw_location_id!r}")

            location_id = int(raw_location_id)
            if location_id in seen_ids:
                raise ValueError(f"{path}:{row_number} duplicate location_id={location_id}")
            seen_ids.add(location_id)

            review_status = (row.get("review_status") or "").strip()
            if review_status not in ALLOWED_REVIEW_STATUSES:
                raise ValueError(
                    f"{path}:{row_number} invalid review_status={review_status!r}; "
                    f"allowed={sorted(ALLOWED_REVIEW_STATUSES)}"
                )

            rows.append(
                ZoneSeed(
                    location_id=location_id,
                    borough=(row.get("borough") or "").strip(),
                    zone=(row.get("zone") or "").strip(),
                    source=(row.get("source") or "").strip(),
                    review_status=review_status,
                    notes=(row.get("notes") or "").strip(),
                )
            )

    return tuple(rows)


@lru_cache(maxsize=1)
def load_crz_zones() -> tuple[ZoneSeed, ...]:
    return _read_seed_csv(CRZ_ZONES_PATH)


@lru_cache(maxsize=1)
def load_buffer_ring_zones() -> tuple[ZoneSeed, ...]:
    return _read_seed_csv(BUFFER_RING_PATH)


@lru_cache(maxsize=1)
def crz_zone_ids() -> frozenset[int]:
    return frozenset(seed.location_id for seed in load_crz_zones())


@lru_cache(maxsize=1)
def buffer_ring_zone_ids() -> frozenset[int]:
    return frozenset(seed.location_id for seed in load_buffer_ring_zones())


def is_crz_zone(location_id: int | str | None) -> bool:
    parsed = _parse_location_id(location_id)
    return parsed in crz_zone_ids()


def is_buffer_ring_zone(location_id: int | str | None) -> bool:
    parsed = _parse_location_id(location_id)
    return parsed in buffer_ring_zone_ids()


def annotate_zone_flags(
    rows: Iterable[dict[str, object]],
    *,
    location_key: str = "location_id",
) -> list[dict[str, object]]:
    """Return copies of dict rows with is_crz and is_buffer_ring flags.

    This is intentionally framework-neutral. Spark/dbt integration comes later.
    """

    annotated: list[dict[str, object]] = []
    for row in rows:
        location_id = row.get(location_key)
        enriched = dict(row)
        enriched["is_crz"] = is_crz_zone(location_id)
        enriched["is_buffer_ring"] = is_buffer_ring_zone(location_id)
        annotated.append(enriched)
    return annotated


def _parse_location_id(value: int | str | None) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text.isdigit():
        return None

    return int(text)
