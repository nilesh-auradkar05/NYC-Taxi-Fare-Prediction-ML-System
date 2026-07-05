from __future__ import annotations

from ingestion.reference.crz import (
    annotate_zone_flags,
    buffer_ring_zone_ids,
    crz_zone_ids,
    is_buffer_ring_zone,
    is_crz_zone,
    load_buffer_ring_zones,
    load_crz_zones,
)

EXPECTED_TLC_ZONES = {
    4: ("Manhattan", "Alphabet City"),
    12: ("Manhattan", "Battery Park"),
    13: ("Manhattan", "Battery Park City"),
    43: ("Manhattan", "Central Park"),
    45: ("Manhattan", "Chinatown"),
    48: ("Manhattan", "Clinton East"),
    50: ("Manhattan", "Clinton West"),
    68: ("Manhattan", "East Chelsea"),
    79: ("Manhattan", "East Village"),
    87: ("Manhattan", "Financial District North"),
    88: ("Manhattan", "Financial District South"),
    90: ("Manhattan", "Flatiron"),
    100: ("Manhattan", "Garment District"),
    107: ("Manhattan", "Gramercy"),
    113: ("Manhattan", "Greenwich Village North"),
    114: ("Manhattan", "Greenwich Village South"),
    125: ("Manhattan", "Hudson Sq"),
    137: ("Manhattan", "Kips Bay"),
    140: ("Manhattan", "Lenox Hill East"),
    141: ("Manhattan", "Lenox Hill West"),
    142: ("Manhattan", "Lincoln Square East"),
    143: ("Manhattan", "Lincoln Square West"),
    144: ("Manhattan", "Little Italy/NoLiTa"),
    148: ("Manhattan", "Lower East Side"),
    158: ("Manhattan", "Meatpacking/West Village West"),
    161: ("Manhattan", "Midtown Center"),
    162: ("Manhattan", "Midtown East"),
    163: ("Manhattan", "Midtown North"),
    164: ("Manhattan", "Midtown South"),
    170: ("Manhattan", "Murray Hill"),
    186: ("Manhattan", "Penn Station/Madison Sq West"),
    209: ("Manhattan", "Seaport"),
    211: ("Manhattan", "SoHo"),
    224: ("Manhattan", "Stuy Town/Peter Cooper Village"),
    229: ("Manhattan", "Sutton Place/Turtle Bay North"),
    230: ("Manhattan", "Times Sq/Theatre District"),
    231: ("Manhattan", "TriBeCa/Civic Center"),
    232: ("Manhattan", "Two Bridges/Seward Park"),
    233: ("Manhattan", "UN/Turtle Bay South"),
    234: ("Manhattan", "Union Sq"),
    237: ("Manhattan", "Upper East Side South"),
    239: ("Manhattan", "Upper West Side South"),
    246: ("Manhattan", "West Chelsea/Hudson Yards"),
    249: ("Manhattan", "West Village"),
    261: ("Manhattan", "World Trade Center"),
}

# These are not left to geometry inference. They are the exact zones that must be
# eyeballed against the official TLC Manhattan map and MTA CRZ boundary.
BOUNDARY_ASSERTIONS = {
    43: "buffer",
    48: "crz",
    50: "crz",
    140: "buffer",
    141: "buffer",
    142: "buffer",
    143: "buffer",
    158: "crz",
    163: "crz",
    224: "crz",
    229: "crz",
    230: "crz",
    233: "crz",
    237: "buffer",
    239: "buffer",
    246: "crz",
}


def test_u02_seed_files_load_and_have_expected_shape():
    crz_rows = load_crz_zones()
    buffer_rows = load_buffer_ring_zones()

    assert crz_rows
    assert buffer_rows

    for row in (*crz_rows, *buffer_rows):
        assert row.location_id > 0
        assert row.borough == "Manhattan"
        assert row.zone
        assert row.source == "MTA_CRZ_TLC_zone_overlay"
        assert row.review_status in {"draft_needs_human_review", "human_verified"}
        assert row.notes


def test_u02_all_seeded_location_ids_match_expected_tlc_zone_names():
    rows = [*load_crz_zones(), *load_buffer_ring_zones()]

    for row in rows:
        assert row.location_id in EXPECTED_TLC_ZONES
        expected_borough, expected_zone = EXPECTED_TLC_ZONES[row.location_id]
        assert row.borough == expected_borough
        assert row.zone == expected_zone


def test_u02_crz_and_buffer_ring_are_disjoint():
    overlap = crz_zone_ids() & buffer_ring_zone_ids()

    assert overlap == frozenset()


def test_u02_boundary_zones_are_asserted_explicitly():
    for location_id, expected_bucket in BOUNDARY_ASSERTIONS.items():
        if expected_bucket == "crz":
            assert is_crz_zone(location_id), f"{location_id} should be in CRZ seed"
            assert not is_buffer_ring_zone(location_id), f"{location_id} should not be in buffer seed"
        elif expected_bucket == "buffer":
            assert is_buffer_ring_zone(location_id), f"{location_id} should be in buffer seed"
            assert not is_crz_zone(location_id), f"{location_id} should not be in CRZ seed"
        else:
            raise AssertionError(f"Unknown expected bucket: {expected_bucket}")


def test_u02_representative_flags_match_hand_reviewed_draft():
    assert is_crz_zone(100)
    assert is_crz_zone("230")
    assert is_crz_zone(261)

    assert is_buffer_ring_zone(142)
    assert is_buffer_ring_zone("237")

    assert not is_crz_zone(142)
    assert not is_buffer_ring_zone(100)
    assert not is_crz_zone(None)
    assert not is_crz_zone("not-a-zone")


def test_u02_annotate_zone_flags_is_framework_neutral():
    rows = [
        {"location_id": 100, "label": "garment"},
        {"location_id": 142, "label": "lincoln-square-east"},
        {"location_id": 999, "label": "unknown"},
    ]

    annotated = annotate_zone_flags(rows)

    assert annotated == [
        {
            "location_id": 100,
            "label": "garment",
            "is_crz": True,
            "is_buffer_ring": False,
        },
        {
            "location_id": 142,
            "label": "lincoln-square-east",
            "is_crz": False,
            "is_buffer_ring": True,
        },
        {
            "location_id": 999,
            "label": "unknown",
            "is_crz": False,
            "is_buffer_ring": False,
        },
    ]
