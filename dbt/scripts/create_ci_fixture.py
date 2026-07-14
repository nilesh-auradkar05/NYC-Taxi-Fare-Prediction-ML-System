from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--path",
        type=Path,
        default=Path("dbt/fixtures/nyc_taxi_ci.duckdb"),
        help="DuckDB file to create. Existing files are replaced.",
    )
    return parser.parse_args()

def create_fixture(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)

    fixture_sql = Path(__file__).resolve().parents[1] / "fixtures" / "fixture.sql"
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(fixture_sql.read_text(encoding="utf-8"))
    finally:
        connection.close()

if __name__ == "__main__":
    arguments = parse_args()
    create_fixture(arguments.path)
    print(f"Created dbt CI fixture: {arguments.path}")
