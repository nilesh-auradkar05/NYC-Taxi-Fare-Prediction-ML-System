from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DBT_DIR = ROOT / "dbt"


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_dbt_project_has_expected_profile_and_paths() -> None:
    project = read_yaml(DBT_DIR / "dbt_project.yml")

    assert project["name"] == "nyc_taxi"
    assert project["profile"] == "nyc_taxi"
    assert project["model-paths"] == ["models"]
    assert project["analysis-paths"] == ["analyses"]


def test_profiles_keep_ci_aws_free_and_athena_environment_driven() -> None:
    profiles_text = (DBT_DIR / "profiles.yml").read_text(encoding="utf-8")
    profile = read_yaml(DBT_DIR / "profiles.yml")["nyc_taxi"]
    ci_target = profile["outputs"]["ci"]
    athena_target = profile["outputs"]["athena"]

    assert ci_target["type"] == "duckdb"
    assert "DBT_DUCKDB_PATH" in ci_target["path"]
    assert "AWS_" not in str(ci_target)

    assert athena_target["type"] == "athena"
    for variable in (
        "DBT_ATHENA_S3_STAGING_DIR",
        "DBT_ATHENA_S3_DATA_DIR",
        "DBT_ATHENA_REGION",
        "DBT_ATHENA_DATABASE",
        "DBT_ATHENA_SCHEMA",
        "DBT_ATHENA_WORK_GROUP",
    ):
        assert f"env_var('{variable}'" in profiles_text

    assert "aws_access_key_id:" not in profiles_text
    assert "aws_secret_access_key:" not in profiles_text
    assert "aws_profile_name:" not in profiles_text


def test_silver_sources_match_existing_glue_relations() -> None:
    source_document = read_yaml(DBT_DIR / "models" / "sources" / "silver.yml")
    source = source_document["sources"][0]

    assert source["name"] == "silver"
    assert "nyc_taxi_silver" in source["schema"]
    assert source["database"] == "{{ target.database }}"
    assert {table["name"] for table in source["tables"]} == {
        "trips",
        "trips_quarantine",
    }


def test_t202_adds_only_fct_trips_zone_day() -> None:
    forbidden_models = {
        "mart_crz_panel.sql",
        "mart_forecast_train.sql",
        "mart_agent_metrics.sql",
    }
    present_sql_files = {path.name for path in (DBT_DIR / "models").rglob("*.sql")}

    assert "fct_trips_zone_day.sql" in present_sql_files
    assert present_sql_files.isdisjoint(forbidden_models)
    assert present_sql_files == {"fct_trips_zone_day.sql"}


def test_fct_trips_zone_day_contract_and_tests() -> None:
    properties = read_yaml(
        DBT_DIR / "models" / "gold" / "fct_trips_zone_day.yml"
    )
    model = properties["models"][0]
    columns = {column["name"]: column for column in model["columns"]}

    assert model["name"] == "fct_trips_zone_day"
    assert model["config"]["contract"]["enforced"] is True
    assert {
        "pickup_date",
        "pu_zone_id",
        "service",
        "trips",
        "total_fare",
        "avg_fare_ex_surcharge",
        "tip_rate",
        "avg_mph",
        "cbd_fee_revenue",
    } == set(columns)

    assert "not_null" in columns["pickup_date"]["data_tests"]
    assert "not_null" in columns["pu_zone_id"]["data_tests"]
    assert "not_null" in columns["service"]["data_tests"]
    accepted_values = columns["service"]["data_tests"][1]["accepted_values"]
    assert accepted_values["arguments"]["values"] == ["yellow", "hvfhv"]

    unique_test = model["data_tests"][0]["unique"]
    unique_expression = unique_test["arguments"]["column_name"]
    assert "pickup_date" in unique_expression
    assert "pu_zone_id" in unique_expression
    assert "service" in unique_expression


def test_fct_trips_zone_day_uses_actual_silver_columns() -> None:
    model_sql = (
        DBT_DIR / "models" / "gold" / "fct_trips_zone_day.sql"
    ).read_text(encoding="utf-8")
    fixture_sql = (DBT_DIR / "fixtures" / "fixture.sql").read_text(
        encoding="utf-8"
    )

    for column in (
        "pickup_ts",
        "pu_zone_id",
        "service",
        "total_amount",
        "fare_ex_surcharge",
        "tip_rate",
        "avg_mph",
        "cbd_congestion_fee",
    ):
        assert column in model_sql
        assert column in fixture_sql

    assert "source('silver', 'trips')" in model_sql
    for formula in (
        "count(*)",
        "sum(total_amount)",
        "avg(fare_ex_surcharge)",
        "avg(tip_rate)",
        "avg(avg_mph)",
        "sum(cbd_congestion_fee)",
    ):
        assert formula in model_sql

    assert "pu_ts" not in fixture_sql
    assert "distance_mi" not in fixture_sql


def test_makefile_and_ci_run_the_duckdb_compile_target() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "dbt compile" in makefile
    assert "--target $(DBT_TARGET)" in makefile
    assert "DBT_TARGET ?= ci" in makefile
    assert "DBT_DUCKDB_PATH ?= dbt/fixtures/nyc_taxi_ci.duckdb" in makefile
    assert "make dbt-compile" in workflow
    assert "AWS_ACCESS_KEY_ID" not in workflow
