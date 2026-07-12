from __future__ import annotations

import tomllib
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


def test_t201_does_not_create_gold_models() -> None:
    forbidden_models = {
        "fct_trips_zone_day.sql",
        "mart_crz_panel.sql",
        "mart_forecast_train.sql",
        "mart_agent_metrics.sql",
    }
    present_sql_files = {path.name for path in (DBT_DIR / "models").rglob("*.sql")}

    assert present_sql_files.isdisjoint(forbidden_models)
    assert not present_sql_files


def test_makefile_and_ci_run_the_duckdb_compile_target() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "dbt compile" in makefile
    assert "--target $(DBT_TARGET)" in makefile
    assert "DBT_TARGET ?= ci" in makefile
    assert "DBT_DUCKDB_PATH ?= fixtures/nyc_taxi_ci.duckdb" in makefile
    assert "make dbt-compile" in workflow
    assert "AWS_ACCESS_KEY_ID" not in workflow
