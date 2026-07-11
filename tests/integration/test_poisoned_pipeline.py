from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    ProxyConnectionError,
    ReadTimeoutError,
    SSLError,
)

TERMINAL_EXECUTION_STATUSES = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED", "PENDING_REDRIVE"}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
YEAR_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
TEST_TICKET = "T-107"
DEFAULT_SERVICE = "yellow"
DEFAULT_YEAR_MONTH = "2099-01"
AWS_PREFLIGHT_SKIP_CODES = {
    "AccessDenied",
    "AccessDeniedException",
    "ExpiredToken",
    "InvalidClientTokenId",
    "UnauthorizedOperation",
    "UnrecognizedClientException",
}
AWS_PREFLIGHT_UNAVAILABLE_EXCEPTIONS = (
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    ProxyConnectionError,
    ReadTimeoutError,
    SSLError,
)
AWS_PREFLIGHT_EXCEPTIONS = (ClientError, *AWS_PREFLIGHT_UNAVAILABLE_EXCEPTIONS)


@dataclass(frozen=True)
class IntegrationConfig:
    state_machine_arn: str
    manifest_table_name: str
    bronze_bucket: str
    silver_database: str
    athena_workgroup: str
    service: str
    year_month: str
    region_name: str
    poll_seconds: float
    timeout_seconds: int
    keep_artifacts: bool


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _terraform_outputs() -> dict[str, Any]:
    completed = subprocess.run(
        ["terraform", "-chdir=infra", "output", "-json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return {name: record["value"] for name, record in payload.items()}


def _map_value(values: dict[str, Any], output_name: str, key: str) -> str:
    mapping = values.get(output_name)
    if not isinstance(mapping, dict) or key not in mapping:
        raise RuntimeError(f"Terraform output {output_name!r} does not contain key {key!r}")
    return str(mapping[key])


def _env_or_output(env_name: str, outputs: dict[str, Any], output_name: str) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    if output_name not in outputs:
        raise RuntimeError(f"Missing Terraform output {output_name!r}; set {env_name} explicitly")
    return str(outputs[output_name])


def resolve_config() -> IntegrationConfig:
    output_env_names = {
        "T107_STATE_MACHINE_ARN",
        "T107_MANIFEST_TABLE",
        "T107_BRONZE_BUCKET",
        "T107_SILVER_DATABASE",
        "T107_ATHENA_WORKGROUP",
    }
    outputs = {} if all(os.getenv(name) for name in output_env_names) else _terraform_outputs()

    state_machine_arn = _env_or_output(
        "T107_STATE_MACHINE_ARN",
        outputs,
        "monthly_ingestion_state_machine_arn",
    )
    manifest_table_name = _env_or_output(
        "T107_MANIFEST_TABLE",
        outputs,
        "manifest_table_name",
    )
    bronze_bucket = os.getenv("T107_BRONZE_BUCKET") or _map_value(outputs, "s3_bucket_names", "bronze")
    silver_database = os.getenv("T107_SILVER_DATABASE") or _map_value(
        outputs,
        "glue_database_names",
        "silver",
    )
    athena_workgroup = os.getenv("T107_ATHENA_WORKGROUP") or _map_value(
        outputs,
        "athena_workgroups",
        "engineering",
    )

    service = os.getenv("T107_SERVICE", DEFAULT_SERVICE).strip().lower()
    year_month = os.getenv("T107_YEAR_MONTH", DEFAULT_YEAR_MONTH).strip()
    if service not in {"yellow", "hvfhv"}:
        raise RuntimeError(f"Unsupported T107_SERVICE={service!r}")
    if not YEAR_MONTH_RE.fullmatch(year_month):
        raise RuntimeError(f"Invalid T107_YEAR_MONTH={year_month!r}; expected YYYY-MM")
    if int(year_month[:4]) < 2090 and not _truthy(os.getenv("T107_ALLOW_REAL_MONTH")):
        raise RuntimeError(
            "T107_YEAR_MONTH must use the reserved 2090+ integration-test range. "
            "Set T107_ALLOW_REAL_MONTH=1 only after manually confirming the partition is disposable."
        )
    if "-dev-" not in state_machine_arn and not _truthy(os.getenv("T107_ALLOW_NON_DEV")):
        raise RuntimeError(
            "Refusing to run T-107 against a state machine without '-dev-' in its ARN. "
            "Set T107_ALLOW_NON_DEV=1 only for an explicitly reviewed non-production environment."
        )
    for identifier, label in ((silver_database, "silver database"),):
        if not IDENTIFIER_RE.fullmatch(identifier):
            raise RuntimeError(f"Unsafe {label} identifier: {identifier!r}")

    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region_name:
        arn_parts = state_machine_arn.split(":")
        if len(arn_parts) > 3:
            region_name = arn_parts[3]
    if not region_name:
        region_name = boto3.Session().region_name
    if not region_name:
        raise RuntimeError("AWS region is unavailable; set AWS_REGION")

    return IntegrationConfig(
        state_machine_arn=state_machine_arn,
        manifest_table_name=manifest_table_name,
        bronze_bucket=bronze_bucket,
        silver_database=silver_database,
        athena_workgroup=athena_workgroup,
        service=service,
        year_month=year_month,
        region_name=region_name,
        poll_seconds=float(os.getenv("T107_POLL_SECONDS", "10")),
        timeout_seconds=int(os.getenv("T107_TIMEOUT_SECONDS", "3600")),
        keep_artifacts=_truthy(os.getenv("T107_KEEP_ARTIFACTS")),
    )


def poisoned_rows(year_month: str) -> list[dict[str, str | None]]:
    year, month = map(int, year_month.split("-"))
    if month == 1:
        prior_year, prior_month = year - 1, 12
    else:
        prior_year, prior_month = year, month - 1
    current = f"{year:04d}-{month:02d}"
    prior = f"{prior_year:04d}-{prior_month:02d}"

    common = {
        "VendorID": "1",
        "passenger_count": "1",
        "PULocationID": "161",
        "DOLocationID": "162",
        "trip_distance": "1.5",
        "extra": "0.0",
        "mta_tax": "0.5",
        "tip_amount": "1.0",
        "tolls_amount": "0.0",
        "improvement_surcharge": "1.0",
        "total_amount": "12.5",
        "congestion_surcharge": "0.0",
        "Airport_fee": "0.0",
        "cbd_congestion_fee": "0.0",
    }
    return [
        {
            **common,
            "tpep_pickup_datetime": None,
            "tpep_dropoff_datetime": f"{current}-15 10:20:00",
            "fare_amount": "10.0",
        },
        {
            **common,
            "tpep_pickup_datetime": f"{current}-15 10:00:00",
            "tpep_dropoff_datetime": f"{current}-15 10:20:00",
            "fare_amount": "-5.0",
        },
        {
            **common,
            "tpep_pickup_datetime": f"{prior}-28 23:50:00",
            "tpep_dropoff_datetime": f"{current}-01 00:10:00",
            "fare_amount": "10.0",
        },
    ]


def write_poisoned_parquet(output_dir: Path, year_month: str) -> Path:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StringType, StructField, StructType

    rows = poisoned_rows(year_month)
    schema = StructType([StructField(name, StringType(), True) for name in rows[0]])
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("t107-poisoned-fixture")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    try:
        spark.createDataFrame(rows, schema=schema).coalesce(1).write.mode("overwrite").parquet(
            str(output_dir)
        )
    finally:
        spark.stop()

    parts = sorted(output_dir.glob("part-*.parquet"))
    if len(parts) != 1:
        raise RuntimeError(f"Expected one generated Parquet part, found {len(parts)}")
    return parts[0]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class AthenaRunner:
    def __init__(self, client: Any, database: str, workgroup: str) -> None:
        self.client = client
        self.database = database
        self.workgroup = workgroup

    def execute(self, query: str, timeout_seconds: int = 300) -> str:
        response = self.client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": self.database},
            WorkGroup=self.workgroup,
        )
        execution_id = response["QueryExecutionId"]
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            execution = self.client.get_query_execution(QueryExecutionId=execution_id)["QueryExecution"]
            state = execution["Status"]["State"]
            if state == "SUCCEEDED":
                return execution_id
            if state in {"FAILED", "CANCELLED"}:
                reason = execution["Status"].get("StateChangeReason", "")
                raise RuntimeError(f"Athena query {state}: {reason}\n{query}")
            time.sleep(2)
        raise TimeoutError(f"Athena query did not finish within {timeout_seconds}s: {query}")

    def rows(self, query: str) -> list[list[str | None]]:
        execution_id = self.execute(query)
        paginator = self.client.get_paginator("get_query_results")
        raw_rows: list[list[str | None]] = []
        for page in paginator.paginate(QueryExecutionId=execution_id):
            for row in page["ResultSet"]["Rows"]:
                raw_rows.append([cell.get("VarCharValue") for cell in row.get("Data", [])])
        return raw_rows[1:] if raw_rows else []

    def scalar_int(self, query: str) -> int:
        rows = self.rows(query)
        if len(rows) != 1 or not rows[0] or rows[0][0] is None:
            raise RuntimeError(f"Expected one integer result from Athena, received {rows!r}")
        return int(rows[0][0])


def _validate_deployed_state_machine(stepfunctions: Any, state_machine_arn: str) -> None:
    response = stepfunctions.describe_state_machine(stateMachineArn=state_machine_arn)
    definition = json.loads(response["definition"])
    states = definition.get("States", {})
    required_states = {
        "SelectPipelineAction",
        "MarkManifestProcessing",
        "DQGate",
        "MarkManifestFailedAfterDQ",
        "NotifyDQFailure",
        "DQFailed",
    }
    missing = sorted(required_states - set(states))
    if missing:
        raise RuntimeError(
            "T-107 requires the corrected T-103 manifest/DQ lifecycle to be deployed; "
            f"missing states: {missing}"
        )

    dq_catches = states["DQGate"].get("Catch", [])
    if not any(catch.get("Next") == "MarkManifestFailedAfterDQ" for catch in dq_catches):
        raise RuntimeError("Deployed DQGate does not route failures to MarkManifestFailedAfterDQ")
    if states["MarkManifestFailedAfterDQ"].get("Next") != "NotifyDQFailure":
        raise RuntimeError("Deployed manifest failure state does not route to NotifyDQFailure")
    notify_state = states["NotifyDQFailure"]
    if notify_state.get("Resource") != "arn:aws:states:::sns:publish":
        raise RuntimeError("Deployed NotifyDQFailure is not using the Step Functions SNS integration")
    if notify_state.get("Next") != "DQFailed":
        raise RuntimeError("Deployed NotifyDQFailure does not route to DQFailed")
    if states["DQFailed"].get("Error") != "DataQualityGateFailed":
        raise RuntimeError("Deployed DQFailed state does not expose DataQualityGateFailed")


def _skip_if_aws_preflight_unavailable(action: str, exc: BaseException) -> None:
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in AWS_PREFLIGHT_SKIP_CODES:
            pytest.skip(f"{TEST_TICKET} integration skipped: AWS credentials cannot {action} ({code})")
        return

    if isinstance(exc, AWS_PREFLIGHT_UNAVAILABLE_EXCEPTIONS):
        pytest.skip(f"{TEST_TICKET} integration skipped: AWS {action} preflight unavailable: {exc}")


def _history_events(stepfunctions: Any, execution_arn: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"executionArn": execution_arn, "includeExecutionData": True}
        if token:
            kwargs["nextToken"] = token
        response = stepfunctions.get_execution_history(**kwargs)
        events.extend(response["events"])
        token = response.get("nextToken")
        if not token:
            return events


def _entered_state_names(events: list[dict[str, Any]]) -> list[str]:
    return [
        event["stateEnteredEventDetails"]["name"]
        for event in events
        if "stateEnteredEventDetails" in event
    ]


def _assert_ordered_subsequence(actual: list[str], expected: list[str]) -> None:
    cursor = 0
    for value in actual:
        if cursor < len(expected) and value == expected[cursor]:
            cursor += 1
    if cursor != len(expected):
        raise AssertionError(f"Expected ordered state path {expected!r}; actual states were {actual!r}")


def _wait_for_execution(
    stepfunctions: Any,
    execution_arn: str,
    poll_seconds: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        execution = stepfunctions.describe_execution(executionArn=execution_arn)
        if execution["status"] in TERMINAL_EXECUTION_STATUSES:
            return execution
        time.sleep(poll_seconds)
    raise TimeoutError(f"Step Functions execution exceeded {timeout_seconds}s: {execution_arn}")


def _execution_name(service: str, year_month: str, token: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"t107-poisoned-{service}-{year_month}-{timestamp}-{token[:8]}"
    return name[:80]


def _safe_cleanup(
    *,
    config: IntegrationConfig,
    token: str,
    object_key: str,
    etag: str,
    s3: Any,
    manifest_table: Any,
    athena: AthenaRunner,
) -> None:
    cleanup_errors: list[str] = []
    source_pattern = f"%t107-poisoned-{token}%"
    for table in ("trips", "trips_quarantine"):
        query = (
            f'DELETE FROM "{config.silver_database}"."{table}" '
            f"WHERE service = {_sql_string(config.service)} "
            f"AND year_month = {_sql_string(config.year_month)} "
            f"AND source_file LIKE {_sql_string(source_pattern)}"
        )
        try:
            athena.execute(query)
        except Exception as exc:  # pragma: no cover - cleanup diagnostics only
            cleanup_errors.append(f"Athena cleanup for {table}: {exc}")

    try:
        current = manifest_table.get_item(
            Key={"service": config.service, "year_month": config.year_month},
            ConsistentRead=True,
        ).get("Item")
        if current and current.get("etag") == etag and current.get("test_run_id") == token:
            manifest_table.delete_item(Key={"service": config.service, "year_month": config.year_month})
    except Exception as exc:  # pragma: no cover - cleanup diagnostics only
        cleanup_errors.append(f"DynamoDB cleanup: {exc}")

    try:
        s3.delete_object(Bucket=config.bronze_bucket, Key=object_key)
    except Exception as exc:  # pragma: no cover - cleanup diagnostics only
        cleanup_errors.append(f"S3 cleanup: {exc}")

    if cleanup_errors:
        warnings.warn("T-107 cleanup was incomplete:\n" + "\n".join(cleanup_errors), stacklevel=2)


@pytest.mark.integration
def test_poisoned_file_halts_pipeline_quarantines_rows_and_publishes_sns(tmp_path: Path) -> None:
    config = resolve_config()
    session = boto3.Session(region_name=config.region_name)
    s3 = session.client("s3")
    stepfunctions = session.client("stepfunctions")
    athena_client = session.client("athena")
    manifest_table = session.resource("dynamodb").Table(config.manifest_table_name)
    athena = AthenaRunner(athena_client, config.silver_database, config.athena_workgroup)

    try:
        _validate_deployed_state_machine(stepfunctions, config.state_machine_arn)
    except AWS_PREFLIGHT_EXCEPTIONS as exc:
        _skip_if_aws_preflight_unavailable("describe the dev state machine", exc)
        raise

    token = uuid.uuid4().hex
    prefix = f"bronze/service={config.service}/year_month={config.year_month}/"
    object_key = f"{prefix}t107-poisoned-{token}.parquet"
    source_pattern = f"%t107-poisoned-{token}%"
    etag = ""

    existing_objects = s3.list_objects_v2(Bucket=config.bronze_bucket, Prefix=prefix).get("Contents", [])
    unexpected_objects = [
        record["Key"]
        for record in existing_objects
        if record["Key"].endswith(".parquet") and "t107-poisoned-" not in record["Key"]
    ]
    if unexpected_objects:
        raise RuntimeError(
            f"Reserved T-107 partition {config.service}/{config.year_month} contains non-test Parquet: "
            f"{unexpected_objects}"
        )
    for record in existing_objects:
        if "t107-poisoned-" in record["Key"]:
            s3.delete_object(Bucket=config.bronze_bucket, Key=record["Key"])

    existing_manifest = manifest_table.get_item(
        Key={"service": config.service, "year_month": config.year_month},
        ConsistentRead=True,
    ).get("Item")
    if existing_manifest and existing_manifest.get("test_ticket") != TEST_TICKET:
        raise RuntimeError(
            f"Reserved T-107 manifest key already belongs to non-test data: {existing_manifest}"
        )
    if existing_manifest:
        manifest_table.delete_item(Key={"service": config.service, "year_month": config.year_month})

    stale_pattern = "%t107-poisoned-%"
    for table in ("trips", "trips_quarantine"):
        athena.execute(
            f'DELETE FROM "{config.silver_database}"."{table}" '
            f"WHERE service = {_sql_string(config.service)} "
            f"AND year_month = {_sql_string(config.year_month)} "
            f"AND source_file LIKE {_sql_string(stale_pattern)}"
        )

    non_test_current_rows = athena.scalar_int(
        f'SELECT count(*) FROM "{config.silver_database}"."trips" '
        f"WHERE service = {_sql_string(config.service)} "
        f"AND year_month = {_sql_string(config.year_month)}"
    )
    if non_test_current_rows:
        raise RuntimeError(
            f"Reserved T-107 partition {config.service}/{config.year_month} already contains "
            f"{non_test_current_rows} silver rows"
        )

    fixture_dir = tmp_path / "poisoned-parquet"
    parquet_file = write_poisoned_parquet(fixture_dir, config.year_month)
    s3.upload_file(str(parquet_file), config.bronze_bucket, object_key)
    head = s3.head_object(Bucket=config.bronze_bucket, Key=object_key)
    etag = str(head["ETag"]).strip('"')
    source_url = s3.generate_presigned_url(
        "head_object",
        Params={"Bucket": config.bronze_bucket, "Key": object_key},
        ExpiresIn=3600,
        HttpMethod="HEAD",
    )
    now = datetime.now(UTC).isoformat()
    manifest_table.put_item(
        Item={
            "service": config.service,
            "year_month": config.year_month,
            "etag": etag,
            "status": "fetched",
            "source_url": source_url,
            "bronze_s3_uri": f"s3://{config.bronze_bucket}/{object_key}",
            "source_content_length": str(head["ContentLength"]),
            "fetched_at": now,
            "updated_at": now,
            "test_ticket": TEST_TICKET,
            "test_run_id": token,
        }
    )

    execution_arn = ""
    try:
        execution = stepfunctions.start_execution(
            stateMachineArn=config.state_machine_arn,
            name=_execution_name(config.service, config.year_month, token),
            input=json.dumps(
                {
                    "service": config.service,
                    "year_month": config.year_month,
                    "source_url": source_url,
                    "trigger": "integration-test",
                    "test_ticket": TEST_TICKET,
                    "test_run_id": token,
                },
                separators=(",", ":"),
            ),
        )
        execution_arn = execution["executionArn"]
        terminal = _wait_for_execution(
            stepfunctions,
            execution_arn,
            config.poll_seconds,
            config.timeout_seconds,
        )
        assert terminal["status"] == "FAILED", terminal
        assert terminal.get("error") == "DataQualityGateFailed", terminal

        events = _history_events(stepfunctions, execution_arn)
        states = _entered_state_names(events)
        expected_failure_path = [
            "CheckManifest",
            "SelectPipelineAction",
            "MarkManifestProcessing",
            "GlueConform",
            "DQGate",
            "MarkManifestFailedAfterDQ",
            "NotifyDQFailure",
            "DQFailed",
        ]
        _assert_ordered_subsequence(states, expected_failure_path)
        assert "EmitSilverUpdatedEvent" not in states
        assert "MarkManifestComplete" not in states
        assert "Done" not in states

        manifest = manifest_table.get_item(
            Key={"service": config.service, "year_month": config.year_month},
            ConsistentRead=True,
        )["Item"]
        assert manifest["status"] == "failed"
        assert manifest["failure_stage"] == "DQGate"
        assert manifest["execution_arn"] == execution_arn
        last_error = manifest.get("last_error", "")
        for metric_name in (
            "dq_02_required_null_rates_ok",
            "dq_03_domain_values_ok",
            "dq_04_timestamp_sanity_ok",
        ):
            assert metric_name in last_error, last_error

        deadline = time.monotonic() + 180
        reason_rows: list[list[str | None]] = []
        while time.monotonic() < deadline:
            reason_rows = athena.rows(
                f'SELECT reason_code, count(*) FROM "{config.silver_database}"."trips_quarantine" '
                f"WHERE service = {_sql_string(config.service)} "
                f"AND year_month = {_sql_string(config.year_month)} "
                f"AND source_file LIKE {_sql_string(source_pattern)} "
                "GROUP BY reason_code"
            )
            if reason_rows:
                break
            time.sleep(5)
        assert reason_rows, "No T-107 quarantine rows were visible in Athena"
        combined_reasons = ",".join(str(row[0]) for row in reason_rows if row and row[0])
        for expected_reason in ("BAD_PICKUP_TS", "BAD_FARE", "MONTH_MISMATCH"):
            assert expected_reason in combined_reasons, combined_reasons
    finally:
        if config.keep_artifacts:
            print(
                json.dumps(
                    {
                        "t107_keep_artifacts": True,
                        "execution_arn": execution_arn,
                        "service": config.service,
                        "year_month": config.year_month,
                        "s3_object": f"s3://{config.bronze_bucket}/{object_key}",
                        "test_run_id": token,
                    },
                    indent=2,
                )
            )
        elif etag:
            _safe_cleanup(
                config=config,
                token=token,
                object_key=object_key,
                etag=etag,
                s3=s3,
                manifest_table=manifest_table,
                athena=athena,
            )
        shutil.rmtree(fixture_dir, ignore_errors=True)
