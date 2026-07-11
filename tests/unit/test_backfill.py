from __future__ import annotations

import csv
import json
import subprocess
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.data_pipeline.backfill import build_plan, execute_plan, main, month_range


class FakeStepFunctions:
    def __init__(self, terminal_statuses: list[str]) -> None:
        self.terminal_statuses = deque(terminal_statuses)
        self.events: list[tuple[str, str]] = []
        self.payloads: list[dict[str, Any]] = []
        self.current: dict[str, deque[str]] = {}

    def start_execution(self, **kwargs: Any) -> dict[str, Any]:
        name = kwargs["name"]
        arn = f"arn:aws:states:us-east-1:123456789012:execution:monthly:{name}"
        self.events.append(("start", name))
        self.payloads.append(json.loads(kwargs["input"]))
        self.current[arn] = deque(["RUNNING", self.terminal_statuses.popleft()])
        return {"executionArn": arn, "startDate": datetime(2026, 7, 9, tzinfo=UTC)}

    def describe_execution(self, **kwargs: Any) -> dict[str, Any]:
        arn = kwargs["executionArn"]
        status = self.current[arn].popleft()
        self.events.append(("describe", status))
        response: dict[str, Any] = {
            "executionArn": arn,
            "status": status,
            "startDate": datetime(2026, 7, 9, tzinfo=UTC),
        }
        if status != "RUNNING":
            response["stopDate"] = response["startDate"] + timedelta(seconds=12)
        if status != "SUCCEEDED":
            response.update(error="Glue.JobFailed", cause="synthetic failure")
        return response


def read_log(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_month_range_is_inclusive_across_year_boundary() -> None:
    assert month_range("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_plan_is_month_major_and_contains_backfill_metadata() -> None:
    plan = build_plan(
        "2023-01",
        "2023-02",
        run_id="20260709T010203Z-deadbeef",
        requested_at="2026-07-09T01:02:03+00:00",
    )

    assert [(item.service, item.year_month) for item in plan] == [
        ("yellow", "2023-01"),
        ("hvfhv", "2023-01"),
        ("yellow", "2023-02"),
        ("hvfhv", "2023-02"),
    ]
    assert plan[0].payload == {
        "service": "yellow",
        "year_month": "2023-01",
        "trigger": "backfill",
        "backfill": {
            "ticket": "T-106",
            "run_id": "20260709T010203Z-deadbeef",
            "requested_at": "2026-07-09T01:02:03+00:00",
        },
    }


def test_plan_can_select_only_one_service() -> None:
    yellow_plan = build_plan(
        "2025-06",
        "2025-10",
        run_id="20260709T010203Z-deadbeef",
        requested_at="2026-07-09T01:02:03+00:00",
        service="yellow",
    )
    hvfhv_plan = build_plan(
        "2025-06",
        "2025-10",
        run_id="20260709T010203Z-deadbeef",
        requested_at="2026-07-09T01:02:03+00:00",
        service="hvfhv",
    )

    assert len(yellow_plan) == 5
    assert {item.service for item in yellow_plan} == {"yellow"}
    assert len(hvfhv_plan) == 5
    assert {item.service for item in hvfhv_plan} == {"hvfhv"}


def test_default_cli_mode_is_dry_run_and_does_not_resolve_aws(monkeypatch, capsys) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("AWS/Terraform resolution must not run in dry-run mode")

    monkeypatch.setattr("src.data_pipeline.backfill.resolve_state_machine_arn", fail_if_called)

    exit_code = main(["--start-month", "2023-01", "--end-month", "2023-01"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "DRY RUN: 2 serial executions; no AWS calls will be made." in output
    assert "yellow" in output
    assert "hvfhv" in output


def test_cli_service_filter_changes_execution_count(monkeypatch, capsys) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("AWS/Terraform resolution must not run in dry-run mode")

    monkeypatch.setattr("src.data_pipeline.backfill.resolve_state_machine_arn", fail_if_called)

    exit_code = main(
        [
            "--service",
            "yellow",
            "--start-month",
            "2025-06",
            "--end-month",
            "2025-10",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "DRY RUN: 5 serial executions; no AWS calls will be made." in output
    assert "SERVICE SELECTION: yellow" in output
    assert "yellow" in output
    assert "hvfhv" not in output


def test_execute_plan_waits_for_each_execution_before_starting_next(tmp_path: Path) -> None:
    plan = build_plan(
        "2023-01",
        "2023-01",
        run_id="20260709T010203Z-deadbeef",
        requested_at="2026-07-09T01:02:03+00:00",
    )
    client = FakeStepFunctions(["SUCCEEDED", "SUCCEEDED"])
    log_file = tmp_path / "runs.csv"

    succeeded = execute_plan(
        plan,
        client=client,
        state_machine_arn="arn:aws:states:us-east-1:123456789012:stateMachine:monthly",
        log_file=log_file,
        poll_seconds=0,
        sleep=lambda _: None,
    )

    assert succeeded is True
    assert [event[0] for event in client.events] == [
        "start",
        "describe",
        "describe",
        "start",
        "describe",
        "describe",
    ]
    rows = read_log(log_file)
    assert [row["final_status"] for row in rows] == ["SUCCEEDED", "SUCCEEDED"]
    assert rows[0]["execution_arn"].startswith("arn:aws:states:")
    assert rows[0]["duration_seconds"] == "12.000"


def test_execute_plan_stops_on_first_failed_execution_and_logs_error(tmp_path: Path) -> None:
    plan = build_plan(
        "2023-01",
        "2023-02",
        run_id="20260709T010203Z-deadbeef",
        requested_at="2026-07-09T01:02:03+00:00",
    )
    client = FakeStepFunctions(["FAILED", "SUCCEEDED", "SUCCEEDED", "SUCCEEDED"])
    log_file = tmp_path / "runs.csv"

    succeeded = execute_plan(
        plan,
        client=client,
        state_machine_arn="arn:aws:states:us-east-1:123456789012:stateMachine:monthly",
        log_file=log_file,
        poll_seconds=0,
        sleep=lambda _: None,
    )

    assert succeeded is False
    assert len([event for event in client.events if event[0] == "start"]) == 1
    rows = read_log(log_file)
    assert len(rows) == 1
    assert rows[0]["service"] == "yellow"
    assert rows[0]["year_month"] == "2023-01"
    assert rows[0]["final_status"] == "FAILED"
    assert rows[0]["error"] == "Glue.JobFailed | synthetic failure"


def test_make_backfill_defaults_to_dry_run() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["make", "-n", "backfill"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "src/data_pipeline/backfill.py" in result.stdout
    assert "--dry-run" in result.stdout
    assert '--service "all"' in result.stdout
    assert "--execute" not in result.stdout


def test_make_backfill_forwards_all_supported_options() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "make",
            "-n",
            "backfill",
            "BACKFILL_EXECUTE=1",
            "BACKFILL_SERVICE=yellow",
            "BACKFILL_START_MONTH=2023-02",
            "BACKFILL_END_MONTH=2023-04",
            "BACKFILL_STATE_MACHINE_ARN=arn:aws:states:us-east-1:123456789012:stateMachine:monthly",
            "BACKFILL_POLL_SECONDS=3.5",
            "BACKFILL_LOG_FILE=outputs/backfill/sample.csv",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    command = result.stdout
    assert "--execute" in command
    assert "--dry-run" not in command
    assert '--service "yellow"' in command
    assert '--start-month "2023-02"' in command
    assert '--end-month "2023-04"' in command
    assert (
        '--state-machine-arn '
        '"arn:aws:states:us-east-1:123456789012:stateMachine:monthly"'
    ) in command
    assert '--poll-seconds "3.5"' in command
    assert '--log-file "outputs/backfill/sample.csv"' in command
