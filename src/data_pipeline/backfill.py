from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import boto3

SERVICES = ("yellow", "hvfhv")
SERVICE_CHOICES = (*SERVICES, "all")
RUNNING_STATUS = "RUNNING"
SUCCESS_STATUS = "SUCCEEDED"
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED", "PENDING_REDRIVE"}
LOG_FIELDS = (
    "timestamp_utc",
    "service",
    "year_month",
    "execution_name",
    "execution_arn",
    "final_status",
    "duration_seconds",
    "notes",
    "error",
)


class StepFunctionsClient(Protocol):
    def start_execution(self, **kwargs: Any) -> dict[str, Any]: ...

    def describe_execution(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PlannedExecution:
    service: str
    year_month: str
    execution_name: str
    payload: dict[str, Any]


def parse_month(value: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid month {value!r}; expected YYYY-MM") from exc
    return parsed.year, parsed.month


def month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start_number = parse_month(start_month)
    end_year, end_number = parse_month(end_month)
    if (start_year, start_number) > (end_year, end_number):
        raise ValueError(f"start month {start_month} is after end month {end_month}")

    months: list[str] = []
    year, month = start_year, start_number
    while (year, month) <= (end_year, end_number):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def selected_services(service: str) -> tuple[str, ...]:
    if service == "all":
        return SERVICES
    if service in SERVICES:
        return (service,)
    raise ValueError(
        f"unsupported service {service!r}; expected one of {', '.join(SERVICE_CHOICES)}"
    )


def build_plan(
    start_month: str,
    end_month: str,
    *,
    run_id: str,
    requested_at: str,
    service: str = "all",
) -> list[PlannedExecution]:
    plan: list[PlannedExecution] = []
    services = selected_services(service)
    for year_month in month_range(start_month, end_month):
        for selected_service in services:
            execution_name = f"backfill-{selected_service}-{year_month}-{run_id}"
            payload = {
                "service": selected_service,
                "year_month": year_month,
                "trigger": "backfill",
                "backfill": {
                    "ticket": "T-106",
                    "run_id": run_id,
                    "requested_at": requested_at,
                },
            }
            plan.append(
                PlannedExecution(
                    service=selected_service,
                    year_month=year_month,
                    execution_name=execution_name,
                    payload=payload,
                )
            )
    return plan


def resolve_state_machine_arn(explicit_arn: str | None) -> str:
    if explicit_arn:
        return explicit_arn

    environment_arn = os.environ.get("NYC_TAXI_STATE_MACHINE_ARN")
    if environment_arn:
        return environment_arn

    command = [
        "terraform",
        "-chdir=infra",
        "output",
        "-raw",
        "monthly_ingestion_state_machine_arn",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        details = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(
            "could not resolve the Step Functions ARN; pass --state-machine-arn, set "
            "NYC_TAXI_STATE_MACHINE_ARN, or make the Terraform output available. "
            f"Underlying error: {details.strip()}"
        ) from exc

    arn = result.stdout.strip()
    if not arn:
        raise RuntimeError("Terraform output monthly_ingestion_state_machine_arn was empty")
    return arn


def append_log_row(log_file: Path, row: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not log_file.exists() or log_file.stat().st_size == 0
    with log_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def wait_for_terminal_status(
    client: StepFunctionsClient,
    execution_arn: str,
    *,
    poll_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    while True:
        response = client.describe_execution(executionArn=execution_arn)
        status = response.get("status")
        if status == RUNNING_STATUS:
            sleep(poll_seconds)
            continue
        if status not in TERMINAL_STATUSES:
            raise RuntimeError(f"DescribeExecution returned unknown status {status!r}")
        return response


def _duration_seconds(details: dict[str, Any], fallback: float) -> float:
    start_date = details.get("startDate")
    stop_date = details.get("stopDate")
    if isinstance(start_date, datetime) and isinstance(stop_date, datetime):
        return max(0.0, (stop_date - start_date).total_seconds())
    return max(0.0, fallback)


def _failure_text(details: dict[str, Any]) -> str:
    parts = [str(details[key]).strip() for key in ("error", "cause") if details.get(key)]
    return " | ".join(parts)


def execute_plan(
    plan: Iterable[PlannedExecution],
    *,
    client: StepFunctionsClient,
    state_machine_arn: str,
    log_file: Path,
    poll_seconds: float,
    stop_on_failure: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> bool:
    all_succeeded = True

    for item in plan:
        started_at = time.monotonic()
        timestamp = now().isoformat()
        execution_arn = ""
        print(f"START {item.service} {item.year_month} name={item.execution_name}", flush=True)

        try:
            response = client.start_execution(
                stateMachineArn=state_machine_arn,
                name=item.execution_name,
                input=json.dumps(item.payload, separators=(",", ":"), sort_keys=True),
            )
            execution_arn = response["executionArn"]
            print(f"ARN   {execution_arn}", flush=True)
        except Exception as exc:
            append_log_row(
                log_file,
                {
                    "timestamp_utc": timestamp,
                    "service": item.service,
                    "year_month": item.year_month,
                    "execution_name": item.execution_name,
                    "execution_arn": execution_arn,
                    "final_status": "START_FAILED",
                    "duration_seconds": f"{time.monotonic() - started_at:.3f}",
                    "notes": "StartExecution failed; no later month was started.",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"FAILED to start {item.service} {item.year_month}: {exc}", file=sys.stderr, flush=True)
            all_succeeded = False
            if stop_on_failure:
                return False
            continue

        try:
            details = wait_for_terminal_status(
                client,
                execution_arn,
                poll_seconds=poll_seconds,
                sleep=sleep,
            )
        except KeyboardInterrupt:
            append_log_row(
                log_file,
                {
                    "timestamp_utc": timestamp,
                    "service": item.service,
                    "year_month": item.year_month,
                    "execution_name": item.execution_name,
                    "execution_arn": execution_arn,
                    "final_status": "LOCAL_INTERRUPTED",
                    "duration_seconds": f"{time.monotonic() - started_at:.3f}",
                    "notes": "Local polling stopped; the AWS execution may still be running. Inspect the ARN.",
                    "error": "KeyboardInterrupt",
                },
            )
            raise
        except Exception as exc:
            append_log_row(
                log_file,
                {
                    "timestamp_utc": timestamp,
                    "service": item.service,
                    "year_month": item.year_month,
                    "execution_name": item.execution_name,
                    "execution_arn": execution_arn,
                    "final_status": "POLL_FAILED",
                    "duration_seconds": f"{time.monotonic() - started_at:.3f}",
                    "notes": "DescribeExecution polling failed; inspect the AWS execution ARN.",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"FAILED while polling {execution_arn}: {exc}", file=sys.stderr, flush=True)
            all_succeeded = False
            if stop_on_failure:
                return False
            continue

        status = details["status"]
        duration = _duration_seconds(details, time.monotonic() - started_at)
        error = _failure_text(details)
        append_log_row(
            log_file,
            {
                "timestamp_utc": timestamp,
                "service": item.service,
                "year_month": item.year_month,
                "execution_name": item.execution_name,
                "execution_arn": execution_arn,
                "final_status": status,
                "duration_seconds": f"{duration:.3f}",
                "notes": "Execution completed." if status == SUCCESS_STATUS else "Execution ended unsuccessfully.",
                "error": error,
            },
        )
        print(f"DONE  {item.service} {item.year_month} status={status} duration={duration:.3f}s", flush=True)

        if status != SUCCESS_STATUS:
            all_succeeded = False
            if error:
                print(f"ERROR {error}", file=sys.stderr, flush=True)
            if stop_on_failure:
                return False

    return all_succeeded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the T-106 monthly ingestion backfill serially.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the execution plan without AWS calls")
    mode.add_argument("--execute", action="store_true", help="start and poll real Step Functions executions")
    parser.add_argument(
        "--service",
        choices=SERVICE_CHOICES,
        default="all",
        help="service to backfill: yellow, hvfhv, or all (default: all)",
    )
    parser.add_argument("--start-month", default="2023-01", metavar="YYYY-MM")
    parser.add_argument("--end-month", default=datetime.now(UTC).strftime("%Y-%m"), metavar="YYYY-MM")
    parser.add_argument("--state-machine-arn")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--log-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.poll_seconds < 0:
        parser.error("--poll-seconds must be non-negative")

    requested_at = datetime.now(UTC).replace(microsecond=0)
    run_id = f"{requested_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    try:
        plan = build_plan(
            args.start_month,
            args.end_month,
            run_id=run_id,
            requested_at=requested_at.isoformat(),
            service=args.service,
        )
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    if not args.execute:
        print(f"DRY RUN: {len(plan)} serial executions; no AWS calls will be made.")
        print(f"SERVICE SELECTION: {args.service}")
        for index, item in enumerate(plan, start=1):
            payload = json.dumps(item.payload, separators=(",", ":"), sort_keys=True)
            print(f"{index:03d} {item.service:<6} {item.year_month} {item.execution_name} input={payload}")
        return 0

    state_machine_arn = resolve_state_machine_arn(args.state_machine_arn)
    log_file = args.log_file or Path("outputs/backfill") / f"t106-{run_id}.csv"
    print(f"EXECUTE: {len(plan)} serial executions")
    print(f"SERVICE SELECTION: {args.service}")
    print(f"STATE MACHINE: {state_machine_arn}")
    print(f"RUN LOG: {log_file}")

    try:
        succeeded = execute_plan(
            plan,
            client=boto3.client("stepfunctions"),
            state_machine_arn=state_machine_arn,
            log_file=log_file,
            poll_seconds=args.poll_seconds,
        )
    except KeyboardInterrupt:
        print("Interrupted. No later month was started; inspect the logged execution ARN.", file=sys.stderr)
        return 130
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
