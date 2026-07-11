from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from tests.integration.test_poisoned_pipeline import (
    _assert_ordered_subsequence,
    _execution_name,
    _skip_if_aws_preflight_unavailable,
    _sql_string,
    _validate_deployed_state_machine,
    poisoned_rows,
)

INTEGRATION_TEST = Path("tests/integration/test_poisoned_pipeline.py")


def _load_module_ast() -> ast.Module:
    return ast.parse(INTEGRATION_TEST.read_text(encoding="utf-8"))


def test_poisoned_rows_cover_null_domain_and_month_mismatch() -> None:
    rows = poisoned_rows("2099-01")

    assert len(rows) == 3
    assert rows[0]["tpep_pickup_datetime"] is None
    assert rows[1]["fare_amount"] == "-5.0"
    assert rows[2]["tpep_pickup_datetime"] == "2098-12-28 23:50:00"
    assert rows[2]["tpep_dropoff_datetime"] == "2099-01-01 00:10:00"


def test_ordered_subsequence_rejects_out_of_order_state_history() -> None:
    expected = ["CheckManifest", "DQGate", "NotifyDQFailure", "DQFailed"]
    _assert_ordered_subsequence(
        ["CheckManifest", "GlueConform", "DQGate", "NotifyDQFailure", "DQFailed"],
        expected,
    )

    try:
        _assert_ordered_subsequence(
            ["CheckManifest", "NotifyDQFailure", "DQGate", "DQFailed"],
            expected,
        )
    except AssertionError as exc:
        assert "Expected ordered state path" in str(exc)
    else:  # pragma: no cover - proves the helper must reject the bad order
        raise AssertionError("Out-of-order state history was accepted")


class _FakeStepFunctions:
    def __init__(self, states: dict) -> None:
        self.states = states

    def describe_state_machine(self, *, stateMachineArn: str) -> dict[str, str]:
        assert stateMachineArn == "arn:test"
        import json

        return {"definition": json.dumps({"States": self.states})}


def _valid_state_machine_states() -> dict:
    return {
        "SelectPipelineAction": {},
        "MarkManifestProcessing": {},
        "DQGate": {"Catch": [{"Next": "MarkManifestFailedAfterDQ"}]},
        "MarkManifestFailedAfterDQ": {"Next": "NotifyDQFailure"},
        "NotifyDQFailure": {
            "Resource": "arn:aws:states:::sns:publish",
            "Next": "DQFailed",
        },
        "DQFailed": {"Error": "DataQualityGateFailed"},
    }


def test_deployed_state_machine_preflight_rejects_old_definition() -> None:
    _validate_deployed_state_machine(_FakeStepFunctions(_valid_state_machine_states()), "arn:test")

    old_states = _valid_state_machine_states()
    del old_states["SelectPipelineAction"]
    try:
        _validate_deployed_state_machine(_FakeStepFunctions(old_states), "arn:test")
    except RuntimeError as exc:
        assert "missing states" in str(exc)
    else:  # pragma: no cover - proves old deployed definitions must be rejected
        raise AssertionError("Old state-machine definition passed the T-107 preflight")


def test_execution_name_and_sql_literal_are_safe() -> None:
    name = _execution_name("yellow", "2099-01", "a" * 32)
    assert len(name) <= 80
    assert name.startswith("t107-poisoned-yellow-2099-01-")
    assert _sql_string("O'Brien") == "'O''Brien'"


def test_aws_permission_preflight_skips_loudly() -> None:
    exc = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "DescribeStateMachine",
    )

    with pytest.raises(pytest.skip.Exception) as skipped:
        _skip_if_aws_preflight_unavailable("describe the dev state machine", exc)

    assert "T-107 integration skipped" in str(skipped.value)
    assert "AccessDeniedException" in str(skipped.value)


def test_aws_contract_errors_do_not_skip() -> None:
    exc = ClientError(
        {"Error": {"Code": "StateMachineDoesNotExist", "Message": "missing"}},
        "DescribeStateMachine",
    )

    assert _skip_if_aws_preflight_unavailable("describe the dev state machine", exc) is None


def test_poisoned_integration_test_targets_required_contracts() -> None:
    source = INTEGRATION_TEST.read_text(encoding="utf-8")
    assert "@pytest.mark.integration" in source
    assert 'terminal["status"] == "FAILED"' in source
    assert 'terminal.get("error") == "DataQualityGateFailed"' in source
    assert '"NotifyDQFailure"' in source
    assert '"EmitSilverUpdatedEvent" not in states' in source
    assert '"MarkManifestComplete" not in states' in source

    for metric_name in (
        "dq_02_required_null_rates_ok",
        "dq_03_domain_values_ok",
        "dq_04_timestamp_sanity_ok",
    ):
        assert metric_name in source
    for reason_code in ("BAD_PICKUP_TS", "BAD_FARE", "MONTH_MISMATCH"):
        assert reason_code in source


def test_integration_test_uses_runtime_generated_parquet_not_committed_binary() -> None:
    tree = _load_module_ast()
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "write_poisoned_parquet" in function_names
    assert not list(Path("tests").rglob("*.parquet"))


def test_make_acceptance_command_filters_poisoned_test() -> None:
    completed = subprocess.run(
        ["make", "-n", "test-integration", "-k", "poisoned"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "pytest tests/integration" in completed.stdout
    assert "-m integration -k poisoned" in completed.stdout
