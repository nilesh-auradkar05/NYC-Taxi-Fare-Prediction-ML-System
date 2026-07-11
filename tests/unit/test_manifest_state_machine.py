import json
from pathlib import Path

TEMPLATE_PATH = (
    Path(__file__).parents[2]
    / "infra"
    / "modules"
    / "stepfunctions_skeleton"
    / "step_functions"
    / "ingestion.asl.json.tftpl"
)


def _load_definition() -> dict:
    rendered = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "${check_manifest_lambda_arn}": "arn:aws:lambda:us-east-1:123:function:check",
        "${fetch_to_bronze_lambda_arn}": "arn:aws:lambda:us-east-1:123:function:fetch",
        "${conform_job_name}": "conform-job",
        "${dq_gate_job_name}": "dq-job",
        "${alert_topic_arn}": "arn:aws:sns:us-east-1:123:alerts",
        "${manifest_table_name}": "manifest-table",
    }
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    return json.loads(rendered)


def test_manifest_actions_route_fetch_resume_and_skip() -> None:
    states = _load_definition()["States"]
    choices = states["SelectPipelineAction"]["Choices"]

    assert choices == [
        {
            "Variable": "$.pipeline_action",
            "StringEquals": "fetch",
            "Next": "FetchToBronze",
        },
        {
            "Variable": "$.pipeline_action",
            "StringEquals": "resume",
            "Next": "MarkManifestProcessing",
        },
    ]
    assert states["SelectPipelineAction"]["Default"] == "NoNewFile"


def test_complete_is_written_only_after_dq_and_event_success() -> None:
    states = _load_definition()["States"]

    assert states["DQGate"]["Next"] == "EmitSilverUpdatedEvent"
    assert states["EmitSilverUpdatedEvent"]["Next"] == "MarkManifestComplete"
    assert states["MarkManifestComplete"]["Parameters"]["ExpressionAttributeValues"][
        ":status"
    ] == {"S": "complete"}


def test_dq_failure_marks_manifest_failed_before_alerting() -> None:
    states = _load_definition()["States"]

    assert states["DQGate"]["Catch"][0]["Next"] == "MarkManifestFailedAfterDQ"
    failed_state = states["MarkManifestFailedAfterDQ"]
    assert failed_state["Resource"] == "arn:aws:states:::dynamodb:updateItem"
    assert failed_state["Parameters"]["ExpressionAttributeValues"][":status"] == {
        "S": "failed"
    }
    assert failed_state["Parameters"]["ExpressionAttributeValues"][
        ":failure_stage"
    ] == {
        "S": "DQGate"
    }
    assert failed_state["Next"] == "NotifyDQFailure"


def test_processing_and_terminal_updates_use_manifest_etag_condition() -> None:
    states = _load_definition()["States"]

    assert (
        "#etag = :etag"
        in states["MarkManifestProcessing"]["Parameters"]["ConditionExpression"]
    )
    assert (
        "#etag = :etag"
        in states["MarkManifestComplete"]["Parameters"]["ConditionExpression"]
    )
    failure_condition = states["MarkManifestFailedAfterDQ"]["Parameters"][
        "ConditionExpression"
    ]
    assert "#etag = :etag" in failure_condition


def test_all_state_transitions_reference_existing_states() -> None:
    states = _load_definition()["States"]
    targets = []
    for state in states.values():
        if "Next" in state:
            targets.append(state["Next"])
        if state.get("Type") == "Choice":
            targets.extend(choice["Next"] for choice in state.get("Choices", []))
            if "Default" in state:
                targets.append(state["Default"])
        targets.extend(catcher["Next"] for catcher in state.get("Catch", []))

    assert set(targets).issubset(states)
