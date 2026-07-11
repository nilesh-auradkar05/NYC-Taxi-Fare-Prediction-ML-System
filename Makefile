.PHONY: setup lint test test-integration poisoned dbt-compile evals backfill

UV_CACHE_DIR ?= /tmp/uv-cache
UV ?= uv --cache-dir $(UV_CACHE_DIR)

test: SHELL := env
test: .SHELLFLAGS := -u MAKEFLAGS -u MFLAGS -u MAKELEVEL uv --cache-dir $(UV_CACHE_DIR) run python -X faulthandler -c

setup:
	$(UV) sync

lint:
	$(UV) run ruff check .

test:
	@import pytest, sys; sys.exit(pytest.main(["tests/unit", "-q", "-s"]))

INTEGRATION_PYTEST_ARGS ?=

ifneq ($(filter poisoned,$(MAKECMDGOALS)),)
INTEGRATION_PYTEST_ARGS += -k poisoned
endif

test-integration:
	$(UV) run pytest tests/integration -q -s -m integration $(INTEGRATION_PYTEST_ARGS)

poisoned:
	@:

dbt-compile:
	@echo "dbt project not scaffolded yet. This is a T-201 target stub."

evals:
	@echo "agent eval harness not scaffolded yet. This is a T-506 target stub."

# backfill controls. Leave optional values empty so backfill.py remains
# the source of truth for defaults such as the start/end month and poll interval.
BACKFILL_EXECUTE ?= 0
BACKFILL_SERVICE ?= all
BACKFILL_START_MONTH ?=
BACKFILL_END_MONTH ?=
BACKFILL_STATE_MACHINE_ARN ?=
BACKFILL_POLL_SECONDS ?=
BACKFILL_LOG_FILE ?=

BACKFILL_CLI_ARGS = $(if $(filter 1 true yes,$(BACKFILL_EXECUTE)),--execute,--dry-run) --service "$(BACKFILL_SERVICE)" $(if $(strip $(BACKFILL_START_MONTH)),--start-month "$(BACKFILL_START_MONTH)") $(if $(strip $(BACKFILL_END_MONTH)),--end-month "$(BACKFILL_END_MONTH)") $(if $(strip $(BACKFILL_STATE_MACHINE_ARN)),--state-machine-arn "$(BACKFILL_STATE_MACHINE_ARN)") $(if $(strip $(BACKFILL_POLL_SECONDS)),--poll-seconds "$(BACKFILL_POLL_SECONDS)") $(if $(strip $(BACKFILL_LOG_FILE)),--log-file "$(BACKFILL_LOG_FILE)")

backfill:
	$(UV) run python src/data_pipeline/backfill.py $(BACKFILL_CLI_ARGS)

.PHONY: tf-fmt tf-init tf-validate tf-plan

tf-fmt:
	cd infra && terraform fmt -recursive

tf-init:
	cd infra && terraform init

tf-validate:
	cd infra && terraform validate

tf-plan:
	cd infra && terraform plan -out=nyc-taxi-lambda.tfplan
