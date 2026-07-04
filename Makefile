.PHONY: setup lint test test-integration dbt-compile evals

UV_CACHE_DIR ?= /tmp/uv-cache
UV ?= uv --cache-dir $(UV_CACHE_DIR)

test test-integration: SHELL := env
test test-integration: .SHELLFLAGS := -u MAKEFLAGS -u MFLAGS -u MAKELEVEL uv --cache-dir $(UV_CACHE_DIR) run python -X faulthandler -c

setup:
	$(UV) sync

lint:
	$(UV) run ruff check .

test:
	@import pytest, sys; sys.exit(pytest.main(["tests/unit", "-q", "-s"]))

test-integration:
	@import pytest, sys; sys.exit(pytest.main(["tests/integration", "-q", "-s", "-m", "integration"]))

dbt-compile:
	@echo "dbt project not scaffolded yet. This is a T-201 target stub."

evals:
	@echo "agent eval harness not scaffolded yet. This is a T-506 target stub."

.PHONY: tf-fmt tf-init tf-validate tf-plan

tf-fmt:
	cd infra && terraform fmt -recursive

tf-init:
	cd infra && terraform init

tf-validate:
	cd infra && terraform validate

tf-plan:
	cd infra && terraform plan -out=nyc-taxi.tfplan