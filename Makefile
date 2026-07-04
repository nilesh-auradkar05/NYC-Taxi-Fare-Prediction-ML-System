.PHONY: setup lint test test-integration dbt-compile evals

UV ?= uv --cache-dir /tmp/uv-cache
RUN_NO_MAKE_ENV := env -u MAKEFLAGS -u MFLAGS -u MAKELEVEL

setup:
	$(RUN_NO_MAKE_ENV) $(UV) sync

lint:
	$(RUN_NO_MAKE_ENV) $(UV) run ruff check .

test:
	$(RUN_NO_MAKE_ENV) $(UV) run pytest tests/unit -q -s

test-integration:
	$(RUN_NO_MAKE_ENV) $(UV) run pytest tests/integration -q -s -m integration

dbt-compile:
	@echo "dbt project not scaffolded yet. This is a T-201 target stub."

evals:
	@echo "agent eval harness not scaffolded yet. This is a T-506 target stub."
