.PHONY: test test_watch test_integration lint type format build start_temporal stop_temporal wait_temporal test_integration_docker docs docs_serve setup

######################
# TESTING AND COVERAGE
######################

TEST ?= .

test:
	uv run pytest $(TEST)

test_integration:
	uv run pytest -m integration $(TEST)

test_watch:
	uv run ptw $(TEST)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --relative --diff-filter=d main . | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=deepagent_temporal
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

type:
	mkdir -p $(MYPY_CACHE) && uv run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)

######################
# DOCS
######################

docs:
	uv run mkdocs build --strict

docs_serve:
	uv run mkdocs serve

######################
# SETUP
######################

setup:
	uv sync
	git config core.hooksPath .githooks

######################
# BUILD
######################

build:
	uv build

######################
# DOCKER TEMPORAL
######################

TEMPORAL_GRPC_PORT ?= 7233

start_temporal:
	TEMPORAL_GRPC_PORT=$(TEMPORAL_GRPC_PORT) docker compose up -d

stop_temporal:
	docker compose down

wait_temporal:
	@echo "Waiting for Temporal server to be ready..."
	@timeout=120; while [ "$$(docker inspect -f '{{.State.Status}}' temporal-create-namespace 2>/dev/null)" != "exited" ] || \
		[ "$$(docker inspect -f '{{.State.ExitCode}}' temporal-create-namespace 2>/dev/null)" != "0" ]; do \
		sleep 2; timeout=$$((timeout - 2)); \
		if [ $$timeout -le 0 ]; then echo "Temporal failed to start"; docker compose logs; exit 1; fi; \
	done
	@echo "Temporal server is ready."

test_integration_docker: start_temporal wait_temporal
	TEMPORAL_ADDRESS=localhost:$(TEMPORAL_GRPC_PORT) uv run pytest -m integration $(TEST)
