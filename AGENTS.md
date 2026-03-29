# Agents

Instructions for AI coding agents working on this project.

## Before making changes

1. Read `CLAUDE.md` for project conventions and architecture overview
2. Read `docs/REQUIREMENTS.md` and `docs/DESIGN.md` to understand the design intent
3. Understand that this is a thin wrapper over `langgraph-temporal` — most Workflow logic lives upstream

## Verification checklist

Run these commands before considering any task complete:

```bash
make format          # auto-fix formatting and imports
make lint            # ruff check + ruff format --diff + mypy
make test            # unit tests (no external services)
```

If your changes affect Temporal Workflow behavior, also run:

```bash
make test_integration        # requires running Temporal server
make test_integration_docker # self-contained: starts Temporal, runs tests, tears down
```

## Code conventions

- Python 3.10+ syntax: `X | Y` unions, no `from __future__ import annotations` unless needed for forward refs
- All functions must have type annotations
- Use single backticks for inline code in docstrings (not Sphinx double backticks)
- Keep imports sorted with `ruff check --select I --fix`
- No unused imports — ruff enforces this

## Pre-commit hook

A git pre-commit hook runs `make format && make lint` on staged Python files. If the hook fails, fix the issues before committing. Do not bypass with `--no-verify`.
