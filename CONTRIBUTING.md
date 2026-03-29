# Contributing to deepagent-temporal

Thank you for your interest in contributing! This project is experimental and welcomes contributions of all kinds.

## Development Setup

### Prerequisites

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/) (for integration tests with Temporal server)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/pradithya/deepagent-temporal.git
cd deepagent-temporal

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[test,lint,docs]"

# Install pre-commit hooks
git config core.hooksPath .githooks
```

### Running Tests

```bash
# Unit tests (no external services needed)
make test

# Integration tests (requires running Temporal server)
make start_temporal       # Start Temporal via Docker
make wait_temporal        # Wait for it to be ready
make test_integration     # Run integration tests
make stop_temporal        # Stop Temporal

# All-in-one: start Temporal, run tests, tear down
make test_integration_docker
```

### Linting and Formatting

A pre-commit hook runs `make format && make lint` on staged files. You can also run them manually:

```bash
make format   # Auto-fix formatting and import sorting
make lint     # Check for issues (ruff + mypy)
make type     # Type checking only (mypy)
```

## Code Conventions

- **Python 3.10+ syntax**: Use `X | Y` unions, not `Union[X, Y]`.
- **Type annotations**: All functions must have type annotations.
- **Import sorting**: Handled by `ruff check --select I --fix`.
- **No unused imports**: Enforced by ruff.
- **Docstrings**: Use single backticks for inline code (not Sphinx double backticks).

## Architecture

This is a **thin wrapper** over [`langgraph-temporal`](https://github.com/pradithya/langgraph-temporal). Most workflow logic lives upstream. Before making changes, read:

- `docs/REQUIREMENTS.md` — functional requirements
- `docs/DESIGN.md` — technical design
- `docs/architecture-decisions.md` — why things are the way they are

The package has four source files:

| File | Purpose |
|---|---|
| `agent.py` | `TemporalDeepAgent` wrapper, composes `TemporalGraph` |
| `middleware.py` | `TemporalSubAgentMiddleware` for Child Workflow dispatch |
| `config.py` | `SubAgentSpec` dataclass |
| `serialization.py` | Payload size validation utilities |

## How to Contribute

### Reporting Issues

Open a [GitHub issue](https://github.com/pradithya/deepagent-temporal/issues) with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version, `temporalio` version, `langgraph-temporal` version

### Submitting Changes

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes.
4. Run the verification checklist:
   ```bash
   make format
   make lint
   make test
   ```
5. If your changes affect Temporal Workflow behavior, also run:
   ```bash
   make test_integration_docker
   ```
6. Open a pull request with a clear description of the change and its motivation.

### Good First Issues

Look for issues labeled `good first issue` on GitHub. These are smaller, well-scoped tasks suitable for new contributors. Examples:

- Adding tests for edge cases
- Improving documentation
- Adding type annotations to test files
- Adding examples for specific use cases

## Questions?

Open a GitHub issue or discussion. We're happy to help you get oriented in the codebase.
