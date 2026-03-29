# Installation

## Requirements

- Python >= 3.10
- [langgraph-temporal](https://github.com/pradithya/langgraph-temporal) >= 0.1.0
- A running [Temporal](https://temporal.io/) server

## Install from PyPI

```bash
pip install deepagent-temporal
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add deepagent-temporal
```

This installs the core dependencies:

- `langgraph-temporal >= 0.1.0`
- `langgraph >= 1.0.0`
- `temporalio >= 1.7.0`

## Running a local Temporal server

For development, use Docker Compose. The repository includes a pre-configured setup:

```bash
git clone https://github.com/pradithya/deepagent-temporal.git
cd deepagent-temporal

cp .env.ci .env
make start_temporal
```

This starts:

- **Temporal Server** on `localhost:7233` (gRPC)
- **Temporal Web UI** on `http://localhost:8233`
- **PostgreSQL** as the persistence backend
- **Elasticsearch** for workflow visibility

To stop:

```bash
make stop_temporal
```

!!! tip "Temporal CLI"
    For a lighter-weight setup, use the [Temporal CLI](https://docs.temporal.io/cli):

    ```bash
    temporal server start-dev
    ```
