# Local Development

For quick testing without deploying a Temporal server, use the built-in local test server.

## Using TemporalDeepAgent.local()

```python
from deepagent_temporal import TemporalDeepAgent

temporal_agent = await TemporalDeepAgent.local(agent)
result = await temporal_agent.ainvoke({"messages": ["hello"]})
```

This starts an **in-process Temporal test server** automatically. No Docker, no external services.

## Full example

```python
import asyncio
from temporalio.worker import UnsandboxedWorkflowRunner
from deepagent_temporal import TemporalDeepAgent

async def main():
    agent = create_deep_agent(...)  # your agent setup

    temporal_agent = await TemporalDeepAgent.local(agent)

    worker = temporal_agent.create_worker(
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    async with worker:
        result = await temporal_agent.ainvoke(
            {"messages": [HumanMessage(content="Hello")]},
            config={"configurable": {"thread_id": "test-1"}},
        )
        print(result)

asyncio.run(main())
```

## Running with Docker Compose

For integration testing closer to production:

```bash
# Start Temporal server
cp .env.ci .env
make start_temporal
make wait_temporal

# Run integration tests
make test_integration

# Or run against Docker
make test_integration_docker

# Tear down
make stop_temporal
```

## Test targets

| Command | Description |
|---|---|
| `make test` | Unit tests (no Temporal server needed) |
| `make test_integration` | Integration tests (requires running Temporal) |
| `make test_integration_docker` | Start Docker Temporal + run integration tests |
