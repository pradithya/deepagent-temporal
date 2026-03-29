# Quick Start

This guide shows how to convert an existing Deep Agent to run on Temporal.

## Prerequisites

- deepagent-temporal [installed](installation.md)
- A running Temporal server (see [Installation](installation.md#running-a-local-temporal-server))

## Before: Vanilla Deep Agent

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic

agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-20250514"),
    tools=[read_file, write_file, execute],
    system_prompt="You are a helpful coding assistant.",
    backend=FilesystemBackend(root_dir="/workspace"),
)

# No durability — if the process crashes, all progress is lost.
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Fix the bug in main.py")]},
    config={"configurable": {"thread_id": "task-123"}},
)
```

## After: Temporal-backed Deep Agent

```python
from datetime import timedelta
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic
from temporalio.client import Client

from deepagent_temporal import TemporalDeepAgent

# 1. Create your agent exactly as before
agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-20250514"),
    tools=[read_file, write_file, execute],
    system_prompt="You are a helpful coding assistant.",
    backend=FilesystemBackend(root_dir="/workspace"),
)

# 2. Connect to Temporal and wrap the agent
client = await Client.connect("localhost:7233")
temporal_agent = TemporalDeepAgent(
    agent,
    client,
    task_queue="coding-agents",
    use_worker_affinity=True,  # automatic worker pinning
)

# 3. Same API — now with durable execution
result = await temporal_agent.ainvoke(
    {"messages": [HumanMessage(content="Fix the bug in main.py")]},
    config={"configurable": {"thread_id": "task-123"}},
)
```

Your existing code changes by three lines.

## Running a Worker

The agent graph executes on a Temporal Worker. Run this in a separate process (or on a dedicated machine for filesystem affinity):

```python
import asyncio
from temporalio.client import Client
from temporalio.worker import UnsandboxedWorkflowRunner

from deepagent_temporal import TemporalDeepAgent

async def main():
    agent = create_deep_agent(...)  # same setup as above

    client = await Client.connect("localhost:7233")
    temporal_agent = TemporalDeepAgent(
        agent, client,
        task_queue="coding-agents",
        use_worker_affinity=True,
    )

    worker = temporal_agent.create_worker(
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    async with worker:
        print("Worker running. Ctrl+C to stop.")
        await asyncio.Future()  # run forever

asyncio.run(main())
```

!!! note "Why `UnsandboxedWorkflowRunner`?"
    LangGraph imports modules restricted by Temporal's default workflow sandbox. `UnsandboxedWorkflowRunner()` is required for all workers.

## What's next?

- [Core Concepts](../guides/concepts.md) — understand how it works under the hood
- [Worker Affinity](../guides/worker-affinity.md) — filesystem-safe execution
- [Sub-Agents](../guides/sub-agents.md) — dispatch sub-agents as Child Workflows
- [Human-in-the-Loop](../guides/human-in-the-loop.md) — zero-resource approval waits
