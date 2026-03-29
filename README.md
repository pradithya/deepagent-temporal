# deepagent-temporal

Temporal integration for [Deep Agents](https://github.com/langchain-ai/deepagents) — durable execution for AI agent workflows.

If your Deep Agent process crashes mid-task, all progress is lost. Sub-agents are ephemeral. Human-in-the-loop approval blocks a running process. `deepagent-temporal` solves these problems by running your Deep Agent as a [Temporal](https://temporal.io) Workflow:

- **Durable execution** — survives process crashes, restarts, and deployments
- **Sub-agent dispatch** — sub-agents run as independent Temporal Child Workflows
- **Worker affinity** — sticky task queues keep file operations on the same machine, side stepping the need of NFS or shared storage.
- **Zero-resource HITL** — workflow pauses consume no compute while waiting for approval

> This project is experimental, use at your own risk

## Installation

```bash
pip install deepagent-temporal
```

Requires Python 3.10+, [langgraph-temporal](https://github.com/pradithya/langgraph-temporal) >= 0.1.0, and a running Temporal server.

## Quick Start

### Before: vanilla Deep Agent

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
# Sub-agents run in-process. HITL blocks a live process.
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Fix the bug in main.py")]},
    config={"configurable": {"thread_id": "task-123"}},
)
```

### After: Temporal-backed Deep Agent

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

The `ainvoke`, `astream`, `get_state`, and `resume` APIs are identical. Your existing code changes by three lines.

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

    # create_worker() returns a WorkerGroup with two internal workers:
    # one on the shared queue (Workflows + discovery) and one on a
    # unique queue (node Activities).
    worker = temporal_agent.create_worker(
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    async with worker:
        print("Worker running. Ctrl+C to stop.")
        await asyncio.Future()  # run forever

asyncio.run(main())
```

## Worker Affinity via Worker-Specific Task Queues

Deep Agents often use `FilesystemBackend` — tools read and write files on the local disk. All Activities for an agent must run on the same worker to keep the filesystem consistent.

Enable `use_worker_affinity=True` and the framework handles it automatically following the [Temporal worker-specific task queues pattern](https://github.com/temporalio/samples-python/tree/main/worker_specific_task_queues):

```python
temporal_agent = TemporalDeepAgent(
    agent, client,
    task_queue="coding-agents",
    use_worker_affinity=True,  # transparent to the client
)
```

**How it works:**
1. `create_worker()` generates a unique queue name per worker process and starts two internal workers: one on the shared queue (Workflows + discovery Activity), one on its unique queue (node Activities)
2. When a Workflow starts, it calls a `get_available_task_queue` Activity on the shared queue — whichever worker picks it up returns its unique queue
3. All subsequent node Activities are dispatched to that discovered queue
4. The discovered queue survives `continue-as-new` — the same worker stays pinned across workflow runs
5. HITL waits have no timeout concern — the queue persists independently

The client never needs to know queue names. Workers self-register.

## Sub-Agents as Child Workflows

Deep Agents can spawn sub-agents via the `task` tool. With `deepagent-temporal`, each sub-agent runs as an independent Temporal Child Workflow with its own durability, timeout, and observability.

### Setting up the middleware

`TemporalSubAgentMiddleware` intercepts `task` tool calls and dispatches them as Child Workflows instead of running them in-process. Inject it **before** graph compilation:

```python
from deepagent_temporal import TemporalSubAgentMiddleware

middleware = TemporalSubAgentMiddleware(
    subagent_specs={
        "researcher": "subagent:researcher",
        "coder": "subagent:coder",
    },
)

agent = create_deep_agent(
    model=model,
    tools=tools,
    middleware=[middleware],  # inject before compilation
    # ... other params
)
```

### Configuring sub-agent execution

```python
temporal_agent = TemporalDeepAgent(
    agent, client,
    task_queue="main-agents",
    subagent_task_queue="sub-agents",           # separate queue for sub-agents
    subagent_execution_timeout=timedelta(minutes=15),  # per sub-agent timeout
)
```

When the LLM invokes the `task` tool, the middleware stores a `SubAgentRequest` in a context variable. The Activity collects it, and the Workflow dispatches a Child Workflow. The result flows back as a `ToolMessage` to the parent agent — exactly matching the behavior of the in-process `SubAgentMiddleware`.

## Human-in-the-Loop

Deep Agents' `interrupt()` works out of the box. The workflow pauses with zero resource consumption and resumes when you send a Signal:

```python
# Start the agent (non-blocking)
handle = await temporal_agent.astart(
    {"messages": [HumanMessage(content="Refactor auth module")]},
    config={"configurable": {"thread_id": "task-456"}},
)

# ... later, check if it's waiting for approval
state = await temporal_agent.get_state(
    {"configurable": {"thread_id": "task-456"}}
)
if state["status"] == "interrupted":
    print("Pending approval:", state["interrupts"])

    # Approve and resume
    await temporal_agent.resume(
        {"configurable": {"thread_id": "task-456"}},
        "approved",
    )
```

## Local Development

For testing without a Temporal server deployment:

```python
temporal_agent = await TemporalDeepAgent.local(agent)
result = await temporal_agent.ainvoke({"messages": ["hello"]})
```

This starts an in-process Temporal test server automatically.

## Testing

```bash
# Unit + integration tests (uses in-process Temporal test server)
make test

# Integration tests only
make test_integration

# Integration tests against Dockerized Temporal
make test_integration_docker
```

## License

MIT
