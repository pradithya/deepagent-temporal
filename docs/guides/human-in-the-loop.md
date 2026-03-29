# Human-in-the-Loop

Deep Agents' `interrupt()` works out of the box with deepagent-temporal. The Workflow pauses with **zero resource consumption** and resumes when you send a Signal.

## Starting a workflow with HITL

```python
# Start the agent (non-blocking)
handle = await temporal_agent.astart(
    {"messages": [HumanMessage(content="Refactor auth module")]},
    config={"configurable": {"thread_id": "task-456"}},
)
```

## Checking for interrupts

```python
state = await temporal_agent.get_state(
    {"configurable": {"thread_id": "task-456"}}
)

if state["status"] == "interrupted":
    print("Pending approval:", state["interrupts"])
```

## Resuming

```python
await temporal_agent.resume(
    {"configurable": {"thread_id": "task-456"}},
    "approved",
)
```

## Complete example

```python
import asyncio
from deepagent_temporal import TemporalDeepAgent

async def run_with_approval():
    # Start agent
    handle = await temporal_agent.astart(
        {"messages": [HumanMessage(content="Deploy to production")]},
        config={"configurable": {"thread_id": "deploy-1"}},
    )

    # Poll for interrupt (in practice, use a webhook or UI)
    while True:
        state = await temporal_agent.get_state(
            {"configurable": {"thread_id": "deploy-1"}}
        )

        if state["status"] == "interrupted":
            print("Agent needs approval:", state["interrupts"])
            approval = input("Approve? (y/n): ")
            await temporal_agent.resume(
                {"configurable": {"thread_id": "deploy-1"}},
                "approved" if approval == "y" else "rejected",
            )

        if state["status"] == "completed":
            break

        await asyncio.sleep(1)

    result = await handle.result()
    print("Final result:", result)
```

## Why this matters for agents

In vanilla Deep Agents, `interrupt()` blocks a running process. With Temporal:

- **No compute wasted** — the Worker is free to handle other Workflows while waiting
- **Survives restarts** — if the Worker restarts, the paused state is preserved in Temporal's event history
- **No timeout** — the Workflow can wait days or weeks for human input
- **Queryable from anywhere** — check interrupt status from any process, API, or UI
