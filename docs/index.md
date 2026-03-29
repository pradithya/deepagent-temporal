# deepagent-temporal

**Temporal integration for [Deep Agents](https://github.com/langchain-ai/deepagents) — durable execution for AI agent workflows.**

If your Deep Agent process crashes mid-task, all progress is lost. Sub-agents are ephemeral. Human-in-the-loop approval blocks a running process. `deepagent-temporal` solves these problems by running your Deep Agent as a [Temporal](https://temporal.io) Workflow.

!!! warning "Experimental"
    This project is experimental. Use at your own risk.

---

## Key features

| Feature | Without Temporal | With deepagent-temporal |
|---|---|---|
| **Durable execution** | Process crash = lost progress | Survives crashes, restarts, deployments |
| **Sub-agents** | In-process, ephemeral | Independent Child Workflows |
| **Worker affinity** | N/A | Sticky task queues keep file ops on same machine |
| **Human-in-the-loop** | Blocks a live process | Zero-resource wait via Temporal Signals |

## 3-line migration

```python
# Connect to Temporal and wrap your agent
client = await Client.connect("localhost:7233")
temporal_agent = TemporalDeepAgent(agent, client, task_queue="coding-agents")

# Same API — now with durable execution
result = await temporal_agent.ainvoke(
    {"messages": [HumanMessage(content="Fix the bug in main.py")]},
    config={"configurable": {"thread_id": "task-123"}},
)
```

The `ainvoke`, `astream`, `get_state`, and `resume` APIs are identical to vanilla Deep Agents.

## Next steps

- [Installation](getting-started/installation.md) — install the package
- [Quick Start](getting-started/quickstart.md) — run your first Temporal-backed agent
- [Core Concepts](guides/concepts.md) — understand the architecture
- [API Reference](reference/agent.md) — full API documentation
