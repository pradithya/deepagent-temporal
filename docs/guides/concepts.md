# Core Concepts

deepagent-temporal wraps a Deep Agent as a Temporal Workflow, delegating to [langgraph-temporal](https://github.com/pradithya/langgraph-temporal) for the execution engine.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Your Code                                           │
│                                                      │
│  temporal_agent = TemporalDeepAgent(agent, client)   │
│  result = await temporal_agent.ainvoke(input)         │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  TemporalDeepAgent                                   │
│  ├── Composes TemporalGraph (from langgraph-temporal)│
│  ├── Injects worker affinity config                  │
│  └── Injects sub-agent config                        │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│  Temporal Server                                     │
│                                                      │
│  LangGraphWorkflow                                   │
│  ├── call_model → Activity (LLM invocation)          │
│  ├── tools → Activity (tool execution)               │
│  ├── Sub-agent → Child Workflow                      │
│  └── interrupt() → Signal wait (zero-resource)       │
└──────────────────────────────────────────────────────┘
```

## How TemporalDeepAgent works

`TemporalDeepAgent` is a thin wrapper around `TemporalGraph` from langgraph-temporal. It:

1. **Composes** — holds a `TemporalGraph` internally, delegating all execution
2. **Injects config** — adds `use_worker_affinity` and `SubAgentConfig` to the configurable dict before each invocation
3. **Creates workers** — delegates to `create_worker()` with the correct affinity and queue settings

This means all the heavy lifting (Workflow orchestration, Activity dispatch, state management, continue-as-new) is handled by langgraph-temporal. deepagent-temporal adds the Deep Agent-specific features on top.

## Key mapping

| Deep Agent concept | Temporal primitive | How |
|---|---|---|
| Agent execution | Workflow | The compiled graph runs as `LangGraphWorkflow` |
| `call_model` node | Activity | LLM invocation runs as a Temporal Activity |
| `tools` node | Activity | Tool execution runs as a Temporal Activity |
| `task` tool (sub-agent) | Child Workflow | Dispatched via `TemporalSubAgentMiddleware` |
| `interrupt()` | Signal + Wait | Workflow pauses, resumes on Signal |
| `FilesystemBackend` | Worker affinity | All Activities pinned to the same worker |
| Process crash | Temporal recovery | Workflow replays from event history |

## Relationship to langgraph-temporal

```
deepagent-temporal          langgraph-temporal
┌──────────────────┐       ┌──────────────────┐
│ TemporalDeepAgent│──────▶│ TemporalGraph    │
│ SubAgentMiddleware│      │ LangGraphWorkflow│
│ SubAgentSpec     │       │ Activities       │
│ SubAgentRequest  │       │ WorkerGroup      │
└──────────────────┘       └──────────────────┘
```

- **langgraph-temporal**: Generic Temporal backend for any LangGraph graph
- **deepagent-temporal**: Deep Agent-specific wrapper with sub-agent dispatch, worker affinity defaults, and middleware
