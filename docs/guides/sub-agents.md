# Sub-Agents as Child Workflows

Deep Agents can spawn sub-agents via the `task` tool. With deepagent-temporal, each sub-agent runs as an independent Temporal Child Workflow with its own durability, timeout, and observability.

## Setting up the middleware

`TemporalSubAgentMiddleware` intercepts `task` tool calls and dispatches them as Child Workflows instead of running them in-process:

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

The `subagent_specs` dict maps sub-agent type names to graph definition references in the `GraphRegistry`.

## Configuring sub-agent execution

```python
from datetime import timedelta

temporal_agent = TemporalDeepAgent(
    agent, client,
    task_queue="main-agents",
    subagent_task_queue="sub-agents",               # separate queue
    subagent_execution_timeout=timedelta(minutes=15),  # per sub-agent timeout
)
```

## How it works

1. The LLM invokes the `task` tool with a sub-agent type and instruction
2. `TemporalSubAgentMiddleware`'s tool function stores a `SubAgentRequest` in a context variable
3. The Activity collects pending requests into `NodeActivityOutput.child_workflow_requests`
4. The Workflow dispatches each request as a **Child Workflow**
5. Results flow back as `ToolMessage` entries to the parent agent

```
Parent Workflow
│
├── call_model Activity → LLM says "use task tool"
├── tools Activity → middleware stores SubAgentRequest
│
├── Child Workflow: researcher
│   └── (runs independently with own durability)
│
└── Result injected back as ToolMessage
```

## SubAgentRequest

Each sub-agent dispatch creates a `SubAgentRequest`:

```python
from deepagent_temporal import SubAgentRequest

request = SubAgentRequest(
    subagent_type="researcher",
    instruction="Find the latest pricing for AWS Lambda",
    tool_call_id="call_abc123",
    initial_state={"messages": [...]},
    graph_definition_ref="subagent:researcher",
)

# Serialization for Temporal payloads
d = request.to_dict()
restored = SubAgentRequest.from_dict(d)
```

## Benefits over in-process sub-agents

| Aspect | In-process | Child Workflow |
|---|---|---|
| Crash recovery | Sub-agent lost | Independent durability |
| Timeout | Process-level | Per-sub-agent Temporal timeout |
| Observability | None | Visible in Temporal UI |
| Resource isolation | Shared memory | Separate worker capacity |
| Parallelism | Limited by process | Distributed across workers |
