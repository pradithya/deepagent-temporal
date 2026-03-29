# deepagent-temporal: Technical Design Document

## 1. Architecture Overview

```
+--------------------------------------------------------------------+
|                       Client Application                            |
|                                                                     |
|  agent = create_deep_agent(model, tools, backend, ...)              |
|  temporal_agent = TemporalDeepAgent(agent, client, task_queue, ...) |
|  result = temporal_agent.ainvoke(input, config)                     |
+----------------------+---------------------------------------------+
                       |
                       | Temporal Client: start_workflow / signal / query
                       v
+--------------------------------------------------------------------+
|                     Temporal Server                                  |
|  +--------------------------------------------------------------+  |
|  |  Workflow: LangGraphWorkflow (reused from langgraph-temporal) |  |
|  |                                                                |  |
|  |  1. Initialize channels from Deep Agent's state schema        |  |
|  |  2. Loop:                                                      |  |
|  |     a. prepare_next_tasks (call_model or tools node)          |  |
|  |     b. Dispatch Activity via sticky queue or Session           |  |
|  |     c. Check for child_workflow_requests (sub-agents)         |  |
|  |        -> Dispatch Child Workflows for task tool calls        |  |
|  |        -> Feed results back as ToolMessages                   |  |
|  |     d. Check for tool-level interrupts (interrupt_on)         |  |
|  |        -> Wait for Signal/Update                              |  |
|  |     e. apply_writes to channel state                          |  |
|  |     f. Emit stream events                                     |  |
|  |  3. Continue-as-new at threshold                               |  |
|  +--------------------------------------------------------------+  |
|                                                                     |
|  +---------------------------+  +-----------------------------+    |
|  | Activity: call_model      |  | Activity: tools             |    |
|  | (on sticky queue/session) |  | (on sticky queue/session)   |    |
|  |                           |  |                             |    |
|  | - Middleware stack:       |  | - Execute tool calls:       |    |
|  |   Summarization          |  |   read_file, write_file,    |    |
|  |   PromptCaching          |  |   edit_file, ls, glob,      |    |
|  |   PatchToolCalls         |  |   grep, execute,            |    |
|  | - LLM invocation         |  |   write_todos, custom tools |    |
|  | - Returns AIMessage +    |  | - Detect task tool calls    |    |
|  |   tool_calls             |  |   -> Return as              |    |
|  +---------------------------+  |   child_workflow_requests   |    |
|                                 +-----------------------------+    |
|                                                                     |
|  +--------------------------------------------------------------+  |
|  | Child Workflow: Sub-Agent (per task tool invocation)           |  |
|  | ID: {parent_id}/subagent/{type}/{step}_{index}                |  |
|  |                                                                |  |
|  | - Own LangGraphWorkflow execution                              |  |
|  | - Own sticky task queue (or parent's queue)                    |  |
|  | - Returns final message as ToolMessage to parent              |  |
|  +--------------------------------------------------------------+  |
+--------------------------------------------------------------------+
                       |
                       | Sticky Task Queue
                       v
+--------------------------------------------------------------------+
|                    Temporal Worker(s)                                |
|                                                                     |
|  worker = temporal_agent.create_worker()                            |
|  - Registers LangGraphWorkflow                                      |
|  - Registers Activities (execute_node, evaluate_conditional_edge)  |
|  - Registers graph + backend in GraphRegistry                       |
|  - Initializes backend (FilesystemBackend, StoreBackend, etc.)     |
|  - Initializes model client (ChatAnthropic, etc.)                   |
+--------------------------------------------------------------------+
```

## 2. Concept Mapping: Deep Agents -> Temporal

| Deep Agents Concept | Temporal Concept | Notes |
|---|---|---|
| `create_deep_agent()` return value (`CompiledStateGraph`) | Workflow Definition (via `TemporalGraph`) | Standard LangGraph-to-Temporal mapping |
| `call_model` node (LLM + middleware) | Activity `execute_node("call_model")` | Entire middleware stack runs within Activity |
| `tools` node (tool dispatch) | Activity `execute_node("tools")` | All tool calls in one LLM turn run within single Activity |
| `task` tool (sub-agent spawn) | Child Workflow dispatch | Detected in Activity output, dispatched by Workflow |
| `SubAgentMiddleware` | `TemporalSubAgentMiddleware` | Replaces in-process invocation with Child Workflow request |
| `interrupt_on` (tool-level HITL) | Activity interrupt result + Workflow Signal/Update wait | Different from node-level `interrupt_before`/`interrupt_after` |
| `BackendProtocol` (filesystem, state, store) | Worker-local resource, accessed via sticky task queue | Backend bound into compiled graph's tool closures at worker startup |
| `SummarizationMiddleware` | Part of `call_model` Activity | Runs within same Activity as LLM call |
| `AnthropicPromptCachingMiddleware` | Part of `call_model` Activity | Cache hints are transient; validated for serialization |
| `PatchToolCallsMiddleware` | Part of `call_model` Activity | UUID generation safe within Activity (non-deterministic boundary) |
| `TodoListMiddleware` + `write_todos` tool | Part of middleware/tools Activities + state channel | `todos` channel persisted in Workflow state |
| `skills_metadata`, `memory_contents` | Private state channels | Excluded from sub-agent state, preserved in continue-as-new |
| `recursion_limit=1000` (Deep Agents default) | Workflow step counter + continue-as-new at 500 steps | Multiple continue-as-new runs within a single agent task |

## 3. Component Design

### 3.1 Package Structure

```
deepagent-temporal/
  deepagent_temporal/
    __init__.py                  # Public API: TemporalDeepAgent, config types
    agent.py                     # TemporalDeepAgent wrapper (composes TemporalGraph)
    middleware.py                 # TemporalSubAgentMiddleware (replaces SubAgentMiddleware)
    config.py                    # SessionOptions, AffinityMode, SessionRecoveryStrategy
    _dispatch.py                 # ActivityDispatcher protocol + implementations
    _interrupt.py                # Tool-level interrupt handling
```

**Dependencies:**
- `deepagents >= 0.5`
- `langgraph-temporal >= 0.1`
- `temporalio >= 1.7`

### 3.2 TemporalDeepAgent (Primary Entry Point)

`TemporalDeepAgent` accepts a pre-compiled `Pregel` graph (the output of
`create_deep_agent()`) and wraps it with `TemporalGraph` from
`langgraph-temporal`, delegating all standard operations (invoke, stream,
get_state, resume). A `create_temporal_deep_agent()` convenience factory
is also provided.

**Important**: If the agent uses sub-agents via the `task` tool, the user
must inject `TemporalSubAgentMiddleware` **before** graph compilation.
Middleware is baked into the compiled graph's node callables and cannot
be patched after compilation. This keeps the `deepagent_temporal` package
free of a hard dependency on `deepagents` (see NG-01).

```python
# deepagent_temporal/agent.py

from langgraph.temporal import TemporalGraph
from langgraph.temporal.config import SubAgentConfig
from deepagent_temporal.middleware import TemporalSubAgentMiddleware

def create_temporal_deep_agent(
    agent: Pregel,
    client: TemporalClient,
    *,
    # Temporal-specific params
    task_queue: str = "deep-agents",
    sticky_task_queue: str | None = None,
    subagent_task_queue: str | None = None,
    subagent_execution_timeout: timedelta = timedelta(minutes=30),
    workflow_execution_timeout: timedelta | None = None,
    workflow_run_timeout: timedelta | None = None,
    stream_backend: StreamBackend | None = None,
) -> "TemporalDeepAgent":
    """Create a TemporalDeepAgent from a pre-compiled Deep Agent graph.

    The caller is responsible for injecting TemporalSubAgentMiddleware
    into the middleware stack before calling create_deep_agent().
    """
    return TemporalDeepAgent(
        agent, client,
        task_queue=task_queue,
        sticky_task_queue=sticky_task_queue,
        subagent_task_queue=subagent_task_queue,
        subagent_execution_timeout=subagent_execution_timeout,
        workflow_execution_timeout=workflow_execution_timeout,
        workflow_run_timeout=workflow_run_timeout,
        stream_backend=stream_backend,
    )


class TemporalDeepAgent:
    """Wraps a Deep Agent for durable execution on Temporal.

    Composes TemporalGraph for standard LangGraph-to-Temporal mapping,
    adding Deep Agent-specific behavior:
    - Worker affinity via sticky task queues
    - Sub-agent dispatch via Child Workflows
    - Tool-level human-in-the-loop via interrupt detection

    Use `create_temporal_deep_agent()` factory to construct instances.
    """

    def __init__(
        self,
        agent: CompiledStateGraph,
        client: TemporalClient,
        *,
        task_queue: str = "deep-agents",
        sticky_task_queue: str | None = None,
        subagent_task_queue: str | None = None,
        subagent_execution_timeout: timedelta = timedelta(minutes=30),
        workflow_execution_timeout: timedelta | None = None,
        workflow_run_timeout: timedelta | None = None,
        stream_backend: StreamBackend | None = None,
    ):
        self._temporal_graph = TemporalGraph(
            agent,
            client,
            task_queue=task_queue,
            workflow_execution_timeout=workflow_execution_timeout,
            workflow_run_timeout=workflow_run_timeout,
            stream_backend=stream_backend,
        )
        self._sticky_task_queue = sticky_task_queue
        self._subagent_task_queue = subagent_task_queue or task_queue
        self._subagent_execution_timeout = subagent_execution_timeout

    async def ainvoke(self, input, config=None, **kwargs):
        """Execute the Deep Agent as a Temporal Workflow."""
        config = self._inject_temporal_config(config)
        return await self._temporal_graph.ainvoke(input, config, **kwargs)

    async def astream(self, input, config=None, **kwargs):
        """Stream Deep Agent execution events."""
        config = self._inject_temporal_config(config)
        async for event in self._temporal_graph.astream(input, config, **kwargs):
            yield event

    async def astart(self, input, config=None):
        """Start a Deep Agent Workflow (non-blocking)."""
        config = self._inject_temporal_config(config)
        return await self._temporal_graph.astart(input, config)

    def get_state(self, config):
        """Query current agent state."""
        return self._temporal_graph.get_state(config)

    async def resume(self, config, value):
        """Send a resume Signal for HITL approval."""
        return await self._temporal_graph.resume(config, value)

    def create_worker(self, **kwargs):
        """Create a Temporal Worker configured for Deep Agent execution.

        Pre-compiles and registers all sub-agent graphs in GraphRegistry
        so they are available for Child Workflow dispatch.
        """
        # Pre-register sub-agent graphs
        self._register_subagent_graphs()
        return self._temporal_graph.create_worker(**kwargs)

    @classmethod
    async def local(cls, agent, **kwargs):
        """Factory for local development with Temporal test server."""
        tg = await TemporalGraph.local(agent)
        instance = cls.__new__(cls)
        instance._temporal_graph = tg
        instance._sticky_task_queue = None
        instance._subagent_task_queue = tg.task_queue
        instance._subagent_execution_timeout = timedelta(minutes=30)
        return instance

    def _inject_temporal_config(self, config):
        """Add affinity and sub-agent config to config["configurable"].

        TemporalGraph._build_workflow_input reads from configurable dict.
        We add sticky_task_queue and subagent_config there so they flow
        into WorkflowInput (requires upstream change to TemporalGraph).
        """
        config = config or {}
        configurable = config.get("configurable", {})
        if self._sticky_task_queue:
            configurable["sticky_task_queue"] = self._sticky_task_queue
        configurable["subagent_config"] = {
            "task_queue": self._subagent_task_queue,
            "execution_timeout": self._subagent_execution_timeout,
        }
        config["configurable"] = configurable
        return config
```

### 3.3 Worker Affinity via Sticky Task Queue

**Important**: The Temporal Python SDK (`temporalio`) does NOT support
`workflow.start_session()`. Sessions are only available in Go and Java SDKs.
Worker affinity in Python must use **sticky task queues** -- a dedicated task
queue per workspace, served by a dedicated worker.

#### 3.3.1 How Sticky Task Queues Work

1. The user configures a `sticky_task_queue` (e.g., `"deep-agent-workspace-42"`)
2. A worker is started that polls this specific queue
3. The Workflow dispatches all Activities to this queue
4. All Activities execute on the same worker, ensuring backend affinity
5. The sticky queue name survives continue-as-new (it is part of `WorkflowInput`)
6. HITL waits have no session expiry concern -- the queue persists independently

```python
# Inside LangGraphWorkflow (upstream change to langgraph-temporal)

@workflow.run
async def run(self, input: WorkflowInput) -> WorkflowOutput:
    # Determine the Activity task queue for affinity
    if input.sticky_task_queue:
        self._activity_task_queue = input.sticky_task_queue
    else:
        self._activity_task_queue = None  # Use default per-node routing

    # ... rest of execution loop ...
```

#### 3.3.2 Activity Dispatch with Affinity

The existing `_task_queue_for_node()` method is modified to respect
the sticky queue as a fallback (single-point change that flows to all
dispatch sites):

```python
def _task_queue_for_node(self, node_name: str) -> str:
    """Get task queue for a node, respecting affinity configuration.

    Precedence:
    1. Sticky task queue (if configured) -- overrides everything for affinity
    2. Per-node task queue override (from node_task_queues config)
    3. Workflow's default task queue
    """
    if self._activity_task_queue:
        return self._activity_task_queue
    if node_name in self._input.node_task_queues:
        return self._input.node_task_queues[node_name]
    return workflow.info().task_queue
```

#### 3.3.3 Limitation: Queue Routing, Not Session Pinning

Sticky task queues provide **queue-level routing**, not **worker-level
pinning**. If multiple workers poll the same sticky queue, Temporal may
distribute Activities across them — there is no guarantee that all
Activities for a workflow land on the same worker. True worker pinning
requires Temporal Sessions (`workflow.start_session()`), which are only
available in the Go and Java SDKs.

The intended deployment model is **one dedicated worker per sticky queue**
(e.g., one worker per workspace). The framework guarantees Activities are
dispatched to the correct queue; the operator guarantees only one worker
serves it. When Session support is added to the Python SDK, it should be
adopted to provide real worker pinning with multiple workers per queue.

#### 3.3.4 No-Affinity Mode (for StateBackend)

When using `StateBackend` (in-memory state stored in LangGraph channels),
no worker affinity is needed. State is serialized through Temporal and
reconstructed on any worker. This is the default when `sticky_task_queue`
is not set.

#### 3.3.5 Worker Failure Recovery

With sticky task queues, if the dedicated worker dies:
- Activities sit in the queue until the worker restarts
- `schedule_to_close_timeout` provides bounded failure detection
- For Kubernetes deployments: use PersistentVolume + pod restart policy
  to ensure the worker restarts on the same node with the same filesystem
- For non-recoverable failures: the Workflow times out and can be retried
  with a Workflow-level RetryPolicy

### 3.4 Sub-Agent Dispatch via Child Workflows

#### 3.4.1 TemporalSubAgentMiddleware

Replaces `SubAgentMiddleware` to intercept `task` tool calls and store them
in a **context variable** instead of invoking sub-agents in-process. This
avoids the problem of returning a non-string from a tool (which would break
LangGraph's `ToolNode` contract).

```python
# deepagent_temporal/middleware.py

import contextvars
from dataclasses import dataclass, field

# Context variable to collect SubAgentRequests during Activity execution.
# The Activity reads this after node execution completes.
_pending_subagent_requests: contextvars.ContextVar[list] = (
    contextvars.ContextVar("_pending_subagent_requests", default=[])
)

@dataclass
class SubAgentRequest:
    """Request for a sub-agent Child Workflow dispatch.

    Stored in context variable during tool execution; collected by the
    Activity and returned in NodeActivityOutput.child_workflow_requests.

    All fields must be JSON-serializable for Temporal payload conversion.
    """
    subagent_type: str
    instruction: str
    tool_call_id: str
    # State context (excluded keys removed, messages replaced with prompt)
    # Serialized via langgraph-temporal's StateSerializer
    initial_state: dict[str, Any]
    # Reference to pre-registered sub-agent graph in GraphRegistry
    graph_definition_ref: str

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict for Temporal payload."""
        return {
            "subagent_type": self.subagent_type,
            "instruction": self.instruction,
            "tool_call_id": self.tool_call_id,
            "initial_state": self.initial_state,
            "graph_definition_ref": self.graph_definition_ref,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SubAgentRequest":
        return cls(**d)


class TemporalSubAgentMiddleware(SubAgentMiddleware):
    """Temporal-aware replacement for SubAgentMiddleware.

    The `task` tool stores a SubAgentRequest in a context variable and
    returns a placeholder ToolMessage. The Activity collects pending
    requests after execution and includes them in the output.
    """

    def _build_task_tool(self, subagents, runtime):
        """Override: store SubAgentRequest in context var, return placeholder."""

        async def temporal_task(
            instruction: str,
            subagent_type: str = "general-purpose",
        ) -> str:
            # Build initial state (same logic as SubAgentMiddleware)
            parent_state = runtime.state
            child_state = {
                k: v for k, v in parent_state.items()
                if k not in _EXCLUDED_STATE_KEYS
            }
            child_state["messages"] = [
                {"role": "user", "content": instruction}
            ]

            # Find matching subagent spec -> graph_definition_ref
            spec = self._find_subagent(subagent_type)
            graph_ref = spec.get("graph_ref", f"subagent:{subagent_type}")

            # Store request in context variable
            req = SubAgentRequest(
                subagent_type=subagent_type,
                instruction=instruction,
                tool_call_id=runtime.tool_call_id,
                initial_state=child_state,
                graph_definition_ref=graph_ref,
            )
            _pending_subagent_requests.get().append(req)

            # Return placeholder -- will be replaced by Child Workflow result
            return f"[Sub-agent '{subagent_type}' dispatched as Child Workflow]"

        return StructuredTool.from_function(
            temporal_task,
            name="task",
            description=TASK_TOOL_DESCRIPTION.format(
                available_agents=self._format_agents()
            ),
        )
```

#### 3.4.2 Activity-Side Collection

The Activity collects `SubAgentRequest` objects from the context variable
after node execution completes. This uses a generic `ChildWorkflowRequest`
protocol in `langgraph-temporal` to avoid a reverse dependency:

```python
# langgraph-temporal: Generic protocol (no dependency on deepagent types)
class ChildWorkflowRequest(Protocol):
    """Protocol for runtime Child Workflow dispatch requests."""
    def to_dict(self) -> dict: ...

# deepagent-temporal: Activity wrapper that collects context var
# (wraps langgraph-temporal's execute_node)

async def execute_deep_agent_node(input: NodeActivityInput) -> NodeActivityOutput:
    """Execute a Deep Agent node, collecting sub-agent requests."""
    # Reset context var for this Activity execution
    _pending_subagent_requests.set([])

    # Delegate to langgraph-temporal's execute_node
    result = await _execute_node_impl(input)

    # Collect any SubAgentRequests stored during tool execution
    pending = _pending_subagent_requests.get()
    if pending:
        result.child_workflow_requests = [
            req.to_dict() for req in pending
        ]

    return result
```

**Key design decision**: The context variable approach avoids:
1. Returning non-string from a tool (breaks ToolNode contract)
2. Importing `SubAgentRequest` in `langgraph-temporal` (reverse dependency)
3. Scanning channel writes for marker types (fragile)

#### 3.4.3 Workflow-Side Child Workflow Dispatch

The Workflow processes `child_workflow_requests` after receiving Activity
results:

```python
# Extension to langgraph-temporal/workflow.py (in the main loop)

results = await self._execute_nodes(next_tasks, graph_def, input)

# Check for sub-agent requests
for result in results:
    if result.child_workflow_requests:
        # Dispatch Child Workflows for all sub-agent requests
        subagent_results = await self._dispatch_subagent_child_workflows(
            result.child_workflow_requests, input
        )
        # Inject ToolMessage results back into the writes
        for req, subagent_output in zip(
            result.child_workflow_requests, subagent_results
        ):
            if isinstance(subagent_output, Exception):
                content = f"Sub-agent '{req['subagent_type']}' failed: {subagent_output}"
            else:
                content = subagent_output.channel_values.get(
                    "messages", [""]
                )[-1]  # Last message is the sub-agent's result
            result.writes.append((
                "messages",
                ToolMessage(
                    content=content,
                    tool_call_id=req["tool_call_id"],
                ),
            ))

async def _dispatch_subagent_child_workflows(
    self, requests: list[dict], input: WorkflowInput
) -> list[WorkflowOutput | Exception]:
    """Dispatch Child Workflows for sub-agent requests.

    Uses return_exceptions=True to handle partial failures gracefully.
    Failed sub-agents return error ToolMessages rather than crashing
    the parent Workflow.
    """
    results = await asyncio.gather(
        *[
            self._dispatch_single_subagent(req, input, idx)
            for idx, req in enumerate(requests)
        ],
        return_exceptions=True,
    )

async def _dispatch_single_subagent(
    self, req: SubAgentRequest, input: WorkflowInput, index: int
) -> WorkflowOutput:
    """Dispatch a single sub-agent as a Child Workflow."""
    child_wf_id = (
        f"{workflow.info().workflow_id}"
        f"/subagent/{req.subagent_type}"
        f"/{self.step}_{index}"
    )

    child_input = WorkflowInput(
        graph_definition_ref=req.subagent_spec.get("graph_ref"),
        input_data=req.initial_state,
        recursion_limit=input.recursion_limit,
        # Sub-agent may have its own affinity
        sticky_task_queue=input.subagent_config.get("sticky_task_queue"),
        session_options=input.subagent_config.get("session_options"),
    )

    try:
        return await workflow.execute_child_workflow(
            LangGraphWorkflow.run,
            child_input,
            id=child_wf_id,
            task_queue=input.subagent_config.get(
                "task_queue", workflow.info().task_queue
            ),
            execution_timeout=input.subagent_config.get(
                "execution_timeout", timedelta(minutes=30)
            ),
            parent_close_policy=ParentClosePolicy.TERMINATE,
            cancellation_type=(
                ChildWorkflowCancellationType.WAIT_CANCELLATION_COMPLETED
            ),
        )
    except Exception as e:
        # Convert failure to error ToolMessage (matching SubAgentMiddleware)
        return WorkflowOutput(
            channel_values={"messages": [
                f"Sub-agent '{req.subagent_type}' failed: {e}"
            ]},
            step=0,
        )
```

### 3.5 Tool-Level Interrupt Handling

Deep Agents' `interrupt_on` operates at the tool level (e.g.,
`interrupt_on={"edit_file": True}`), which is different from
`langgraph-temporal`'s node-level `interrupt_before`/`interrupt_after`.

The tool-level interrupt must be detected inside the `tools` node Activity
and propagated to the Workflow:

```python
# deepagent_temporal/_interrupt.py

@dataclass
class ToolInterrupt:
    """Marker for tool-level interrupt detected in Activity."""
    tool_name: str
    tool_args: dict[str, Any]
    tool_call_id: str
    # Partial writes from tools that executed before the interrupt
    partial_writes: list[tuple[str, Any]]


# Inside the tools node Activity (via HumanInTheLoopMiddleware):
# When interrupt_on is triggered, the existing LangGraph interrupt()
# mechanism raises GraphInterrupt. The Activity catches this and
# returns it as an interrupt result -- this is already handled by
# langgraph-temporal's execute_node Activity.
#
# The key insight: Deep Agents' HumanInTheLoopMiddleware uses
# LangGraph's standard interrupt() function. So the existing
# langgraph-temporal interrupt handling (GraphInterrupt -> Activity
# returns interrupt payload -> Workflow waits for Signal) works
# out of the box.
#
# The only addition needed: the interrupt payload must include
# tool-level details (tool name, args) so the Query handler can
# expose them to the approval UI.
```

**Key insight**: Deep Agents' `HumanInTheLoopMiddleware` internally uses
LangGraph's standard `interrupt()` function. This means the existing
`langgraph-temporal` interrupt handling already works:

1. `HumanInTheLoopMiddleware` calls `interrupt(tool_call_details)`
2. `GraphInterrupt` is raised inside the Activity
3. Activity returns `NodeActivityOutput` with `interrupts` field
4. Workflow stores interrupts and waits for `resume_signal`
5. On resume, Workflow re-executes the Activity with resume values

The only addition needed is ensuring the interrupt payload includes tool-level
details (tool name, arguments) so the `get_current_state` Query exposes them.

### 3.6 State Serialization Considerations

#### 3.6.1 Middleware Metadata Sanitization

`AnthropicPromptCachingMiddleware` adds `cache_control` dicts to messages.
These must survive Temporal serialization:

```python
# Validation: cache_control is a simple dict {"type": "ephemeral"}
# which is JSON-serializable and compatible with Temporal's default
# PayloadConverter. No special handling needed.
#
# However, if future middleware adds non-serializable metadata, a
# sanitization hook should strip it before checkpoint serialization.
```

#### 3.6.2 Large State Payloads

Deep Agents with long conversation histories can exceed Temporal's 2MB payload
limit. The integration reuses `LargePayloadCodec` from `langgraph-temporal`:

```python
from langgraph.temporal._codec import LargePayloadCodec

# Configure on the Temporal Client
client = await Client.connect(
    "localhost:7233",
    data_converter=DataConverter(
        payload_codec=LargePayloadCodec(
            store=S3BlobStore(bucket="agent-payloads"),
            threshold_bytes=2_000_000,
        ),
    ),
)
```

#### 3.6.3 UntrackedValue and Private State

Deep Agents uses `PrivateStateAttr` annotations for `skills_metadata` and
`memory_contents`. The existing `langgraph-temporal` `UntrackedValue` filtering
must be verified to handle these correctly. If they are implemented as
`UntrackedValue` channels, they are automatically excluded from Temporal
history. If not, explicit filtering is needed.

### 3.7 Continue-As-New

Deep Agents sets `recursion_limit=1000` by default. With
`langgraph-temporal`'s default `CONTINUE_AS_NEW_THRESHOLD=500`, an agent task
will trigger at least one continue-as-new. The restored state must include:

1. Full channel state (messages, todos, files, etc.)
2. `channel_versions` and `versions_seen`
3. Step counter
4. Pending interrupts (if any)
5. `sticky_task_queue` name (preserved across runs for affinity)
6. `subagent_config` (preserved for Child Workflow dispatch)
7. `skills_metadata` and `memory_contents` (may need `LargePayloadCodec`)

```python
# Continue-as-new input (extension to langgraph-temporal's RestoredState)

@dataclass
class RestoredState:
    channels: dict[str, Any]
    checkpoint: Checkpoint
    step: int
    # New field for Deep Agent affinity
    sticky_task_queue: str | None = None  # Preserved across runs
```

### 3.8 Worker Setup

```python
# worker.py (run on each Deep Agent worker machine)

from deepagent_temporal import TemporalDeepAgent
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from temporalio.client import Client

async def main():
    client = await Client.connect("temporal.mycompany.com:7233")

    # Create the Deep Agent (same as client-side)
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[my_custom_tool],
        backend=FilesystemBackend(root_dir="/workspace"),
    )

    # Wrap with Temporal
    temporal_agent = TemporalDeepAgent(
        agent,
        client,
        task_queue="deep-agents",
        affinity_mode=AffinityMode.STICKY_QUEUE,
    )

    # Create and run worker
    # The worker registers:
    # - LangGraphWorkflow (from langgraph-temporal)
    # - execute_node, dynamic_execute_node, evaluate_conditional_edge
    # - The agent's compiled graph in GraphRegistry
    # - The backend in a BackendRegistry
    worker = temporal_agent.create_worker(
        max_concurrent_activities=10,
        max_concurrent_workflow_tasks=100,
    )
    await worker.run()
```

## 4. Required Upstream Changes to `langgraph-temporal`

All changes are backward-compatible (new optional fields with `None` defaults).

### 4.1 `config.py` -- WorkflowInput

```python
@dataclass
class WorkflowInput:
    # ... existing fields ...

    # New: Worker affinity via sticky task queue
    sticky_task_queue: str | None = None

    # New: Sub-agent configuration
    subagent_config: SubAgentConfig | None = None

@dataclass
class SubAgentConfig:
    task_queue: str | None = None
    sticky_task_queue: str | None = None
    execution_timeout: timedelta = timedelta(minutes=30)
```

### 4.2 `config.py` -- NodeActivityOutput

```python
@dataclass
class NodeActivityOutput:
    # ... existing fields ...

    # New: Child Workflow requests for runtime sub-agent dispatch.
    # Uses list[dict] (not a specific type) to avoid coupling
    # langgraph-temporal to deepagent-temporal types.
    child_workflow_requests: list[dict] | None = None
```

### 4.3 `config.py` -- RestoredState

```python
@dataclass
class RestoredState:
    # ... existing fields ...

    # New: Sticky task queue preserved across continue-as-new
    sticky_task_queue: str | None = None
```

### 4.4 `workflow.py` -- Task Queue Routing with Affinity

Modify `_task_queue_for_node()` to respect sticky queue (single-point change
that flows to all Activity dispatch sites):

```python
@workflow.defn
class LangGraphWorkflow:
    # ... existing code ...

    def _task_queue_for_node(self, node_name: str) -> str:
        """Get task queue, respecting sticky affinity.

        Precedence:
        1. Sticky task queue (overrides per-node routing for affinity)
        2. Per-node task queue (from node_task_queues config)
        3. Workflow's default task queue
        """
        if self._activity_task_queue:  # Set from input.sticky_task_queue
            return self._activity_task_queue
        if node_name in (self._input.node_task_queues or {}):
            return self._input.node_task_queues[node_name]
        return workflow.info().task_queue
```

### 4.5 `workflow.py` -- Child Workflow Request Processing

Add processing for `child_workflow_requests` after Activity execution,
inserted between `_execute_nodes` and interrupt handling in the main loop:

```python
# In the main loop, after _execute_nodes but before interrupt handling:

for result in results:
    if result.child_workflow_requests:
        subagent_results = await self._dispatch_child_workflow_requests(
            result.child_workflow_requests
        )
        # Inject results back as channel writes at original positions
        for req, output in zip(
            result.child_workflow_requests, subagent_results
        ):
            if isinstance(output, Exception):
                content = f"Sub-agent failed: {output}"
            else:
                msgs = output.channel_values.get("messages", [])
                content = msgs[-1] if msgs else ""
            result.writes.append((
                "messages",
                serialize_tool_message(req["tool_call_id"], content),
            ))
```

### 4.6 `workflow.py` -- Continue-As-New Must Forward New Fields

The continue-as-new `WorkflowInput` construction must include new fields:

```python
if steps_in_this_run >= CONTINUE_AS_NEW_THRESHOLD:
    workflow.continue_as_new(
        WorkflowInput(
            # ... existing fields ...
            sticky_task_queue=input.sticky_task_queue,  # Preserve affinity
            subagent_config=input.subagent_config,      # Preserve sub-agent config
            restored_state=RestoredState(
                # ... existing fields ...
                sticky_task_queue=input.sticky_task_queue,
            ),
        )
    )
```

### 4.7 `graph.py` -- WorkflowInput Construction Extension

`TemporalGraph._build_workflow_input()` must read new fields from
`config["configurable"]`:

```python
def _build_workflow_input(self, input_data, config, **kwargs):
    configurable = config.get("configurable", {})
    return WorkflowInput(
        # ... existing fields ...
        sticky_task_queue=configurable.get("sticky_task_queue"),
        subagent_config=configurable.get("subagent_config"),
    )
```

## 5. Execution Flow

### 5.1 Normal Agent Step (LLM Call -> Tool Execution)

```
Workflow                    Worker (Activity)
   |                              |
   |-- execute_activity --------->|
   |   (call_model node)          |
   |                              |-- Middleware stack:
   |                              |   1. SummarizationMiddleware
   |                              |      (may compactmessages)
   |                              |   2. PromptCachingMiddleware
   |                              |      (add cache hints)
   |                              |   3. LLM call (ChatAnthropic)
   |                              |   4. PatchToolCallsMiddleware
   |                              |      (fix tool call IDs)
   |                              |
   |<-- NodeActivityOutput -------|
   |   (writes: AIMessage +       |
   |    tool_calls)               |
   |                              |
   |-- apply_writes ------------->|
   |                              |
   |-- execute_activity --------->|
   |   (tools node)               |
   |                              |-- Execute tool calls:
   |                              |   read_file, write_file, etc.
   |                              |   (on same worker via affinity)
   |                              |
   |<-- NodeActivityOutput -------|
   |   (writes: ToolMessages)     |
   |                              |
   |-- apply_writes               |
   |-- emit_stream_events         |
   |-- prepare_next_tasks (loop)  |
```

### 5.2 Sub-Agent Dispatch

```
Workflow                    Worker (Activity)         Child Workflow
   |                              |                        |
   |-- execute_activity --------->|                        |
   |   (tools node)               |                        |
   |                              |-- Execute tool calls   |
   |                              |   task("Research X",   |
   |                              |    type="researcher")  |
   |                              |                        |
   |<-- NodeActivityOutput -------|                        |
   |   (child_workflow_requests:  |                        |
   |    [SubAgentRequest(...)])   |                        |
   |                              |                        |
   |-- execute_child_workflow ----|----------------------->|
   |   ID: parent/subagent/       |                        |
   |       researcher/5_0         |                        |
   |                              |                        |-- Agent loop
   |                              |                        |   (own Activities)
   |                              |                        |
   |<----- WorkflowOutput -------|------------------------|
   |   (final_message)            |                        |
   |                              |                        |
   |-- Inject ToolMessage         |                        |
   |   into writes                |                        |
   |-- apply_writes               |                        |
```

### 5.3 Human-in-the-Loop (Tool Approval)

```
Workflow                    Worker (Activity)           Client
   |                              |                        |
   |-- execute_activity --------->|                        |
   |   (tools node)               |                        |
   |                              |-- HumanInTheLoop       |
   |                              |   middleware detects    |
   |                              |   edit_file tool call   |
   |                              |-- interrupt(details)    |
   |                              |   -> GraphInterrupt     |
   |                              |                        |
   |<-- NodeActivityOutput -------|                        |
   |   (interrupts: [{tool:       |                        |
   |    "edit_file", args: ...}]) |                        |
   |                              |                        |
   |-- status = "interrupted"     |                        |
   |-- Wait for Signal            |                        |
   |   (zero resources)           |                        |
   |                              |                        |
   |                              |         Query: get_current_state
   |<----------------------------------------------------- |
   |   {status: "interrupted",    |                        |
   |    interrupts: [{tool:       |                        |
   |    "edit_file", args: ...}]} |                        |
   |                              |                        |
   |                              |         Signal: resume("approved")
   |<----------------------------------------------------- |
   |                              |                        |
   |-- Re-execute Activity ------>|                        |
   |   (with resume_values)       |                        |
   |                              |-- Edit file proceeds   |
   |<-- NodeActivityOutput -------|                        |
```

## 6. Configuration Reference

### 6.1 Default Activity Configuration

| Node | `start_to_close_timeout` | `schedule_to_close_timeout` | `heartbeat_timeout` | Retry |
|------|--------------------------|----------------------------|---------------------|-------|
| `call_model` | 5 min | 15 min | 60s | Yes: max 3 attempts, 5s initial, 2x backoff. Retries on rate limits, network errors. |
| `tools` | 30 min | 60 min | 60s | No (side effects not idempotent) |

### 6.2 Idempotency Key

Tools that need idempotency (e.g., `execute`) can derive a key from
Activity info available at runtime:

```python
# Inside Activity execution
idempotency_key = f"{activity.info().workflow_id}/{input.task_path}/{tool_name}"
```

### 6.3 Search Attributes

The Workflow sets the following search attributes for observability:

```python
workflow.upsert_search_attributes({
    "agent_name": agent_name,
    "thread_id": workflow.info().workflow_id,
    "current_step": str(self.step),
    "sub_agent_count": len(active_child_workflows),
})
```

### 6.4 Workflow ID and Reuse Policy

- Workflow ID = `thread_id` from config
- `WorkflowIDReusePolicy.ALLOW_DUPLICATE` -- allows re-running the same thread
- `WorkflowIDConflictPolicy.FAIL` -- prevents concurrent runs on same thread

## 7. Testing Strategy

### 7.1 Unit Tests

- `TemporalDeepAgent` construction and configuration injection.
- `TemporalSubAgentMiddleware` `task` tool returns `SubAgentRequest`.
- `SubAgentRequest` serialization/deserialization.
- Affinity mode selection and task queue computation.
- Interrupt payload construction with tool-level details.

### 7.2 Integration Tests (with Temporal Test Server)

- **Basic agent execution**: `create_deep_agent()` + `TemporalDeepAgent` runs
  a simple prompt-response cycle.
- **Tool execution**: Agent reads/writes files via backend within Session.
- **Sub-agent dispatch**: Agent spawns sub-agent via `task` tool; sub-agent
  runs as Child Workflow; result returned as ToolMessage.
- **Parallel sub-agents**: Multiple `task` calls in one LLM response dispatch
  concurrent Child Workflows.
- **HITL approval**: `interrupt_on={"edit_file": True}` pauses, resumes on
  Signal.
- **Worker crash recovery**: Kill worker mid-Activity; restart; verify
  workflow resumes.
- **Continue-as-new**: Agent with 600+ steps triggers continue-as-new;
  verify state preservation.
- **Sticky queue affinity**: Verify consecutive Activities land on same worker.

### 7.3 Error Path Tests

- Sub-agent Child Workflow failure -> error ToolMessage to parent
- LLM rate limit -> Activity retry with backoff
- Backend unavailable at worker startup -> fail-fast error
- `ContextOverflowError` -> retry triggers summarization
- Activity timeout exceeded -> Workflow handles timeout error
- Partial sub-agent failure (1 of 3 fails) -> mixed results

### 7.4 Serialization Round-Trip Tests

- All Deep Agent state types survive Temporal JSON serialization
- `cache_control` metadata from PromptCachingMiddleware serializes correctly
- Large message histories with `LargePayloadCodec`
- `skills_metadata` and `memory_contents` across continue-as-new

### 7.5 Limitations of Test Server

- `UnsandboxedWorkflowRunner()` is required (Deep Agents imports restricted
  modules).
- Sticky queue tests should verify Activity affinity by checking worker
  identity across multiple Activities.

## 8. Open Questions

### Q1: SubAgentRequest Detection Point

Where exactly in the Activity should `SubAgentRequest` be detected?

**Option A**: In `execute_node`, after all writers run, scan writes for
`SubAgentRequest` instances. Simple but couples Activity to Deep Agent types.

**Option B**: In a post-processing hook on `NodeActivityOutput`. Cleaner
separation but requires a new extension point in `langgraph-temporal`.

**Recommendation**: Option A for v0.1 (pragmatic), refactor to Option B in v0.2.

### Q2: Sub-Agent Graph Registration

Each sub-agent type needs a compiled graph registered in `GraphRegistry` on
the worker. Deep Agents creates these lazily in `SubAgentMiddleware`. For
Temporal, they must be pre-registered.

**Resolution**: `TemporalDeepAgent.create_worker()` iterates the sub-agent
specs and pre-compiles/registers each sub-agent graph in the registry.

### Q3: Backend Injection into Activities

Activities need access to the backend instance (for file operations). The
current `GraphRegistry` stores compiled graphs. Should backends be stored
in the same registry or a separate one?

**Resolution**: Extend `GraphRegistry` to also store backend references, or
create a `BackendRegistry` with the same pattern. The backend is registered
on the worker at startup and looked up by reference in Activities.

### Q4: Workflow Versioning for Agent Updates

When the Deep Agent's model, tools, or middleware change, in-flight Workflows
may break during replay.

**Resolution**: Use Temporal Worker Versioning (Build IDs) to route in-flight
Workflows to workers running the old agent configuration. New Workflows use
the updated configuration. Document that users should use Build ID-based
versioning for production deployments.

### Q5: In-Flight Child Workflows During Continue-As-New

If the parent Workflow triggers continue-as-new while Child Workflows (sub-agents)
are still running, what happens?

**Resolution**: Continue-as-new should only trigger between steps, after all
Activities and Child Workflows for the current step have completed. The existing
`langgraph-temporal` threshold check happens at the end of each step, so this
is naturally satisfied. However, this should be validated with a test.

### Q6: Streaming Architecture for Sub-Agents

FR-07.3 requires sub-agent progress streaming. Child Workflow events do not
automatically bubble up through the parent's stream.

**Proposed resolution**: The parent Workflow polls Child Workflow state via
`workflow.get_external_workflow_handle().query()` and emits sub-agent progress
events to its own stream buffer. This adds polling overhead but avoids
complex cross-workflow streaming infrastructure. Alternatively, sub-agents
can publish to the same `StreamBackend` (e.g., Redis) as the parent, keyed
by parent Workflow ID.

### Q7: `SummarizationMiddleware` and Continue-As-New Interaction

If summarization triggers mid-conversation, the compacted messages must be
the ones carried across continue-as-new, not the original full history.

**Resolution**: This is naturally handled because summarization modifies the
`messages` channel in-place during the `call_model` Activity. The modified
state (with compacted messages) is what gets applied via `apply_writes` and
stored in the Workflow's channel state. When continue-as-new serializes the
channel state, it gets the compacted version. However, this should be
validated with an integration test.

## 9. Deferred Features (v0.2+)

### 9.1 Per-Tool Activity Decomposition

v0.1 executes all tool calls within a single `tools` node Activity. v0.2
may decompose this so each tool call is a separate Activity, providing:
- Per-tool visibility in Temporal UI
- Per-tool retry policies (retry read-only tools, skip side-effect tools)
- Per-tool timeout configuration
- This requires modifying the Deep Agent graph structure at compilation time.

### 9.2 Temporal Workflow Updates for HITL

Replace Signals with Workflow Updates (Temporal 1.10+) for HITL approval.
Updates provide synchronous request-response semantics, letting the approval
UI confirm receipt of the approval.

### 9.3 Graceful Degradation

If Temporal is unavailable, fall back to non-durable in-process execution.
This would ease adoption by allowing gradual migration.
