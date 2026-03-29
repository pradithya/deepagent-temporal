# deepagent-temporal: Requirements Document

## 1. Problem Statement

Deep Agents (from `langchain-ai/deepagents`) is a batteries-included agent
harness built on LangGraph. It provides planning, filesystem operations, shell
execution, sub-agent spawning, and context management out of the box via
`create_deep_agent()`. However, Deep Agents inherits the same durability and
scalability limitations as vanilla LangGraph:

1. **No durable execution for long-running agent tasks.** Deep Agents can run
   complex, multi-step workflows that span minutes to hours (research tasks,
   code generation, multi-file refactoring). If the process crashes mid-task,
   all progress is lost. The agent must restart from scratch.

2. **Sub-agents are ephemeral and process-local.** The `task` tool spawns
   sub-agents as in-process LangGraph invocations. These cannot be distributed
   across workers, survive process crashes, or be independently scaled.
   Long-running sub-agents block the parent's resources.

3. **Shell execution and file operations lack durability.** The `execute` tool
   runs shell commands that may have side effects (git operations, builds,
   deployments). If the process crashes after a side effect but before recording
   the result, the agent has no way to recover or detect the partial state.

4. **Human-in-the-loop via `interrupt_on` consumes resources.** Deep Agents
   supports HITL approval for tools (e.g., `interrupt_on={"edit_file": True}`),
   but the waiting process must remain alive. For approvals that take hours or
   days, this wastes compute.

5. **No execution affinity for stateful backends.** Deep Agents uses pluggable
   backends (`FilesystemBackend`, `SandboxBackend`) that maintain local state
   (files on disk, sandbox sessions). When a sub-agent or the main agent is
   distributed or restarted, the backend state must be co-located with the
   execution. There is no mechanism to ensure execution affinity.

6. **No audit trail for agent actions.** In regulated environments, every tool
   call, file modification, and shell command executed by the agent needs a
   tamper-evident audit log. Deep Agents provides LangSmith integration for
   observability, but not a durable, event-sourced execution history.

The `langgraph-temporal` library already solves these problems for generic
LangGraph graphs. This integration extends that to Deep Agents specifically,
addressing the unique challenges of the Deep Agents execution model (sub-agents,
filesystem backends, shell execution, context summarization).

## 2. Goals

- **G-01**: Enable Deep Agents to run as Temporal Workflows with true durable
  execution, surviving process crashes, restarts, and deployments.
- **G-02**: Use Temporal Sessions to ensure that a Deep Agent's execution
  (including all tool calls, file operations, and shell commands) runs on the
  same worker, maintaining backend state affinity (filesystem, sandbox).
- **G-03**: Map Deep Agent sub-agent invocations (`task` tool) to Temporal
  Child Workflows, providing independent durability, scaling, and observability
  for each sub-agent.
- **G-04**: Reuse `langgraph-temporal` as a dependency, leveraging its existing
  `TemporalGraph`, `LangGraphWorkflow`, Activity infrastructure, streaming,
  and checkpoint support rather than reimplementing Temporal integration.
- **G-05**: Support Deep Agents' human-in-the-loop (`interrupt_on`) via
  Temporal Signals, consuming zero resources while waiting for approval.
- **G-06**: Provide a complete, event-sourced audit trail of all agent actions
  (tool calls, file operations, shell commands) via Temporal's Event History.
- **G-07**: Maintain full backward compatibility with the existing
  `create_deep_agent()` API. Users should be able to make their Deep Agent
  durable with minimal code changes.
- **G-08**: Support parallel sub-agent execution across distributed workers
  while maintaining session affinity for each sub-agent's backend state.

## 3. Non-Goals

- **NG-01**: Modifying the core `deepagents` library. The integration must be
  a separate package (`deepagent-temporal` or similar) that depends on
  `deepagents` and `langgraph-temporal`.
- **NG-02**: Replacing Deep Agents' backend abstraction (`BackendProtocol`).
  The integration wraps existing backends to provide durability, not replace
  them.
- **NG-03**: Supporting Deep Agents CLI (`deepagents_cli`) directly. The CLI
  has its own execution model (Textual TUI, interactive sessions). The
  integration targets the library API (`create_deep_agent()`).
- **NG-04**: Supporting async sub-agents (`AsyncSubAgentMiddleware`) via
  Temporal. These already use remote LangGraph servers and have their own
  durability model.
- **NG-05**: Providing a Temporal-based sandbox backend. Users continue using
  their existing sandbox providers (Modal, Daytona, etc.). Temporal provides
  execution durability, not sandboxing.

## 4. Functional Requirements

### FR-01: Deep Agent to Temporal Workflow Mapping

The library must provide a mechanism to wrap a Deep Agent (the compiled
`CompiledStateGraph` returned by `create_deep_agent()`) as a Temporal Workflow.

- **FR-01.1**: Accept the output of `create_deep_agent()` as input, since it
  returns a standard `CompiledStateGraph`.
- **FR-01.2**: Provide a `TemporalDeepAgent` wrapper (or similar) that extends
  or composes `TemporalGraph` from `langgraph-temporal`.
- **FR-01.3**: Preserve the Deep Agent's middleware stack (TodoListMiddleware,
  FilesystemMiddleware, SubAgentMiddleware, SummarizationMiddleware, etc.)
  within the Temporal execution context.
- **FR-01.4**: The wrapper must support all `create_deep_agent()` parameters:
  `model`, `tools`, `system_prompt`, `middleware`, `subagents`, `skills`,
  `memory`, `backend`, `interrupt_on`, etc.

### FR-02: Worker Affinity for Stateful Backends

The library must ensure execution affinity for stateful backends. Two
strategies are supported depending on the backend type:

**Strategy A -- Sticky Task Queue (recommended for `FilesystemBackend`):**
Create a unique task queue per agent workspace (e.g.,
`deep-agent-{workspace_id}`). A dedicated worker serves this queue. All
Activities for the agent run on this queue, providing natural affinity that
survives continue-as-new. Worker failure is handled by restarting the worker
on the same machine (e.g., Kubernetes pod with PersistentVolume).

**Note on Temporal Sessions:** The Temporal Python SDK (`temporalio`) does
NOT support `workflow.start_session()`. Sessions are only available in Go
and Java SDKs. Worker affinity in Python must use sticky task queues.

**Strategy B -- No Affinity (for `StateBackend`):**
When using `StateBackend` (state stored in LangGraph channels, serialized
through Temporal), no worker affinity is needed. Any worker can execute
Activities since state is passed through Temporal's event history.

- **FR-02.1**: All Activities for a single Deep Agent invocation with a
  stateful backend must execute on the same worker via a sticky task queue.
  For `StateBackend`, no affinity is required.
- **FR-02.2**: The sticky task queue name must be deterministic from the
  agent's workspace identifier, enabling workers to self-register. The
  Workflow must use this queue for all Activity dispatch.
- **FR-02.3**: The sticky task queue name must survive continue-as-new
  (included in `WorkflowInput` and `RestoredState`).
- **FR-02.4**: Worker failure with sticky queues is handled by restarting
  the worker on the same machine (e.g., Kubernetes pod with
  PersistentVolume). Activities include `schedule_to_close_timeout` to
  provide bounded failure detection when the target worker is unavailable.
- **FR-02.5**: HITL waits have no affinity concern with sticky queues -- the
  queue persists independently of any timeout.
- **FR-02.6**: Sub-agents spawned via the `task` tool must use their own
  affinity mechanism. Options: (a) use the same sticky task queue as the
  parent (same worker, shared filesystem), (b) use a separate task queue
  (different worker, requires shared external storage or `StateBackend`).
  The sub-agent affinity strategy must be configurable per sub-agent type.
- **FR-02.8**: When a sub-agent runs on a different worker with
  `FilesystemBackend`, the integration must either (a) require shared
  storage (NFS, S3-FUSE), (b) sync files from parent to child at Child
  Workflow start, or (c) convert to `StateBackend` for distributed
  sub-agents. The default must fail fast if backend prerequisites are not
  met on the target worker.

### FR-03: Sub-Agent as Child Workflow

The library must map Deep Agent sub-agent invocations to Temporal Child
Workflows. **Important**: This is fundamentally different from
`langgraph-temporal`'s subgraph-to-Child-Workflow mapping. Sub-agents in
Deep Agents are runtime-dispatched via the `task` tool (the LLM dynamically
selects the sub-agent type), not compile-time structural subgraphs detected
by inspecting the graph topology. A new dispatch mechanism is required.

- **FR-03.1**: A `TemporalSubAgentMiddleware` must replace
  `SubAgentMiddleware` in the middleware stack. When the `task` tool is
  invoked, instead of creating and invoking a sub-agent in-process, the
  middleware must return a special result (e.g., `SubAgentRequest`) that
  signals the Workflow to dispatch a Child Workflow.
- **FR-03.2**: The Activity executing the agent node must detect `task` tool
  calls in the node output and return a `NodeActivityOutput` with a new
  `child_workflow_requests` field containing the sub-agent specifications
  (sub-agent type, instruction, state context).
- **FR-03.3**: The Workflow must process `child_workflow_requests` by
  dispatching Child Workflows and feeding the results back as `ToolMessage`
  entries before proceeding to the next LLM turn.
- **FR-03.4**: Each sub-agent Child Workflow must be independently durable:
  it must survive worker crashes and be independently queryable/observable
  in the Temporal UI.
- **FR-03.5**: Parallel sub-agent invocations (multiple `task` tool calls in
  a single LLM response) must execute as concurrent Child Workflows via
  `asyncio.gather()` on the Workflow side.
- **FR-03.6**: The Child Workflow must receive an initial state constructed
  by applying the same state transformation as
  `SubAgentMiddleware._create_subagent_state()`: excluded keys
  (`_EXCLUDED_STATE_KEYS`) removed, fresh `messages` list with only the
  task prompt as a `HumanMessage`, inherited non-excluded state channels,
  and inherited backend reference.
- **FR-03.7**: The sub-agent's result (final message in `messages`) must be
  propagated back to the parent Workflow as a `ToolMessage`, matching
  existing `SubAgentMiddleware` behavior.
- **FR-03.8**: Cancellation of the parent Workflow must propagate to all
  active sub-agent Child Workflows via
  `ParentClosePolicy.TERMINATE`.
- **FR-03.9**: Sub-agent type (`subagent_type` parameter) must be preserved
  in the Child Workflow metadata/search attributes for observability.
- **FR-03.10**: Child Workflow ID must include a uniqueness component to
  prevent collisions when the same sub-agent type is invoked multiple times:
  `{parent_wf_id}/subagent/{subagent_type}/{step}_{index}`.
- **FR-03.11**: Sub-agent Child Workflow failures must be caught and
  converted to error `ToolMessage` responses to the parent agent (matching
  existing `SubAgentMiddleware` error handling behavior), not propagated as
  Workflow failures.
- **FR-03.12**: Sub-agent Child Workflows must have a configurable
  `execution_timeout` (default: 30 minutes) to prevent runaway sub-agents.

### FR-04: Node Execution as Activities

**Important architectural note**: In `langgraph-temporal`, the entire LangGraph
node is executed as a single Activity via `execute_node`. Deep Agents' compiled
graph has two primary node types: `call_model` (LLM call + middleware) and
`tools` (tool dispatch). Individual tool calls are NOT separate Activities;
they execute within the `tools` node Activity. This matches `langgraph-temporal`'s
existing node-level Activity model.

Middleware that generates non-deterministic values (e.g.,
`PatchToolCallsMiddleware` generating unique IDs) is safe because it executes
within Activities, not in Workflow deterministic code.

- **FR-04.1**: The `call_model` node (LLM invocation + middleware stack) must
  execute as a Temporal Activity within the Session/sticky queue. This
  includes `SummarizationMiddleware` auto-compaction, prompt caching, and
  tool call patching -- all happen within the same Activity.
- **FR-04.2**: The `tools` node (tool dispatch for all tool calls in a single
  LLM response) must execute as a Temporal Activity within the
  Session/sticky queue. Individual tool calls within this Activity are NOT
  separately visible in Temporal history (this is a known trade-off for
  v0.1; per-tool Activities may be added in v0.2 with graph decomposition).
- **FR-04.3**: The `execute` tool (shell command execution) must use Activity
  heartbeats to report progress for long-running commands. The Activity must
  handle Temporal cancellation by terminating the subprocess (SIGTERM, then
  SIGKILL after a grace period).
- **FR-04.4**: Node Activities must be configured with appropriate timeouts
  and heartbeats:
  - `call_model`: `start_to_close_timeout=5min`,
    `heartbeat_timeout=60s` (heartbeat with token count progress)
  - `tools`: `start_to_close_timeout=30min` (may include long `execute`
    calls), `heartbeat_timeout=60s`
  - `schedule_to_close_timeout` should also be set for retryable nodes to
    bound total time including retries.
- **FR-04.5**: Node Activities must support Temporal retry policies.
  `call_model` should retry on transient LLM errors (rate limits, network
  timeouts). `tools` should generally not retry (tool side effects may not
  be idempotent), unless the specific tool failure is known to be transient.
- **FR-04.6**: Middleware-injected transient metadata (e.g., `cache_control`
  from `AnthropicPromptCachingMiddleware`) must be validated as
  serialization-safe before being included in Temporal event history. If
  non-serializable metadata is detected, the integration must strip it
  before state checkpoint serialization.

### FR-05: Human-in-the-Loop via Temporal Signals

The library must map Deep Agents' `interrupt_on` mechanism to Temporal Signals.

- **FR-05.1**: When `interrupt_on` is configured for a tool (e.g.,
  `interrupt_on={"edit_file": True}`), the Workflow must pause before
  executing the tool Activity and wait for a Temporal Signal.
- **FR-05.2**: The interrupt must expose the pending tool call details
  (tool name, arguments) via a Temporal Query, enabling external systems
  to present the approval request to a human.
- **FR-05.3**: The resume Signal must support approval, rejection, or
  modification of the tool call arguments.
- **FR-05.4**: While waiting for approval, the Workflow must consume zero
  compute resources (Temporal's native signal-wait behavior).
- **FR-05.5**: The library must map `interrupt_on: dict[str, bool]` (Deep
  Agents' tool-level interrupt configuration) to the Temporal Signal/Query
  model. **Note**: This is different from `langgraph-temporal`'s node-level
  `interrupt_before`/`interrupt_after`. Tool-level interrupts must be
  detected inside the `tools` node Activity (which returns an interrupt
  result with tool call details) and handled at the Workflow level.
- **FR-05.6**: Consider using Temporal Workflow Updates (Temporal 1.10+)
  instead of Signals for HITL approval. Updates provide synchronous
  request-response semantics, letting the approval UI confirm the Workflow
  received and processed the approval. Signals are fire-and-forget.

### FR-06: Context Summarization Durability

The library must ensure that Deep Agents' auto-summarization works correctly
within the Temporal execution model.

- **FR-06.1**: `SummarizationMiddleware` triggers auto-compaction within the
  `call_model` Activity (it is part of the middleware stack). The
  summarization LLM call executes as part of the same Activity -- it is NOT
  independently retryable. The Activity must use heartbeats to signal
  summarization progress. If finer-grained durability is required, a future
  version may introduce a dedicated pre-model-call Activity step.
- **FR-06.2**: The offloaded conversation history (stored at
  `/conversation_history/{thread_id}.md`) must be persisted via the backend
  within the Session/sticky queue, ensuring the file is accessible for
  later retrieval on the same worker.
- **FR-06.3**: Continue-as-new must preserve all agent state channels
  including `skills_metadata` and `memory_contents`. These must be
  validated for serialization compatibility with Temporal's payload size
  limits (use `LargePayloadCodec` for large skill/memory payloads).

### FR-07: Streaming Support

The library must support streaming Deep Agent execution events.

- **FR-07.1**: Token-by-token LLM output must be streamable to the client
  via `langgraph-temporal`'s `StreamBackend` abstraction.
- **FR-07.2**: Tool call events (start, progress, completion) must be
  streamable.
- **FR-07.3**: Sub-agent progress events must be streamable (at minimum:
  sub-agent started, sub-agent completed).
- **FR-07.4**: The streaming mechanism must work across the Session boundary
  (client may be on a different machine than the Session worker).

### FR-08: Configuration and Initialization

- **FR-08.1**: The library must provide a `TemporalDeepAgent` wrapper (or
  `create_temporal_deep_agent()` factory) that accepts all
  `create_deep_agent()` parameters plus Temporal-specific configuration.
- **FR-08.2**: Temporal-specific configuration must include: Temporal client,
  task queue, session options, per-tool Activity options, workflow timeouts,
  and stream backend.
- **FR-08.3**: The library must provide helpers to create a Temporal Worker
  configured for Deep Agent execution, including Session worker capabilities.
- **FR-08.4**: A `local()` factory method must be provided for local
  development using Temporal's test server.
- **FR-08.5**: The library must support configuring different task queues for
  the main agent vs. sub-agents, enabling heterogeneous worker pools.

### FR-09: Reuse of `langgraph-temporal`

The integration must maximize reuse of `langgraph-temporal` components.

- **FR-09.1**: Use `TemporalGraph` as the base for wrapping the Deep Agent's
  `CompiledStateGraph`.
- **FR-09.2**: Reuse `LangGraphWorkflow` for orchestrating the agent's
  execution loop (node scheduling, channel state management, interrupt
  handling).
- **FR-09.3**: Reuse `execute_node` Activity for executing Deep Agent graph
  nodes as Temporal Activities.
- **FR-09.4**: Reuse `TemporalCheckpointSaver` for state queries.
- **FR-09.5**: Reuse `PollingStreamBackend` (and other `StreamBackend`
  implementations) for streaming.
- **FR-09.6**: Reuse `EncryptionCodec` and `LargePayloadCodec` for payload
  security and size management.
- **FR-09.7**: Extend (not fork) `langgraph-temporal` where Deep Agent-specific
  behavior is needed. The following upstream changes are required in
  `langgraph-temporal` (all backward-compatible with optional fields):
  - `WorkflowInput`: Add optional `session_options` and
    `sticky_task_queue` fields.
  - `LangGraphWorkflow`: Add an Activity dispatch abstraction (e.g.,
    `ActivityDispatcher` protocol) so Session/sticky-queue dispatch can
    be injected without modifying the Workflow class. Add support for
    `child_workflow_requests` in `NodeActivityOutput` for runtime
    sub-agent dispatch.
  - `NodeActivityOutput`: Add optional `child_workflow_requests` field.
  - `create_worker`: Add Session worker configuration support.
  - `RestoredState`: Add worker affinity hint for continue-as-new.

### FR-10: Error Handling

- **FR-10.1**: Sub-agent Child Workflow failures must be caught and returned
  as error `ToolMessage` responses to the parent agent, not propagated as
  parent Workflow failures.
- **FR-10.2**: Tool execution failures within the `tools` node Activity must
  be handled at the Activity level (matching LangGraph's existing error
  handling) and returned as error tool messages.
- **FR-10.3**: LLM API errors (rate limits, context overflow) must be
  retryable via Activity retry policy. `ContextOverflowError` should trigger
  summarization on the next attempt.
- **FR-10.4**: Backend lifecycle errors (initialization failures, permission
  denied) must surface as non-retryable Activity failures with descriptive
  error messages.

### FR-11: Backend Lifecycle

- **FR-11.1**: Backend initialization must occur on the worker where
  Activities execute. For `FilesystemBackend`, the `root_dir` must exist
  on the worker. For `StoreBackend`, the LangGraph store connection must
  be available. The integration must validate backend prerequisites at
  worker startup and fail fast if not met.
- **FR-11.2**: The backend instance must be accessible within Activities
  via the `GraphRegistry` (same mechanism as the compiled graph). The
  backend must be registered on the worker and looked up by reference
  in Activities.
- **FR-11.3**: LLM model client initialization must occur on workers.
  API keys and model configuration must be resolved at the worker level,
  not serialized through Temporal event history.

## 5. Non-Functional Requirements

### NFR-01: Performance

- Session overhead must add less than 10ms per Activity dispatch compared to
  non-session Activities.
- Sub-agent Child Workflow creation must complete within 100ms.
- End-to-end latency for a simple tool call (e.g., `read_file`) must not
  exceed 100ms overhead compared to non-Temporal execution.

### NFR-02: Reliability

- Agent execution must survive worker crashes. On restart, the agent must
  resume from the last completed tool call or LLM turn.
- Session worker failure must be detected within the `session_timeout` and
  handled according to the configured recovery strategy (FR-02.4).
- All tool executions within a Session must be idempotency-aware. The library
  must provide an idempotency key (`{workflow_id}/{step}/{tool_name}`) for
  tools that need it.

### NFR-03: Observability

- Each tool invocation must be visible as a separate Activity in the Temporal
  UI, with the tool name as the Activity type.
- Sub-agent executions must be visible as Child Workflows in the Temporal UI.
- Temporal Search Attributes must include: `agent_name`, `thread_id`,
  `current_tool`, `sub_agent_count`.
- Integration with LangSmith must be preserved (the existing LangSmith
  tracing in Deep Agents should continue to work within Activities).

### NFR-04: Scalability

- Support hundreds of concurrent Deep Agent Workflow executions per namespace.
- Sub-agent Child Workflows must scale independently of the parent.
- Session workers must support configurable concurrency limits for both
  Activities and Sessions.

### NFR-05: Security

- All state serialization must support encryption via `langgraph-temporal`'s
  `EncryptionCodec`.
- LLM API keys and tool credentials must never appear in Temporal Event
  History. They must be resolved at the worker level.
- The `execute` tool's command and output must be encrypted in Event History
  for environments handling sensitive data.

### NFR-06: Compatibility

- Python 3.11+ (matching Deep Agents' minimum).
- Compatible with `deepagents` version 0.5+.
- Compatible with `langgraph-temporal` version 0.1+.
- Compatible with `temporalio` SDK version 1.7+.

## 6. User Stories

### US-01: Durable Research Agent

*As a developer, I want my Deep Agent to perform multi-hour research tasks
(reading files, executing commands, spawning sub-agents) and survive process
restarts without losing progress.*

Acceptance criteria:
- Create a Deep Agent with Temporal integration using `TemporalDeepAgent`.
- Start a research task that involves 50+ tool calls.
- Kill and restart the worker mid-execution.
- The agent resumes from the last completed tool call.

### US-02: Distributed Sub-Agent Execution

*As a developer, I want my Deep Agent to spawn sub-agents that run on different
workers, enabling parallel research on separate topics.*

Acceptance criteria:
- Main agent spawns 3 sub-agents via `task` tool in parallel.
- Each sub-agent runs as a separate Child Workflow.
- Sub-agents execute on different workers (verified via Temporal UI).
- Main agent collects all results and synthesizes a response.

### US-03: Session-Pinned File Operations

*As a developer, I want my Deep Agent to read and write files on the local
filesystem, with all operations executing on the same worker to maintain
filesystem consistency.*

Acceptance criteria:
- Agent writes a file via `write_file`.
- Agent reads the same file via `read_file` in a subsequent step.
- Both operations execute on the same worker (Session affinity).
- Worker crash and restart recovers to the correct Session worker.

### US-04: Human-Approved Code Edits

*As a developer, I want my Deep Agent to pause before editing files and wait
for my approval, without consuming compute while waiting.*

Acceptance criteria:
- Configure `interrupt_on={"edit_file": True}`.
- Agent proposes a file edit.
- Workflow pauses; worker can be scaled to zero.
- Human approves via Signal; agent proceeds with the edit.

### US-05: Auditable Agent Execution

*As a compliance officer, I need a complete record of every tool call, file
modification, and shell command executed by the agent.*

Acceptance criteria:
- Every tool call is recorded as an Activity in Temporal Event History.
- Full tool inputs and outputs are queryable.
- Event History can be exported for audit.

### US-06: Long-Running Code Generation Agent

*As a developer, I want my Deep Agent to generate and test code over multiple
iterations, with the execution surviving overnight.*

Acceptance criteria:
- Agent runs a code-test-fix loop spanning 200+ steps.
- Continue-as-new triggers automatically to prevent history bloat.
- Agent resumes after overnight worker restart.
- All context (files written, todos tracked) is preserved.

## 7. API Design

### 7.1 Primary API (Sticky Task Queue -- recommended)

```python
from deepagent_temporal import TemporalDeepAgent, AffinityMode
from deepagents import create_deep_agent
from temporalio.client import Client

# 1. Create a standard Deep Agent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[my_custom_tool],
    system_prompt="You are a research assistant.",
    backend=FilesystemBackend(root_dir="/workspace"),
    interrupt_on={"execute": True},
)

# 2. Wrap with Temporal (sticky task queue mode)
client = await Client.connect("localhost:7233")
temporal_agent = TemporalDeepAgent(
    agent,
    client,
    task_queue="deep-agents",
    affinity_mode=AffinityMode.STICKY_QUEUE,
)

# 3. Use it -- delegates to TemporalGraph for standard operations
result = await temporal_agent.ainvoke(
    {"messages": [{"role": "user", "content": "Research quantum computing"}]},
    config={"configurable": {"thread_id": "research-001"}},
)

# 4. Start a worker (on machine with /workspace mounted)
worker = temporal_agent.create_worker()
await worker.run()
```

### 7.2 Session Mode (for externally-stored backends)

```python
from deepagent_temporal import SessionOptions, SessionRecoveryStrategy

temporal_agent = TemporalDeepAgent(
    agent,
    client,
    task_queue="deep-agents",
    affinity_mode=AffinityMode.SESSION,
    session_options=SessionOptions(
        session_idle_timeout=timedelta(hours=2),  # Account for HITL waits
        recovery_strategy=SessionRecoveryStrategy.RECREATE_SESSION,
    ),
)
```

### 7.3 Sub-Agent Configuration

```python
temporal_agent = TemporalDeepAgent(
    agent,
    client,
    task_queue="deep-agents",
    subagent_task_queue="deep-agent-subagents",  # Separate queue
    subagent_affinity_mode=AffinityMode.SESSION,  # Independent sessions
    subagent_execution_timeout=timedelta(minutes=30),
)
```

### 7.4 Local Development

```python
temporal_agent = await TemporalDeepAgent.local(agent)
result = await temporal_agent.ainvoke(input, config)
```

## 8. Migration Path

### Phase 1: Core Integration (v0.1)

- `TemporalDeepAgent` wrapper with Session support.
- Tool execution as Activities within Sessions.
- Sub-agent execution as Child Workflows.
- Human-in-the-loop via Signals.
- Reuse of `langgraph-temporal` core components.

### Phase 2: Advanced Features (v0.2)

- Per-tool Activity customization (timeouts, retries, task queues).
- Streaming support for tool progress and LLM tokens.
- Sub-agent session isolation configuration.
- Context summarization durability.

### Phase 3: Production Hardening (v0.3)

- Session recovery strategies for different backend types.
- Search Attributes for agent observability.
- LangSmith + Temporal unified tracing.
- Performance benchmarks and optimization.
