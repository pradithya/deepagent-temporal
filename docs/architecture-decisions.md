# Architecture Decision Record

This document captures the key architectural decisions behind `deepagent-temporal`, the alternatives considered, and the tradeoffs accepted.

## ADR-01: Why Temporal?

### Problem

Deep Agents (from `langchain-ai/deepagents`) lacks durable execution. Long-running agent tasks that span minutes to hours lose all progress on process crash. Sub-agents are ephemeral and process-local. Human-in-the-loop approval blocks a live process.

### Alternatives Considered

| Alternative | Pros | Cons | Why Rejected |
|---|---|---|---|
| **LangGraph Platform** (managed) | Official support, streaming, managed infrastructure | Vendor lock-in, pricing, no self-hosted option for all teams | Teams with existing Temporal infrastructure want to reuse it |
| **LangGraph Platform** (self-hosted) | Official support | Requires LangGraph Platform license; limited customization | Not all teams can or want to run LangGraph Platform |
| **Celery + Redis/RabbitMQ** | Simple, widely deployed | No native workflow orchestration, no event sourcing, manual state management, no replay | Building durable workflows on Celery means reimplementing what Temporal provides |
| **Custom checkpointing** (database + retry loop) | No new infrastructure | Fragile, no standardized replay, no workflow-as-code, reinventing the wheel | Every team would build this differently; no ecosystem |
| **Temporal** | Durable execution, event sourcing, workflow-as-code, replay, signals, child workflows, battle-tested | Operational dependency (server required), learning curve, serialization constraints | **Selected** — best match for the problem space |

### Decision

Use Temporal as the durable execution backend. It provides:

- **Workflow-as-code**: The execution graph is code, not YAML/JSON configuration.
- **Event sourcing**: Every Activity is recorded. Full audit trail for free.
- **Signals and Queries**: Native primitives for HITL (zero-resource waits) and state inspection.
- **Child Workflows**: Natural mapping for sub-agent dispatch.
- **Continue-as-new**: Handles long-running agents that exceed history limits.
- **Battle-tested**: Used in production at Uber, Netflix, Snap, and others for mission-critical workflows.

### Tradeoffs Accepted

- Temporal server is an operational dependency — it must be deployed and maintained.
- Serialization constraints — all state must be JSON-serializable (see `docs/serialization.md`).
- `UnsandboxedWorkflowRunner` required (see `docs/sandbox-tradeoffs.md`).

---

## ADR-02: Worker-Specific Task Queues for Affinity

### Problem

Deep Agents uses `FilesystemBackend` — tools read and write files on local disk. All Activities for an agent must execute on the same worker to maintain filesystem consistency.

### Alternatives Considered

| Alternative | Pros | Cons | Why Rejected |
|---|---|---|---|
| **Temporal Sessions** (Go/Java) | First-class SDK support | Not available in Python SDK (`temporalio`) | Cannot use — Python SDK limitation |
| **Shared filesystem** (NFS/EFS) | Any worker can run Activities | NFS performance for small file I/O is poor; adds infrastructure dependency | Viable but adds latency and complexity |
| **State-based backend only** | No affinity needed | Not all Deep Agent backends support this; filesystem operations need local state | Limits backend choices |
| **Worker-specific task queues** | Works in Python SDK, no NFS needed, survives continue-as-new | Requires two workers per process, queue discovery Activity | **Selected** |

### Decision

Use the [Temporal worker-specific task queues pattern](https://github.com/temporalio/samples-python/tree/main/worker_specific_task_queues):

1. Each worker generates a unique task queue name.
2. Two internal workers run: one on the shared queue (Workflows + discovery), one on the unique queue (Activities).
3. Workflows discover the worker's unique queue via a `get_available_task_queue` Activity.
4. All subsequent Activities are dispatched to the discovered queue.

This provides worker affinity without requiring Sessions (unavailable in Python) or shared filesystems.

### Tradeoffs Accepted

- One extra Activity call per workflow for queue discovery.
- Worker failure requires restarting on the same machine (or using PersistentVolumes in Kubernetes).
- Two internal workers per process adds minor resource overhead.

---

## ADR-03: Sub-Agents as Child Workflows

### Problem

Deep Agents spawns sub-agents via the `task` tool. In vanilla Deep Agents, sub-agents run in-process — they share the parent's memory, can't survive crashes independently, and can't be distributed across workers.

### Alternatives Considered

| Alternative | Pros | Cons | Why Rejected |
|---|---|---|---|
| **In-process** (vanilla) | Simple, shared memory | No durability, no distribution, blocks parent | Current limitation we're solving |
| **Separate Temporal Workflows** (unlinked) | Independent | No parent-child relationship, manual result propagation | Loses Temporal's parent-child semantics |
| **Activities** | Simpler dispatch | No independent durability, limited to Activity timeouts, no sub-agent state inspection | Sub-agents need full workflow capabilities |
| **Child Workflows** | Independent durability, observability, timeout, cancellation propagation | More complex dispatch, serialization overhead | **Selected** |

### Decision

Map sub-agent invocations to Temporal Child Workflows:

- `TemporalSubAgentMiddleware` intercepts `task` tool calls.
- Instead of invoking sub-agents in-process, it stores `SubAgentRequest` objects in a context variable.
- The Activity collects pending requests after execution.
- The Workflow dispatches Child Workflows and feeds results back as `ToolMessage` entries.

Child Workflow IDs follow the pattern: `{parent_wf_id}/subagent/{type}/{step}_{index}`.

### Tradeoffs Accepted

- Sub-agent dispatch is runtime-dynamic (LLM chooses the type), unlike `langgraph-temporal`'s compile-time subgraph mapping.
- Middleware must be injected before graph compilation — it cannot be patched after.
- Sub-agent failures are caught and returned as error messages (not propagated as parent failures).

---

## ADR-04: Reuse `langgraph-temporal` as Foundation

### Problem

Building Temporal integration for Deep Agents from scratch would duplicate significant effort. `langgraph-temporal` already handles the core LangGraph-to-Temporal mapping.

### Decision

Compose (not fork) `langgraph-temporal`. `TemporalDeepAgent` wraps `TemporalGraph` and delegates standard operations while adding Deep Agent-specific behavior through configuration injection.

### What We Reuse

- `TemporalGraph` — workflow wrapping and client API
- `LangGraphWorkflow` — workflow orchestration (node scheduling, state management, interrupts)
- `execute_node` Activity — node execution within Activities
- `StreamBackend` — streaming infrastructure
- `RetryPolicyConfig`, `ActivityOptions` — configuration types
- `SubAgentConfig` — child workflow dispatch configuration

### What We Add

- Worker affinity via sticky task queues (`use_worker_affinity`)
- `TemporalSubAgentMiddleware` for runtime sub-agent dispatch
- `SubAgentRequest` / `SubAgentSpec` for sub-agent configuration
- `validate_payload_size` for serialization boundary guards
- Retry policy recommendations for LLM workloads

### Tradeoffs Accepted

- Coupled to `langgraph-temporal`'s internal API (e.g., `_child_workflow_requests_var`).
- Some Deep Agent-specific features require upstream changes to `langgraph-temporal` (documented in `docs/REQUIREMENTS.md` FR-09.7).

---

## ADR-05: Tool-Level Interrupts via Signals

### Problem

Deep Agents supports `interrupt_on` at the tool level (e.g., `interrupt_on={"edit_file": True}`). This is different from LangGraph's node-level `interrupt_before`/`interrupt_after`.

### Decision

Map tool-level interrupts to Temporal Signals:

1. The `tools` node Activity detects tool calls that require approval.
2. It returns an interrupt result with the pending tool call details.
3. The Workflow pauses and waits for a Signal.
4. The Signal carries approval/rejection/modification.
5. On approval, the Workflow continues execution.

While waiting, the Workflow consumes zero compute resources (Temporal's native signal-wait).

### Tradeoffs Accepted

- Signals are fire-and-forget — the sender does not get confirmation that the workflow received the signal. Consider Temporal Updates (1.10+) for synchronous acknowledgment in future versions.

---

## What This Project Does NOT Solve

- **Sandbox execution** — `deepagent-temporal` does not provide sandboxed code execution. Use existing sandbox providers (Modal, Daytona) with Deep Agents' `SandboxBackend`.
- **LLM provider failover** — Temporal retries Activities, but does not switch between LLM providers on failure. Use LangChain's fallback chains for this.
- **Multi-tenant isolation** — Temporal namespaces provide some isolation, but `deepagent-temporal` does not manage tenant boundaries.
- **Real-time token streaming** — Activities are request-response. Token-level streaming uses a sidecar channel (see Limitations in README).
- **Automatic cost optimization** — Retry policies prevent runaway costs, but the library does not monitor or limit LLM API spend.
