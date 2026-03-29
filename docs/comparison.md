# Comparison with Alternatives

This document compares `deepagent-temporal` with other deployment options for LangGraph-based agents.

## Quick Comparison

| | `deepagent-temporal` | LangGraph Platform | `temporal-ai-agent` |
|---|---|---|---|
| **Runtime** | Temporal (self-hosted) | LangGraph Platform (managed or self-hosted) | Temporal (self-hosted) |
| **Agent framework** | Deep Agents / LangGraph | LangGraph | Framework-agnostic |
| **Migration effort** | 3-line change from vanilla Deep Agents | Deploy to LangGraph Platform | Rewrite agent as Temporal Workflow |
| **Streaming** | Limited — Activity-level buffering (see below) | Full token-level streaming | N/A |
| **HITL** | Temporal Signals (zero-resource wait) | LangGraph interrupts (process must stay alive) | Temporal Signals |
| **Sub-agent durability** | Child Workflows (independent) | In-process (shared fate) | Manual |
| **Crash recovery** | Automatic (Temporal replay) | Checkpoint-based | Automatic (Temporal replay) |
| **Audit trail** | Temporal Event History (every Activity) | LangSmith | Temporal Event History |
| **Cost** | Temporal infra only | LangGraph Platform pricing | Temporal infra only |
| **Best for** | Teams with existing Temporal + Deep Agents | Teams wanting managed deployment | Teams building Temporal-native agents from scratch |

## Detailed Comparison

### vs. LangGraph Platform

**LangGraph Platform** is LangChain's official deployment solution. It provides managed infrastructure, streaming, persistence, and a REST API.

**Choose `deepagent-temporal` when:**
- Your team already operates Temporal infrastructure.
- You need Temporal's event-sourced audit trail for compliance.
- You want sub-agents to survive parent crashes independently (Child Workflows).
- You want zero-resource HITL waits (Temporal Signals vs. process-bound interrupts).
- You want to avoid additional managed service costs.

**Choose LangGraph Platform when:**
- You want managed infrastructure with minimal operational burden.
- You need full token-level streaming (Temporal Activities buffer responses).
- You prefer official LangChain support and documentation.
- You don't have existing Temporal expertise or infrastructure.

### vs. `temporal-ai-agent`

[`temporal-ai-agent`](https://github.com/temporalio/temporal-ai-agent) is a Temporal community sample that demonstrates building AI agents directly as Temporal Workflows, without LangGraph.

**Choose `deepagent-temporal` when:**
- You already have a Deep Agent / LangGraph agent and want to add durability.
- You want to leverage LangGraph's middleware, tools, and agent patterns.
- You want minimal migration effort (3-line code change).

**Choose `temporal-ai-agent` when:**
- You're building a new agent from scratch and want Temporal-native design.
- You don't need LangGraph's abstractions.
- You want full control over the Temporal Workflow logic.

### vs. Custom Checkpointing

Some teams implement custom checkpointing using a database (PostgreSQL, Redis) and a retry loop.

**Choose `deepagent-temporal` when:**
- You want battle-tested workflow orchestration (not a custom retry loop).
- You need workflow-as-code with replay semantics.
- You need child workflow orchestration for sub-agents.
- You want an event-sourced audit trail without building one.

**Choose custom checkpointing when:**
- Your agent is simple (few steps, no sub-agents).
- You can't introduce Temporal as a dependency.
- You need minimal infrastructure.

## Streaming Limitations

`deepagent-temporal` uses Temporal Activities for node execution. Activities are request-response: the entire node (LLM call or tool execution) runs to completion, and the result is returned as a single payload.

This means:
- **Token-level streaming** is not natively supported through Temporal's Activity mechanism.
- The `astream` API uses `langgraph-temporal`'s `StreamBackend` abstraction, which provides Activity-level events (node started, node completed) rather than token-by-token output.
- For token-level streaming, a sidecar channel (Redis pub/sub, SSE endpoint) running parallel to the workflow is needed. This is a roadmap item.

LangGraph Platform supports full token-level streaming natively.
