# Comparison with Alternatives

This document compares `deepagent-temporal` with other deployment options for LangGraph-based agents.

## Quick Comparison

| | `deepagent-temporal` | LangGraph Platform | `temporal-ai-agent` |
|---|---|---|---|
| **Runtime** | Temporal (self-hosted) | LangGraph Platform (managed or self-hosted) | Temporal (self-hosted) |
| **Agent framework** | Deep Agents / LangGraph | LangGraph | Framework-agnostic |
| **Migration effort** | 3-line change from vanilla Deep Agents | Deploy to LangGraph Platform | Rewrite agent as Temporal Workflow |
| **Streaming** | Token-level via callback capture + optional Redis Streams (see below) | Full token-level streaming (native) | N/A |
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
- You need native token-level streaming with zero configuration (deepagent-temporal supports token streaming but requires opt-in setup and optional Redis for real-time delivery).
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

## Streaming Architecture

`deepagent-temporal` supports token-level LLM streaming through two mechanisms:

### Phase 1: Callback-Based Token Capture (no extra infrastructure)

Enable `enable_token_streaming=True` on `TemporalDeepAgent`. A `StreamingNodeWrapper` wraps each graph node and injects a `TokenCapturingHandler` (LangChain `BaseCallbackHandler`) into the config before `ainvoke()`. The handler intercepts `on_llm_new_token` events from the chat model and captures individual tokens.

Without Redis, tokens are buffered in the Activity result and delivered to the client after the Activity completes. This gives token-level *data granularity* but not token-level *latency* — the client receives all tokens at once when the LLM call finishes.

### Phase 2: Real-Time Delivery via Redis Streams

Add a `RedisStreamBackend` for real-time delivery (~10-50ms per token). The handler publishes each token to a Redis Stream as it arrives from the LLM. The client subscribes via `XREAD` and receives tokens in real-time. Temporal still handles durable state transitions; Redis handles low-latency delivery.

If Redis is unavailable, streaming degrades gracefully to Phase 1 behavior.

### Comparison with LangGraph Platform

LangGraph Platform supports full token-level streaming natively (in-process, no sidecar). `deepagent-temporal` achieves comparable results with `RedisStreamBackend`, but requires:

- Opt-in configuration (`enable_token_streaming=True`)
- Redis for real-time delivery (optional — without it, tokens are buffered)
- Slightly higher per-token latency (~10-50ms vs. in-process)

See [docs/streaming-design.md](streaming-design.md) for the full architecture, data flow diagrams, and middleware compatibility notes.
