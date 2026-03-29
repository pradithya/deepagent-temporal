# Streaming Design

This document describes the token-level LLM streaming architecture for `deepagent-temporal`.

## Problem

Temporal Activities are request-response: the entire LLM call runs to completion before the result is returned. The current `astream` API emits events when nodes start and complete, but does not stream individual LLM tokens. This is the #1 credibility gap identified in the project assessment.

## Key Design Insight

**`node.bound.astream()` does NOT produce token-level chunks.** LangGraph node functions wrapped as `RunnableCallable` yield exactly one chunk from `astream()` — the final result. Switching `ainvoke` to `astream` inside the Activity does not help.

The correct approach: inject a **LangChain callback handler** into the Activity's config that intercepts `on_llm_new_token` events from the chat model running *inside* `ainvoke()`. This is the same mechanism LangGraph uses internally (`StreamMessagesHandler` in `langgraph.pregel._messages`).

## Architecture

### Phase 1: Callback-Based Token Capture

Tokens are captured inside the Activity via callback injection. They arrive at the client after the Activity completes (same latency as today, but token-level data granularity).

```
Chat Model (inside node function via ainvoke)
  │ fires on_llm_new_token callback
  ▼
TokenCapturingHandler (injected by StreamingNodeWrapper)
  │ creates TokenEvent, calls sink()
  ▼
CONFIG_KEY_STREAM handler → custom_data list
  │ Activity completes
  ▼
NodeActivityOutput.custom_data = [TokenEvent, ...]
  │
  ▼
Workflow._emit_stream_events → stream_buffer
  │
  ▼
Client polls get_stream_buffer → receives TokenEvents
```

**How it works:**
1. `StreamingNodeWrapper` wraps `node.bound` in the compiled graph
2. When `_execute_node_impl` calls `node.bound.ainvoke()`, our wrapper intercepts the call
3. The wrapper injects a `TokenCapturingHandler` into `config["callbacks"]`
4. The handler intercepts `on_llm_new_token` from the LLM and publishes `TokenEvent` objects
5. The upstream Activity code runs unchanged — it still calls `ainvoke()` on the wrapper

**Usage:**

```python
temporal_agent = TemporalDeepAgent(
    agent, client,
    task_queue="my-agents",
    enable_token_streaming=True,
)

# Worker wraps graph nodes automatically
worker = temporal_agent.create_worker(...)

# Client receives token events
async for token_event in temporal_agent.astream(
    input, config, stream_mode="tokens"
):
    print(token_event["token"], end="", flush=True)
```

### Phase 2: Real-Time Delivery via Redis Streams

Tokens are published to Redis Streams as they arrive from the LLM (~10-50ms latency). Temporal handles durable state; Redis handles real-time delivery.

```
Chat Model (inside node function via ainvoke)
  │ fires on_llm_new_token callback
  ▼
TokenCapturingHandler
  ├──▶ Redis XADD (real-time, ~1ms)
  │      │
  │      ▼
  │    Client XREAD subscription (~10-50ms total)
  │
  └──▶ Summary to custom_data (durable path)
         │
         ▼
       Temporal Event History (small payload)
```

**Usage:**

```python
from deepagent_temporal import TemporalDeepAgent, RedisStreamBackend

redis_backend = RedisStreamBackend(redis_url="redis://localhost:6379")

temporal_agent = TemporalDeepAgent(
    agent, client,
    enable_token_streaming=True,
    redis_stream_backend=redis_backend,
)

# Real-time token delivery
async for token_event in temporal_agent.astream(
    input, config, stream_mode="tokens"
):
    print(token_event["token"], end="", flush=True)
```

## Components

| Component | File | Purpose |
|---|---|---|
| `TokenEvent` | `streaming.py` | Token data container |
| `TokenCapturingHandler` | `streaming.py` | LangChain callback that captures LLM tokens |
| `StreamingNodeWrapper` | `activity.py` | Wraps `node.bound` to inject callback handler |
| `wrap_graph_for_streaming` | `activity.py` | Walks graph nodes and wraps them |
| `RedisStreamBackend` | `streaming.py` | Redis Streams publish/subscribe |
| `create_streaming_worker` | `worker.py` | Worker factory with streaming pre-configured |

## Middleware Compatibility

- **SummarizationMiddleware**: May trigger two LLM calls. The handler tracks `_llm_call_count` and resets the token index on each new call.
- **PatchToolCallsMiddleware**: Patches tool call IDs after LLM completion. Streamed tool call chunks may have different IDs than the patched final result. Clients should use the final `AIMessage` for tool call IDs.
- **AnthropicPromptCachingMiddleware**: Modifies the request only. No interaction with streaming.

## Tradeoffs

| | Phase 1 (Polling) | Phase 2 (Redis) |
|---|---|---|
| Token latency | Full LLM call duration | ~10-50ms per token |
| Infrastructure | None (Temporal only) | Redis required |
| Durability | Tokens in Event History | Tokens in Redis (ephemeral), summary in Event History |
| History bloat | Risk with many tokens | Minimal (summary only) |
| Graceful degradation | N/A | Falls back to Phase 1 if Redis unavailable |

## Event History Considerations

Storing individual token events in `NodeActivityOutput.custom_data` adds to Temporal Event History. For a 2000-token LLM response at ~50 bytes per token event, that's ~100KB per Activity result. Over hundreds of steps, this can approach Temporal's 50MB Event History limit.

**Phase 1 mitigation**: Use `SummarizationMiddleware` to keep conversations shorter. Monitor with `validate_payload_size()`.

**Phase 2 solution**: Tokens go to Redis only. `custom_data` stores a small summary (`{"token_count": N}`), keeping Event History lean.
