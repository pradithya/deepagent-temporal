# Token Streaming

This guide explains how to enable token-level LLM streaming with `deepagent-temporal`.

## Background

Temporal Activities are request-response: the entire LLM call runs to completion inside an Activity before the result is returned. By default, `astream` emits Activity-level events (node started, node completed) — not individual LLM tokens.

`deepagent-temporal` solves this by injecting a LangChain callback handler (`TokenCapturingHandler`) that intercepts `on_llm_new_token` events from the chat model running *inside* `ainvoke()`. This captures tokens without changing the Activity execution model.

## Phase 1: Buffered Token Streaming (No Extra Infrastructure)

Enable `enable_token_streaming=True` to capture individual tokens. Without Redis, tokens are buffered in the Activity result and delivered after the Activity completes:

```python
from deepagent_temporal import TemporalDeepAgent

temporal_agent = TemporalDeepAgent(
    agent, client,
    task_queue="my-agents",
    enable_token_streaming=True,
)

# On the worker side, create_worker() wraps graph nodes automatically
worker = temporal_agent.create_worker(
    workflow_runner=UnsandboxedWorkflowRunner(),
)

# On the client side, use stream_mode="tokens"
async for event in temporal_agent.astream(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "t1"}},
    stream_mode="tokens",
):
    print(event["token"], end="", flush=True)
```

Each event is a dict with:

| Field | Type | Description |
|---|---|---|
| `type` | `str` | Always `"token"` |
| `token` | `str` | The token text |
| `node_name` | `str` | Graph node that produced this token |
| `index` | `int` | Token position within this LLM call |
| `is_final` | `str` | `"1"` if this is the last event, `"0"` otherwise |
| `attempt` | `int` | Activity attempt number (for deduplication) |

**Latency**: Tokens arrive after the full LLM call completes. This gives token-level *data granularity* but not real-time delivery.

## Phase 2: Real-Time Streaming via Redis

For real-time token delivery (~10-50ms per token), add a `RedisStreamBackend`:

```python
from deepagent_temporal import TemporalDeepAgent, RedisStreamBackend

redis_backend = RedisStreamBackend(redis_url="redis://localhost:6379")

temporal_agent = TemporalDeepAgent(
    agent, client,
    task_queue="my-agents",
    enable_token_streaming=True,
    redis_stream_backend=redis_backend,
)
```

With Redis configured:

1. The `TokenCapturingHandler` publishes each token to a Redis Stream via `XADD` as it arrives from the LLM
2. The client subscribes via `XREAD` and receives tokens in real-time
3. Temporal still handles durable state (the Activity result contains a summary, not individual tokens)

### Redis Configuration

```python
redis_backend = RedisStreamBackend(
    redis_url="redis://localhost:6379",
    channel_prefix="deepagent:stream:",  # Redis key prefix
    stream_maxlen=5000,                   # Approximate max entries per stream
    stream_ttl_seconds=300,               # TTL after stream completes
)
```

### Standalone Streaming Worker

For advanced setups, use `create_streaming_worker` directly:

```python
from deepagent_temporal import create_streaming_worker

worker = create_streaming_worker(
    graph,
    client,
    task_queue="my-agents",
    redis_url="redis://localhost:6379",
    use_worker_affinity=True,
    node_names=["call_model"],  # Only wrap LLM-calling nodes
)
```

## Graceful Degradation

If Redis is unavailable:

- `RedisStreamBackend.publish` catches connection errors and logs a warning
- Tokens are still captured by the callback handler
- The client falls back to receiving tokens from the Temporal stream buffer (Phase 1 behavior)

No crash, no data loss on the durable path.

## Middleware Compatibility

The callback handler works with all Deep Agent middleware:

- **SummarizationMiddleware** — May trigger two LLM calls (summarization + response). The handler tracks `_llm_call_count` and resets the token index on each new call, so clients receive clean token sequences.
- **PatchToolCallsMiddleware** — Patches tool call IDs after LLM completion. Streamed tool call chunks may have different IDs than the patched final result. Use the final `AIMessage` from the Activity result for authoritative tool call IDs.
- **AnthropicPromptCachingMiddleware** — Modifies the request only. No interaction with streaming.

## Event History Considerations

When using Phase 1 (no Redis), individual token events are stored in `NodeActivityOutput.custom_data`, which is part of Temporal's Event History. A 2000-token response adds ~100KB per Activity result. Over hundreds of steps, this can approach Temporal's 50MB Event History limit.

Mitigations:

- Use `SummarizationMiddleware` to keep conversations shorter
- Use `validate_payload_size()` to monitor state size
- Upgrade to Phase 2 (Redis) — only a small summary is stored in Event History

## How It Works Internally

1. `TemporalDeepAgent.create_worker()` calls `wrap_graph_for_streaming()` which wraps each node's `bound` attribute with a `StreamingNodeWrapper`
2. When upstream `_execute_node_impl` calls `node.bound.ainvoke()`, the wrapper intercepts the call
3. The wrapper injects a `TokenCapturingHandler` into `config["callbacks"]`
4. The LLM fires `on_llm_new_token` callbacks as it generates tokens
5. The handler captures each token and publishes it (to Redis or to the Activity's `custom_data`)
6. The original `ainvoke()` result is returned unchanged — the Activity completes normally
7. The client receives tokens via Redis subscription or Temporal Query polling

This design avoids forking upstream `langgraph-temporal` code. The `StreamingNodeWrapper` is transparent to `_execute_node_impl`.
