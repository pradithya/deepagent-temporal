# Serialization Boundary

This document describes how state crosses between LangGraph and Temporal, what gets serialized, and how to handle large payloads.

## How State Flows Through Temporal

When a Deep Agent runs as a Temporal Workflow, state is serialized at every Activity boundary:

1. **Workflow dispatches Activity** — the current channel state is serialized into the Activity input (via `NodeActivityInput`).
2. **Activity executes a graph node** — the node runs (LLM call, tool execution, etc.) and produces output.
3. **Activity returns result** — the output is serialized into `NodeActivityOutput` and recorded in Temporal's Event History.
4. **Workflow applies writes** — the Workflow deserializes the output and updates channel state.

Every crossing between Workflow and Activity is a serialization boundary.

## Serialization Format

`langgraph-temporal` uses JSON serialization by default (via Temporal's standard `DataConverter`). The state dict is serialized with Python's `json` module, with LangChain message types converted to their dict representations.

### Supported Types

| Type | Serialization | Notes |
|---|---|---|
| `str`, `int`, `float`, `bool`, `None` | Native JSON | No issues |
| `list`, `dict` | Native JSON | Nested values must also be serializable |
| `HumanMessage`, `AIMessage`, `ToolMessage` | Dict via LangChain's `.dict()` | Automatically handled by `langgraph-temporal` |
| `datetime` | ISO 8601 string | Via custom encoder |
| `bytes` | Base64-encoded string | Via `LargePayloadCodec` |
| Arbitrary Python objects | **Not supported** | Will raise `TypeError` at serialization time |

### What Breaks

- **Lambda functions or closures** in state channels — these cannot be serialized.
- **Open file handles or database connections** — not serializable.
- **Custom objects without `__dict__`** or JSON-incompatible attributes.
- **Circular references** in state — JSON does not support them.

If you store non-serializable objects in state channels, the Activity will fail with a `TypeError` at the serialization boundary. This is by design — failing loudly is better than silent corruption.

## Payload Size Limits

Temporal has a hard limit on event payload size:

- **Default**: ~2 MB per event (configurable per namespace via `frontend.maxPayloadSize`)
- **Recommended maximum**: 1 MB per event to leave headroom for Temporal metadata

### Where Size Matters

The largest payloads in a Deep Agent workflow are typically:

1. **`NodeActivityOutput` from `call_model`** — includes the full state with message history.
2. **`NodeActivityOutput` from `tools`** — includes tool call results, which may contain file contents.
3. **Continue-as-new state** — the full channel state is serialized when the workflow restarts.

A 500-message conversation with average 200 tokens per message produces roughly 400 KB of serialized state. This is within limits, but long-running agents can exceed 2 MB.

### Payload Size Guard

Use `validate_payload_size()` to check state size before it hits Temporal:

```python
from deepagent_temporal import validate_payload_size

# In a custom middleware or before invoking the agent:
state = {"messages": conversation_history}
validate_payload_size(state)  # Warns at 1 MB, raises at 2 MB

# Custom thresholds:
validate_payload_size(
    state,
    warn_bytes=500_000,     # Warn at 500 KB
    error_bytes=1_500_000,  # Error at 1.5 MB
)
```

The guard raises `PayloadTooLargeError` when state exceeds the error threshold.

## Claim-Check Pattern for Large State

When state exceeds payload limits, use the **claim-check pattern**: store the full state in external storage and pass only a reference through the workflow.

### How It Works

```
Agent State (large)          External Store (S3/Redis)
+-------------------+       +------------------------+
| messages: [...]   | ----> | s3://bucket/state/abc  |
| 2.5 MB            |       | (full state stored)    |
+-------------------+       +------------------------+
        |
        v
Temporal Event History
+-------------------+
| state_ref: "abc"  |
| 50 bytes          |
+-------------------+
```

### Recommended Implementation

1. **Before Activity return**: If state exceeds threshold, upload to external store (S3, GCS, Redis) and replace the large field with a reference.
2. **Before Activity execution**: If state contains a reference, fetch from external store and restore the full field.

This pattern is not yet built into `deepagent-temporal` (tracked as a future enhancement). For now, implement it in a custom middleware:

```python
import boto3
import json

s3 = boto3.client("s3")
BUCKET = "my-agent-state"
THRESHOLD = 1_000_000  # 1 MB

def offload_if_large(state: dict, key: str, thread_id: str) -> dict:
    """Replace a large state field with an S3 reference."""
    payload = json.dumps(state[key], default=str).encode()
    if len(payload) > THRESHOLD:
        s3_key = f"state/{thread_id}/{key}"
        s3.put_object(Bucket=BUCKET, Key=s3_key, Body=payload)
        state[key] = {"__ref__": f"s3://{BUCKET}/{s3_key}"}
    return state

def restore_if_ref(state: dict, key: str) -> dict:
    """Restore a state field from an S3 reference."""
    value = state.get(key)
    if isinstance(value, dict) and "__ref__" in value:
        uri = value["__ref__"]
        bucket, s3_key = uri.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        state[key] = json.loads(obj["Body"].read())
    return state
```

### Conversation History Offloading

Deep Agents' `SummarizationMiddleware` already addresses growing conversation history by auto-compacting messages and writing the full history to a file (`/conversation_history/{thread_id}.md`). This is the recommended first line of defense.

If summarization is not sufficient, offload the `messages` channel to external storage using the claim-check pattern above.

## Encryption

For sensitive state (API responses, user data, PII), use `langgraph-temporal`'s `EncryptionCodec`:

```python
from langgraph.temporal.codec import EncryptionCodec

# The codec encrypts all payloads in Temporal Event History
codec = EncryptionCodec(key=b"your-32-byte-encryption-key-here")
```

This ensures that state is encrypted at rest in Temporal's persistence layer. LLM API keys and tool credentials should never appear in state — they must be resolved at the worker level (see `docs/REQUIREMENTS.md` NFR-05).
