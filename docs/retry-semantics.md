# Retry Semantics

This document explains how retries work in `deepagent-temporal` and how to configure them to avoid double-retry costs.

## The Double-Retry Problem

Two retry layers exist when running LLM agents on Temporal:

1. **LLM SDK retries** — The LLM client SDK (e.g., `langchain-anthropic`, `openai`) retries on rate limits (HTTP 429) and transient errors (5xx) internally, with exponential backoff.
2. **Temporal Activity retries** — Temporal retries failed Activities according to the Activity's `RetryPolicy`. If an LLM call fails after the SDK exhausts its retry budget, Temporal retries the entire Activity — re-invoking the model from scratch.

Without configuration, both layers retry independently:

```
LLM SDK retry 1 → fail
LLM SDK retry 2 → fail
LLM SDK retry 3 ��� fail (SDK budget exhausted)
Activity fails → Temporal retry 1:
  LLM SDK retry 1 → fail
  LLM SDK retry 2 → fail
  LLM SDK retry 3 → fail
Activity fails → Temporal retry 2:
  ...
```

Each Temporal retry re-invokes the model. This can cause unexpected API costs — especially with expensive models.

## Recommended Configuration

### Option A: Disable Temporal Retries for LLM Activities (Recommended)

Let the LLM SDK handle its own retries. Set `max_attempts=1` on LLM-calling Activities so Temporal does not retry them:

```python
from deepagent_temporal import TemporalDeepAgent

temporal_agent = TemporalDeepAgent(
    agent, client,
    task_queue="my-agents",
    node_retry_policies=TemporalDeepAgent.recommended_retry_policies(),
)
```

`recommended_retry_policies()` returns:

| Node | `max_attempts` | Rationale |
|---|---|---|
| `call_model` | 1 | LLM SDK handles rate-limit/transient retries internally |
| `tools` | 1 | Tool side-effects (file writes, shell commands) are not idempotent |

### Option B: Disable SDK Retries, Use Temporal Retries

Alternatively, disable retries in the LLM SDK and let Temporal handle all retries. This gives you a single retry layer with Temporal's observability (retry attempts visible in Event History).

```python
from langchain_anthropic import ChatAnthropic
from langgraph.temporal.config import RetryPolicyConfig

# Disable SDK retries
model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_retries=0,  # Disable SDK retries
)

# Let Temporal handle retries
temporal_agent = TemporalDeepAgent(
    agent, client,
    node_retry_policies={
        "call_model": RetryPolicyConfig(
            max_attempts=5,
            initial_interval_seconds=2.0,
            backoff_coefficient=2.0,
            max_interval_seconds=60.0,
            non_retryable_error_types=["ContextOverflowError"],
        ),
        "tools": RetryPolicyConfig(max_attempts=1),
    },
)
```

### Option C: Custom Per-Node Policies

For advanced use cases, configure different retry policies per node:

```python
from deepagent_temporal import RetryPolicyConfig

temporal_agent = TemporalDeepAgent(
    agent, client,
    node_retry_policies={
        # LLM calls: no Temporal retry (SDK handles it)
        "call_model": RetryPolicyConfig(max_attempts=1),
        # Read-only tools: safe to retry
        "tools": RetryPolicyConfig(
            max_attempts=3,
            initial_interval_seconds=1.0,
            backoff_coefficient=2.0,
        ),
    },
)
```

## Activity Timeouts

In addition to retry policies, configure Activity timeouts to bound execution time:

```python
from langgraph.temporal.config import ActivityOptions

temporal_agent = TemporalDeepAgent(
    agent, client,
    node_activity_options={
        "call_model": ActivityOptions(
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(seconds=60),
        ),
        "tools": ActivityOptions(
            start_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(seconds=60),
        ),
    },
)
```

| Timeout | `call_model` | `tools` | Purpose |
|---|---|---|---|
| `start_to_close_timeout` | 5 min | 30 min | Max time for a single execution |
| `heartbeat_timeout` | 60s | 60s | Detect stuck activities |
| `schedule_to_close_timeout` | 10 min | 60 min | Max time including retries |

## Cost Awareness

Each Temporal retry of an LLM Activity **re-invokes the model**. With the recommended `max_attempts=1` configuration:

- A failed LLM call costs only the SDK's internal retries (typically 3 attempts with backoff).
- Without this setting, Temporal's default unlimited retry policy could cause runaway costs.

Monitor your LLM API usage dashboards when experimenting with retry configurations.
