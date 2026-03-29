"""Shared fixtures for deepagent-temporal tests."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest


@pytest.fixture
def task_queue() -> str:
    """Unique task queue name per test to avoid collisions."""
    return f"test-queue-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def temporal_env() -> Any:
    """Start a time-skipping test server or connect to an external one.

    When ``TEMPORAL_ADDRESS`` is set, connect to the real Temporal server at
    that address (e.g. ``localhost:7233`` via Docker Compose).  Otherwise fall
    back to the in-process time-skipping test server.
    """
    temporal_address = os.environ.get("TEMPORAL_ADDRESS", "").strip()
    if temporal_address:
        from temporalio.client import Client as TemporalClient

        client = await TemporalClient.connect(temporal_address)
        yield client
    else:
        from temporalio.testing import WorkflowEnvironment

        env = await WorkflowEnvironment.start_time_skipping()
        yield env
        await env.shutdown()


@pytest.fixture
def temporal_client(temporal_env: Any) -> Any:
    """Return the Temporal client from the test environment."""
    if hasattr(temporal_env, "client"):
        return temporal_env.client
    return temporal_env


@pytest.fixture
async def redis_backend() -> Any:
    """Return a RedisStreamBackend connected to the test Redis.

    Uses ``REDIS_URL`` env var (set by ``test_integration_docker``).
    Skips the test if Redis is not available.
    """
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if not redis_url:
        pytest.skip("REDIS_URL not set — run with make test_integration_docker")

    from deepagent_temporal.streaming import RedisStreamBackend

    backend = RedisStreamBackend(redis_url=redis_url, stream_ttl_seconds=30)
    # Verify connectivity
    try:
        r = await backend._get_redis()
        await r.ping()
    except Exception as exc:
        pytest.skip(f"Redis not reachable at {redis_url}: {exc}")

    yield backend
    await backend.close()
