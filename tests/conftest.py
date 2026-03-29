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
