"""Tests for Redis Streams backend.

Unit tests use mocked Redis. Integration tests require a running Redis
instance (skipped if unavailable — run via ``make test_integration_docker``).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from deepagent_temporal.streaming import RedisStreamBackend, TokenEvent


class TestRedisStreamBackendInit:
    def test_default_config(self) -> None:
        backend = RedisStreamBackend()
        assert backend._redis_url == "redis://localhost:6379"
        assert backend._channel_prefix == "deepagent:stream:"
        assert backend._stream_maxlen == 5000
        assert backend._stream_ttl_seconds == 300

    def test_custom_config(self) -> None:
        backend = RedisStreamBackend(
            redis_url="redis://custom:6380",
            channel_prefix="test:",
            stream_maxlen=100,
            stream_ttl_seconds=60,
        )
        assert backend._redis_url == "redis://custom:6380"
        assert backend._channel_prefix == "test:"
        assert backend._stream_maxlen == 100
        assert backend._stream_ttl_seconds == 60

    def test_stream_key(self) -> None:
        backend = RedisStreamBackend(channel_prefix="prefix:")
        assert backend._stream_key("wf-123") == "prefix:wf-123"


class TestRedisStreamBackendPublish:
    @pytest.mark.asyncio
    async def test_publish_token_event(self) -> None:
        mock_redis = AsyncMock()
        backend = RedisStreamBackend()
        backend._redis = mock_redis

        event = TokenEvent(
            token="hello",
            node_name="call_model",
            index=0,
        )
        await backend.publish("wf-1", event)

        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "deepagent:stream:wf-1"
        assert call_args[0][1]["type"] == "token"
        assert call_args[0][1]["token"] == "hello"

    @pytest.mark.asyncio
    async def test_publish_dict_event(self) -> None:
        mock_redis = AsyncMock()
        backend = RedisStreamBackend()
        backend._redis = mock_redis

        await backend.publish("wf-1", {"type": "custom", "data": "test"})

        mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_handles_connection_error(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.xadd.side_effect = ConnectionError("Redis down")
        backend = RedisStreamBackend()
        backend._redis = mock_redis

        # Should not raise
        await backend.publish("wf-1", TokenEvent("t", "n", 0))


class TestRedisStreamBackendComplete:
    @pytest.mark.asyncio
    async def test_publish_complete(self) -> None:
        mock_redis = AsyncMock()
        backend = RedisStreamBackend(stream_ttl_seconds=120)
        backend._redis = mock_redis

        await backend.publish_complete("wf-1")

        # Should publish sentinel and set TTL
        mock_redis.xadd.assert_called_once()
        sentinel_data = mock_redis.xadd.call_args[0][1]
        assert sentinel_data["type"] == "stream_complete"
        mock_redis.expire.assert_called_once_with("deepagent:stream:wf-1", 120)


class TestRedisStreamBackendSubscribe:
    @pytest.mark.asyncio
    async def test_subscribe_yields_events(self) -> None:
        mock_redis = AsyncMock()
        backend = RedisStreamBackend()
        backend._redis = mock_redis

        # Simulate xread returning token events then stream_complete
        mock_redis.xread.side_effect = [
            [
                (
                    "deepagent:stream:wf-1",
                    [
                        ("1-0", {"type": "token", "token": "Hello"}),
                        ("2-0", {"type": "token", "token": " World"}),
                    ],
                )
            ],
            [
                (
                    "deepagent:stream:wf-1",
                    [
                        ("3-0", {"type": "stream_complete"}),
                    ],
                )
            ],
        ]

        events = []
        async for event in backend.subscribe("wf-1"):
            events.append(event)

        assert len(events) == 2
        assert events[0]["token"] == "Hello"
        assert events[1]["token"] == " World"


class TestRedisStreamBackendCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_deletes_key(self) -> None:
        mock_redis = AsyncMock()
        backend = RedisStreamBackend()
        backend._redis = mock_redis

        await backend.cleanup("wf-1")

        mock_redis.delete.assert_called_once_with("deepagent:stream:wf-1")

    @pytest.mark.asyncio
    async def test_cleanup_handles_error(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.delete.side_effect = ConnectionError("Redis down")
        backend = RedisStreamBackend()
        backend._redis = mock_redis

        # Should not raise
        await backend.cleanup("wf-1")


class TestRedisStreamBackendClose:
    @pytest.mark.asyncio
    async def test_close(self) -> None:
        mock_redis = AsyncMock()
        backend = RedisStreamBackend()
        backend._redis = mock_redis

        await backend.close()

        mock_redis.aclose.assert_called_once()
        assert backend._redis is None

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self) -> None:
        backend = RedisStreamBackend()
        # Should not raise
        await backend.close()


class TestRedisImportError:
    @pytest.mark.asyncio
    async def test_raises_import_error_without_redis(self) -> None:
        backend = RedisStreamBackend()
        backend._redis = None

        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            with pytest.raises(ImportError, match="redis"):
                await backend._get_redis()


# ---------------------------------------------------------------------------
# Integration tests — require a real Redis (via docker-compose)
# Run with: make test_integration_docker
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRedisStreamIntegrationPublishSubscribe:
    """End-to-end publish/subscribe through a real Redis instance."""

    @pytest.mark.asyncio
    async def test_publish_and_subscribe_round_trip(
        self, redis_backend: RedisStreamBackend
    ) -> None:
        """Publish tokens, subscribe, verify they arrive in order."""
        workflow_id = f"test-wf-{uuid.uuid4().hex[:8]}"

        tokens = ["Hello", " ", "World", "!"]

        # Publish tokens in background
        async def publisher() -> None:
            await asyncio.sleep(0.05)  # let subscriber start first
            for i, tok in enumerate(tokens):
                event = TokenEvent(
                    token=tok,
                    node_name="call_model",
                    index=i,
                    attempt=1,
                )
                await redis_backend.publish(workflow_id, event)
            await redis_backend.publish_complete(workflow_id)

        # Subscribe and collect
        received: list[dict[str, Any]] = []

        async def subscriber() -> None:
            async for event in redis_backend.subscribe(workflow_id, block_ms=500):
                received.append(event)

        await asyncio.gather(publisher(), subscriber())

        assert len(received) == len(tokens)
        received_tokens = [e["token"] for e in received]
        assert received_tokens == tokens

    @pytest.mark.asyncio
    async def test_cleanup_removes_stream(
        self, redis_backend: RedisStreamBackend
    ) -> None:
        """Verify cleanup deletes the Redis stream key."""
        workflow_id = f"test-cleanup-{uuid.uuid4().hex[:8]}"

        await redis_backend.publish(
            workflow_id,
            TokenEvent(token="x", node_name="n", index=0),
        )

        r = await redis_backend._get_redis()
        assert await r.exists(redis_backend._stream_key(workflow_id))

        await redis_backend.cleanup(workflow_id)

        assert not await r.exists(redis_backend._stream_key(workflow_id))

    @pytest.mark.asyncio
    async def test_maxlen_caps_stream_size(
        self, redis_backend: RedisStreamBackend
    ) -> None:
        """Verify MAXLEN ~ keeps the stream bounded.

        ``MAXLEN ~`` is approximate — Redis only trims when it can
        remove an entire macro-node (typically 100 entries). We use
        a large enough volume to trigger trimming reliably.
        """
        redis_backend._stream_maxlen = 100
        workflow_id = f"test-maxlen-{uuid.uuid4().hex[:8]}"

        # Publish 500 events to reliably trigger approximate trimming
        for i in range(500):
            await redis_backend.publish(
                workflow_id,
                TokenEvent(token=f"t{i}", node_name="n", index=i),
            )

        r = await redis_backend._get_redis()
        key = redis_backend._stream_key(workflow_id)
        length = await r.xlen(key)
        # Approximate trimming should keep it well under 500
        assert length < 500
        # But it won't be exactly 100 — give generous room
        assert length <= 250

        await redis_backend.cleanup(workflow_id)

    @pytest.mark.asyncio
    async def test_ttl_set_on_complete(self, redis_backend: RedisStreamBackend) -> None:
        """Verify publish_complete sets a TTL on the stream key."""
        workflow_id = f"test-ttl-{uuid.uuid4().hex[:8]}"

        await redis_backend.publish(
            workflow_id,
            TokenEvent(token="x", node_name="n", index=0),
        )
        await redis_backend.publish_complete(workflow_id)

        r = await redis_backend._get_redis()
        key = redis_backend._stream_key(workflow_id)
        ttl = await r.ttl(key)
        assert ttl > 0
        assert ttl <= redis_backend._stream_ttl_seconds

        await redis_backend.cleanup(workflow_id)

    @pytest.mark.asyncio
    async def test_multiple_workflows_isolated(
        self, redis_backend: RedisStreamBackend
    ) -> None:
        """Verify streams for different workflows don't interfere."""
        wf1 = f"test-iso-1-{uuid.uuid4().hex[:8]}"
        wf2 = f"test-iso-2-{uuid.uuid4().hex[:8]}"

        await redis_backend.publish(
            wf1, TokenEvent(token="wf1-tok", node_name="n", index=0)
        )
        await redis_backend.publish(
            wf2, TokenEvent(token="wf2-tok", node_name="n", index=0)
        )
        await redis_backend.publish_complete(wf1)
        await redis_backend.publish_complete(wf2)

        wf1_events: list[dict[str, Any]] = []
        async for event in redis_backend.subscribe(wf1, block_ms=500):
            wf1_events.append(event)

        wf2_events: list[dict[str, Any]] = []
        async for event in redis_backend.subscribe(wf2, block_ms=500):
            wf2_events.append(event)

        assert len(wf1_events) == 1
        assert wf1_events[0]["token"] == "wf1-tok"
        assert len(wf2_events) == 1
        assert wf2_events[0]["token"] == "wf2-tok"

        await redis_backend.cleanup(wf1)
        await redis_backend.cleanup(wf2)

    @pytest.mark.asyncio
    async def test_deduplication_by_attempt(
        self, redis_backend: RedisStreamBackend
    ) -> None:
        """Verify that attempt number is preserved for client-side dedup."""
        workflow_id = f"test-dedup-{uuid.uuid4().hex[:8]}"

        # Simulate two attempts publishing to the same stream
        for attempt in [1, 2]:
            for i in range(3):
                tok_event = TokenEvent(
                    token=f"tok{i}",
                    node_name="call_model",
                    index=i,
                    attempt=attempt,
                )
                await redis_backend.publish(workflow_id, tok_event)
        await redis_backend.publish_complete(workflow_id)

        events: list[dict[str, Any]] = []
        async for received in redis_backend.subscribe(workflow_id, block_ms=500):
            events.append(received)

        # All 6 events arrive (3 per attempt)
        assert len(events) == 6
        # Client can deduplicate by filtering for latest attempt
        attempt_2_events = [e for e in events if e.get("attempt") == "2"]
        assert len(attempt_2_events) == 3

        await redis_backend.cleanup(workflow_id)
