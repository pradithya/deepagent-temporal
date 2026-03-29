"""Token-level streaming support for Deep Agent Temporal integration.

Provides callback-based token capture from LLM calls running inside
Temporal Activities, and a Redis Streams backend for real-time delivery.

Token capture uses LangChain's callback system (``on_llm_new_token``)
rather than ``astream()`` because LangGraph node functions wrapped as
``RunnableCallable`` do not produce token-level chunks from ``astream``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk

logger = logging.getLogger(__name__)


@dataclass
class TokenEvent:
    """A single token or token batch from LLM streaming.

    Attributes:
        token: The token text.
        node_name: The graph node that produced this token.
        index: Token position within this LLM call.
        is_final: Whether this is the last token in the LLM call.
        attempt: Activity attempt number (for deduplication on retry).
    """

    token: str
    node_name: str
    index: int
    is_final: bool = False
    attempt: int = 1

    def to_dict(self) -> dict[str, str | int]:
        """Serialize to a dict safe for Redis Streams.

        Redis Streams only accept ``str``, ``int``, ``float``, or
        ``bytes`` values — not ``bool``. Booleans are converted to
        ``"1"``/``"0"`` strings.
        """
        return {
            "type": "token",
            "token": self.token,
            "node_name": self.node_name,
            "index": self.index,
            "is_final": "1" if self.is_final else "0",
            "attempt": self.attempt,
        }


class TokenCapturingHandler(BaseCallbackHandler):
    """LangChain callback handler that captures LLM tokens.

    Intercepts ``on_llm_new_token`` events from chat models running
    inside a LangGraph node's ``ainvoke()``. Each token is wrapped as a
    ``TokenEvent`` and forwarded to the configured ``publish`` callback.

    Modeled on LangGraph's ``StreamMessagesHandler`` from
    ``langgraph.pregel._messages``.

    Args:
        node_name: Name of the graph node being executed.
        publish: Callback invoked for each token event.
        heartbeat_fn: Optional callable for Activity heartbeats.
            Called every ``heartbeat_interval`` tokens.
        heartbeat_interval: Number of tokens between heartbeats.
        attempt: Activity attempt number for deduplication.
    """

    def __init__(
        self,
        node_name: str,
        publish: Callable[[TokenEvent], None],
        *,
        heartbeat_fn: Callable[..., None] | None = None,
        heartbeat_interval: int = 50,
        attempt: int = 1,
    ) -> None:
        super().__init__()
        self._node_name = node_name
        self._publish = publish
        self._heartbeat_fn = heartbeat_fn
        self._heartbeat_interval = heartbeat_interval
        self._attempt = attempt
        self._index = 0
        self._llm_call_count = 0

    @property
    def token_count(self) -> int:
        """Number of tokens captured so far."""
        return self._index

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Track LLM call count for SummarizationMiddleware handling.

        SummarizationMiddleware may trigger two LLM calls (summarization
        + actual response). We reset the token index on each new call so
        the client receives clean token sequences.
        """
        self._llm_call_count += 1
        self._index = 0

    def on_llm_new_token(  # type: ignore[override]
        self,
        token: str,
        *,
        chunk: ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture a token from the LLM and publish it."""
        if not isinstance(chunk, ChatGenerationChunk):
            return

        event = TokenEvent(
            token=token,
            node_name=self._node_name,
            index=self._index,
            attempt=self._attempt,
        )
        self._index += 1
        self._publish(event)

        # Heartbeat every N tokens to prevent Activity timeout
        if self._heartbeat_fn and self._index % self._heartbeat_interval == 0:
            try:
                self._heartbeat_fn(f"node={self._node_name} tokens={self._index}")
            except RuntimeError:
                # Not in Activity context (e.g., unit tests)
                pass

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Emit a final token event marking the end of the LLM call."""
        event = TokenEvent(
            token="",
            node_name=self._node_name,
            index=self._index,
            is_final=True,
            attempt=self._attempt,
        )
        self._publish(event)


def create_token_handler(
    node_name: str,
    sink: Callable[[TokenEvent], None],
    *,
    heartbeat_fn: Callable[..., None] | None = None,
    heartbeat_interval: int = 50,
    attempt: int = 1,
) -> TokenCapturingHandler:
    """Factory for creating a configured ``TokenCapturingHandler``.

    Args:
        node_name: Name of the graph node being executed.
        sink: Callback that receives each ``TokenEvent``.
        heartbeat_fn: Optional Activity heartbeat callable.
        heartbeat_interval: Tokens between heartbeats.
        attempt: Activity attempt number.
    """
    return TokenCapturingHandler(
        node_name,
        sink,
        heartbeat_fn=heartbeat_fn,
        heartbeat_interval=heartbeat_interval,
        attempt=attempt,
    )


class RedisStreamBackend:
    """Real-time token streaming via Redis Streams.

    Publishes token events to a Redis Stream keyed by workflow ID.
    Clients subscribe via ``XREAD`` for real-time delivery. Temporal
    handles durable state; Redis handles low-latency token delivery.

    Requires ``redis[hiredis]>=5.0.0`` (optional dependency).

    Args:
        redis_url: Redis connection URL.
        channel_prefix: Prefix for Redis Stream keys.
        stream_maxlen: Approximate max entries per stream (``MAXLEN ~``).
        stream_ttl_seconds: TTL for stream keys after completion.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "deepagent:stream:",
        stream_maxlen: int = 5000,
        stream_ttl_seconds: int = 300,
    ) -> None:
        self._redis_url = redis_url
        self._channel_prefix = channel_prefix
        self._stream_maxlen = stream_maxlen
        self._stream_ttl_seconds = stream_ttl_seconds
        self._redis: Any = None

    async def _get_redis(self) -> Any:
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:
                raise ImportError(
                    "RedisStreamBackend requires the 'redis' package. "
                    "Install it with: pip install 'deepagent-temporal[streaming]'"
                ) from exc
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _stream_key(self, workflow_id: str) -> str:
        return f"{self._channel_prefix}{workflow_id}"

    async def publish(
        self, workflow_id: str, event: TokenEvent | dict[str, Any]
    ) -> None:
        """Publish a token event to the Redis Stream.

        Silently logs errors on connection failure (graceful degradation).
        """
        try:
            r = await self._get_redis()
            key = self._stream_key(workflow_id)
            data = event.to_dict() if isinstance(event, TokenEvent) else event
            await r.xadd(
                key,
                data,
                maxlen=self._stream_maxlen,
                approximate=True,
            )
        except Exception:
            logger.warning(
                "Failed to publish token event to Redis for workflow %s",
                workflow_id,
                exc_info=True,
            )

    async def publish_complete(self, workflow_id: str) -> None:
        """Publish a stream-complete sentinel and set TTL."""
        try:
            r = await self._get_redis()
            key = self._stream_key(workflow_id)
            await r.xadd(key, {"type": "stream_complete"})
            await r.expire(key, self._stream_ttl_seconds)
        except Exception:
            logger.warning(
                "Failed to publish stream_complete for workflow %s",
                workflow_id,
                exc_info=True,
            )

    async def subscribe(
        self,
        workflow_id: str,
        *,
        last_id: str = "0-0",
        block_ms: int = 1000,
    ) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to token events from a Redis Stream.

        Yields token event dicts. Stops on ``stream_complete`` sentinel.

        Args:
            workflow_id: The workflow to subscribe to.
            last_id: Redis Stream ID to start reading from.
            block_ms: Milliseconds to block on XREAD.
        """
        r = await self._get_redis()
        key = self._stream_key(workflow_id)
        current_id = last_id

        while True:
            entries = await r.xread({key: current_id}, block=block_ms, count=100)
            if not entries:
                continue
            for _stream_name, messages in entries:
                for msg_id, data in messages:
                    current_id = msg_id
                    if data.get("type") == "stream_complete":
                        return
                    yield data

    async def cleanup(self, workflow_id: str) -> None:
        """Delete the Redis Stream for a workflow."""
        try:
            r = await self._get_redis()
            await r.delete(self._stream_key(workflow_id))
        except Exception:
            logger.warning(
                "Failed to cleanup Redis stream for workflow %s",
                workflow_id,
                exc_info=True,
            )

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
