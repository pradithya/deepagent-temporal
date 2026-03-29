"""Streaming-aware Activity wrappers for Deep Agent nodes.

Provides ``StreamingNodeWrapper`` which wraps a LangGraph node's ``bound``
attribute to inject a ``TokenCapturingHandler`` callback before ``ainvoke()``.
The upstream ``_execute_node_impl`` calls ``node.bound.ainvoke()`` unchanged;
our wrapper intercepts the call to add token capture.

This avoids forking upstream Activity code while enabling token streaming.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.pregel import Pregel

from deepagent_temporal.streaming import (
    RedisStreamBackend,
    TokenCapturingHandler,
    TokenEvent,
)

logger = logging.getLogger(__name__)


class StreamingNodeWrapper:
    """Wraps a LangGraph node's ``bound`` runnable to inject token capture.

    When ``ainvoke()`` is called (by upstream ``_execute_node_impl``), the
    wrapper injects a ``TokenCapturingHandler`` into the config's callback
    list, then delegates to the original runnable. The handler intercepts
    ``on_llm_new_token`` events from the chat model.

    Args:
        original_bound: The original node runnable (``node.bound``).
        node_name: Name of the graph node.
        token_sink: Callback invoked for each ``TokenEvent``.
        redis_backend: Optional Redis backend for real-time publishing.
    """

    def __init__(
        self,
        original_bound: Any,
        node_name: str,
        token_sink: Callable[[TokenEvent], None] | None = None,
        redis_backend: RedisStreamBackend | None = None,
    ) -> None:
        self._original = original_bound
        self._node_name = node_name
        self._token_sink = token_sink
        self._redis_backend = redis_backend

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Invoke the node with token capture callbacks injected.

        Adds a ``TokenCapturingHandler`` to ``config["callbacks"]`` before
        delegating to the original runnable's ``ainvoke``.
        """
        config = dict(config) if config else {}

        # Determine sink: Redis publish or CONFIG_KEY_STREAM passthrough
        if self._redis_backend is not None:
            sink = self._make_redis_sink()
        elif self._token_sink is not None:
            sink = self._token_sink
        else:
            sink = self._make_stream_handler_sink(config)

        # Get heartbeat function if in Activity context
        heartbeat_fn = self._get_heartbeat_fn()

        # Get attempt number for deduplication
        attempt = self._get_attempt_number()

        handler = TokenCapturingHandler(
            self._node_name,
            sink,
            heartbeat_fn=heartbeat_fn,
            attempt=attempt,
        )

        # Inject callback handler into config
        existing_callbacks = config.get("callbacks") or []
        config["callbacks"] = list(existing_callbacks) + [handler]

        return await self._original.ainvoke(input, config, **kwargs)

    def _make_redis_sink(self) -> Callable[[TokenEvent], None]:
        """Create a sink that publishes tokens to Redis."""
        backend = self._redis_backend
        assert backend is not None

        def sink(event: TokenEvent) -> None:
            import asyncio

            workflow_id = self._get_workflow_id()
            if workflow_id:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(backend.publish(workflow_id, event))
                except RuntimeError:
                    pass

        return sink

    def _make_stream_handler_sink(
        self, config: dict[str, Any]
    ) -> Callable[[TokenEvent], None]:
        """Create a sink that routes tokens through CONFIG_KEY_STREAM.

        This makes tokens appear in ``custom_data`` and ultimately in
        the Workflow's ``stream_buffer`` for client polling.
        """
        # The stream handler is stored under the "__pregel_stream" key in
        # the configurable dict. We use the string literal to avoid the
        # deprecation warning from importing CONFIG_KEY_STREAM.
        _CONFIG_KEY_STREAM = "__pregel_stream"

        configurable = config.get("configurable", {})
        stream_fn = configurable.get(_CONFIG_KEY_STREAM)

        def sink(event: TokenEvent) -> None:
            if stream_fn is not None:
                stream_fn(event.to_dict(), "custom")

        return sink

    @staticmethod
    def _get_heartbeat_fn() -> Callable[..., None] | None:
        """Get the Activity heartbeat function if available."""
        try:
            import temporalio.activity as activity

            activity.info()  # Verify we're in Activity context
            return activity.heartbeat
        except (RuntimeError, ImportError):
            return None

    @staticmethod
    def _get_attempt_number() -> int:
        """Get the current Activity attempt number."""
        try:
            import temporalio.activity as activity

            return activity.info().attempt
        except (RuntimeError, ImportError):
            return 1

    @staticmethod
    def _get_workflow_id() -> str | None:
        """Get the current workflow ID from Activity info."""
        try:
            import temporalio.activity as activity

            return activity.info().workflow_id
        except (RuntimeError, ImportError):
            return None

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the original runnable."""
        return getattr(self._original, name)

    def __repr__(self) -> str:
        return f"StreamingNodeWrapper({self._original!r}, node={self._node_name!r})"


def wrap_graph_for_streaming(
    graph: Pregel,
    *,
    token_sink: Callable[[TokenEvent], None] | None = None,
    redis_backend: RedisStreamBackend | None = None,
    node_names: list[str] | None = None,
) -> Pregel:
    """Wrap graph nodes with ``StreamingNodeWrapper`` for token capture.

    Modifies the graph in-place by replacing each target node's ``bound``
    attribute with a ``StreamingNodeWrapper``.

    Args:
        graph: Compiled LangGraph Pregel graph.
        token_sink: Callback for token events (Phase 1 / fallback).
        redis_backend: Redis backend for real-time publishing (Phase 2).
        node_names: Specific nodes to wrap. Defaults to all nodes.

    Returns:
        The modified graph (same object, mutated in-place).
    """
    targets = node_names or list(graph.nodes.keys())

    for name in targets:
        if name not in graph.nodes:
            logger.warning("Node '%s' not found in graph, skipping wrapper", name)
            continue

        node = graph.nodes[name]
        if isinstance(node.bound, StreamingNodeWrapper):
            continue  # Already wrapped

        node.bound = StreamingNodeWrapper(  # type: ignore[assignment]
            node.bound,
            node_name=name,
            token_sink=token_sink,
            redis_backend=redis_backend,
        )

    return graph
