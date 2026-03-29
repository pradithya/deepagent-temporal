"""Streaming-aware worker factory for Deep Agent Temporal integration.

Provides ``create_streaming_worker`` which wraps graph nodes with
``StreamingNodeWrapper`` before creating the Temporal worker, enabling
token-level capture and optional real-time Redis delivery.
"""

from __future__ import annotations

from typing import Any

from langgraph.pregel import Pregel
from temporalio.client import Client as TemporalClient

from deepagent_temporal.activity import wrap_graph_for_streaming
from deepagent_temporal.streaming import RedisStreamBackend


def create_streaming_worker(
    graph: Pregel,
    client: TemporalClient,
    task_queue: str = "deep-agents",
    *,
    redis_url: str | None = None,
    redis_stream_backend: RedisStreamBackend | None = None,
    use_worker_affinity: bool = False,
    worker_queue_file: str | None = None,
    node_names: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a Temporal Worker with streaming-enabled graph nodes.

    Wraps target graph nodes with ``StreamingNodeWrapper`` for token-level
    capture, then delegates to ``langgraph.temporal.worker.create_worker``.

    Args:
        graph: Compiled LangGraph Pregel graph.
        client: Temporal client instance.
        task_queue: Default task queue name.
        redis_url: Redis connection URL for real-time streaming.
            Convenience alternative to passing a ``redis_stream_backend``.
        redis_stream_backend: Pre-configured Redis backend instance.
        use_worker_affinity: Enable worker-specific task queue affinity.
        worker_queue_file: Path to persist worker queue name.
        node_names: Specific nodes to enable streaming on.
            Defaults to all nodes.
        **kwargs: Additional arguments passed to ``create_worker``.

    Returns:
        A Temporal Worker (or ``WorkerGroup`` if affinity enabled).
    """
    from langgraph.temporal.worker import create_worker

    backend = redis_stream_backend
    if backend is None and redis_url is not None:
        backend = RedisStreamBackend(redis_url=redis_url)

    wrap_graph_for_streaming(
        graph,
        redis_backend=backend,
        node_names=node_names,
    )

    return create_worker(
        graph,
        client,
        task_queue,
        use_worker_affinity=use_worker_affinity,
        worker_queue_file=worker_queue_file,
        **kwargs,
    )
