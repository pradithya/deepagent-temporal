"""TemporalDeepAgent - wraps a Deep Agent for durable execution on Temporal.

Composes `TemporalGraph` from langgraph-temporal, delegating all standard
operations while injecting Deep Agent-specific configuration (worker
affinity, sub-agent config) into the Temporal workflow.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from langgraph.pregel import Pregel
from langgraph.temporal.config import ActivityOptions, SubAgentConfig
from langgraph.temporal.graph import TemporalGraph
from langgraph.temporal.streaming import StreamBackend
from temporalio.client import Client as TemporalClient
from temporalio.client import WorkflowHandle


class TemporalDeepAgent:
    """Wraps a Deep Agent for durable execution on Temporal.

    Composes `TemporalGraph` for standard LangGraph-to-Temporal mapping,
    adding Deep Agent-specific behavior:
    - Worker affinity via worker-specific task queues
    - Sub-agent dispatch via Child Workflows
    - Tool-level human-in-the-loop via interrupt detection

    Args:
        agent: A compiled Pregel graph (output of `create_deep_agent()`).
        client: A Temporal client instance.
        task_queue: Default task queue for the workflow and activities.
        use_worker_affinity: When True, the Workflow discovers a
            worker-specific task queue at startup and pins all Activities
            to that worker. Follows the Temporal worker-specific task
            queues pattern.
        worker_queue_file: Path to persist the worker-specific queue name.
            On restart, the worker re-registers on the same queue so that
            in-flight Activities resume on this worker. If None, a new
            queue name is generated each time.
        subagent_task_queue: Task queue for sub-agent Child Workflows.
            Defaults to `task_queue` if not specified.
        subagent_execution_timeout: Maximum execution time for sub-agent
            Child Workflows.
        node_activity_options: Per-node Activity configuration overrides.
        workflow_execution_timeout: Maximum time for entire workflow
            execution including retries.
        workflow_run_timeout: Maximum time for a single workflow run.
        stream_backend: Backend for streaming events.
    """

    def __init__(
        self,
        agent: Pregel,
        client: TemporalClient,
        *,
        task_queue: str = "deep-agents",
        use_worker_affinity: bool = False,
        worker_queue_file: str | None = None,
        subagent_task_queue: str | None = None,
        subagent_execution_timeout: timedelta | None = None,
        node_activity_options: dict[str, ActivityOptions] | None = None,
        workflow_execution_timeout: timedelta | None = None,
        workflow_run_timeout: timedelta | None = None,
        stream_backend: StreamBackend | None = None,
    ) -> None:
        self._temporal_graph = TemporalGraph(
            agent,
            client,
            task_queue=task_queue,
            node_activity_options=node_activity_options,
            workflow_execution_timeout=workflow_execution_timeout,
            workflow_run_timeout=workflow_run_timeout,
            stream_backend=stream_backend,
        )
        self._use_worker_affinity = use_worker_affinity
        self._worker_queue_file = worker_queue_file
        self._task_queue = task_queue
        self._subagent_task_queue = subagent_task_queue or task_queue
        self._subagent_execution_timeout = subagent_execution_timeout or timedelta(
            minutes=30
        )

    async def ainvoke(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute the Deep Agent as a Temporal Workflow."""
        config = self._inject_temporal_config(config)
        return await self._temporal_graph.ainvoke(input, config, **kwargs)

    async def astream(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream Deep Agent execution events."""
        config = self._inject_temporal_config(config)
        async for event in self._temporal_graph.astream(input, config, **kwargs):
            yield event

    async def astart(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> WorkflowHandle:
        """Start a Deep Agent Workflow (non-blocking)."""
        config = self._inject_temporal_config(config)
        return await self._temporal_graph.astart(input, config, **kwargs)

    async def get_state(self, config: dict[str, Any]) -> dict[str, Any]:
        """Query current agent state."""
        return await self._temporal_graph.get_state(config)

    async def resume(self, config: dict[str, Any], value: Any) -> None:
        """Send a resume Signal for HITL approval."""
        await self._temporal_graph.resume(config, value)

    def create_worker(self, **kwargs: Any) -> Any:
        """Create a Temporal Worker configured for Deep Agent execution.

        When `use_worker_affinity` is True, returns a `WorkerGroup` with
        two workers: one on the shared queue (Workflows + discovery) and
        one on a worker-specific queue (node Activities).

        If `worker_queue_file` was set, the worker-specific queue name is
        persisted to disk so a restarted worker re-registers on the same
        queue (preserving affinity for in-flight Activities).
        """
        from langgraph.temporal.worker import create_worker

        return create_worker(
            self._temporal_graph.graph,
            self._temporal_graph.client,
            self._task_queue,
            use_worker_affinity=self._use_worker_affinity,
            worker_queue_file=self._worker_queue_file,
            **kwargs,
        )

    @classmethod
    async def local(
        cls,
        agent: Pregel,
        *,
        task_queue: str = "deep-agents",
        **kwargs: Any,
    ) -> TemporalDeepAgent:
        """Factory for local development with Temporal test server.

        Args:
            agent: A compiled Pregel graph instance.
            task_queue: Default task queue name.
            **kwargs: Additional arguments passed to TemporalDeepAgent.

        Returns:
            A TemporalDeepAgent configured with the local test server client.
        """
        from temporalio.testing import WorkflowEnvironment

        env = await WorkflowEnvironment.start_local()
        return cls(agent, env.client, task_queue=task_queue, **kwargs)

    def _inject_temporal_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        """Add affinity and sub-agent config to `config["configurable"]`.

        TemporalGraph._build_workflow_input reads from configurable dict.
        We add `use_worker_affinity` and `subagent_config` there so they
        flow into WorkflowInput.
        """
        config = dict(config) if config else {}
        configurable = dict(config.get("configurable", {}))

        if self._use_worker_affinity:
            configurable["use_worker_affinity"] = True

        configurable["subagent_config"] = SubAgentConfig(
            task_queue=self._subagent_task_queue,
            execution_timeout_seconds=(
                self._subagent_execution_timeout.total_seconds()
            ),
        )

        config["configurable"] = configurable
        return config


def create_temporal_deep_agent(
    agent: Pregel,
    client: TemporalClient,
    *,
    task_queue: str = "deep-agents",
    use_worker_affinity: bool = False,
    worker_queue_file: str | None = None,
    subagent_task_queue: str | None = None,
    subagent_execution_timeout: timedelta | None = None,
    node_activity_options: dict[str, ActivityOptions] | None = None,
    workflow_execution_timeout: timedelta | None = None,
    workflow_run_timeout: timedelta | None = None,
    stream_backend: StreamBackend | None = None,
) -> TemporalDeepAgent:
    """Factory function to create a TemporalDeepAgent.

    Convenience wrapper around TemporalDeepAgent constructor.

    Args:
        agent: A compiled Pregel graph (output of `create_deep_agent()`).
        client: A Temporal client instance.
        task_queue: Default task queue.
        use_worker_affinity: Enable worker-specific task queue affinity.
        worker_queue_file: Path to persist the worker-specific queue name.
        subagent_task_queue: Task queue for sub-agent Child Workflows.
        subagent_execution_timeout: Max execution time for sub-agents.
        node_activity_options: Per-node Activity configuration.
        workflow_execution_timeout: Max time for entire workflow.
        workflow_run_timeout: Max time for a single workflow run.
        stream_backend: Backend for streaming events.

    Returns:
        A configured TemporalDeepAgent instance.
    """
    return TemporalDeepAgent(
        agent,
        client,
        task_queue=task_queue,
        use_worker_affinity=use_worker_affinity,
        worker_queue_file=worker_queue_file,
        subagent_task_queue=subagent_task_queue,
        subagent_execution_timeout=subagent_execution_timeout,
        node_activity_options=node_activity_options,
        workflow_execution_timeout=workflow_execution_timeout,
        workflow_run_timeout=workflow_run_timeout,
        stream_backend=stream_backend,
    )
