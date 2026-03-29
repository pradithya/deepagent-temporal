"""Tests for TemporalDeepAgent wrapper."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.temporal.config import SubAgentConfig, WorkflowOutput
from langgraph.temporal.converter import GraphRegistry


class TestTemporalDeepAgentConstruction:
    def setup_method(self) -> None:
        GraphRegistry.reset()

    def test_default_construction(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        agent = TemporalDeepAgent(mock_graph, mock_client)

        assert agent._use_worker_affinity is False
        assert agent._subagent_task_queue == "deep-agents"
        assert agent._subagent_execution_timeout == timedelta(minutes=30)

    def test_custom_construction(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        agent = TemporalDeepAgent(
            mock_graph,
            mock_client,
            task_queue="my-queue",
            use_worker_affinity=True,
            subagent_task_queue="sub-queue",
            subagent_execution_timeout=timedelta(minutes=10),
        )

        assert agent._use_worker_affinity is True
        assert agent._subagent_task_queue == "sub-queue"
        assert agent._subagent_execution_timeout == timedelta(minutes=10)

    def test_subagent_queue_defaults_to_task_queue(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        agent = TemporalDeepAgent(mock_graph, mock_client, task_queue="custom-q")

        assert agent._subagent_task_queue == "custom-q"


class TestTemporalDeepAgentConfigInjection:
    def setup_method(self) -> None:
        GraphRegistry.reset()

    def test_inject_worker_affinity(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        agent = TemporalDeepAgent(
            mock_graph,
            mock_client,
            use_worker_affinity=True,
        )

        config = agent._inject_temporal_config(None)
        assert config["configurable"]["use_worker_affinity"] is True

    def test_inject_subagent_config(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        agent = TemporalDeepAgent(
            mock_graph,
            mock_client,
            use_worker_affinity=True,
            subagent_task_queue="sub-q",
            subagent_execution_timeout=timedelta(minutes=15),
        )

        config = agent._inject_temporal_config(None)
        sac = config["configurable"]["subagent_config"]
        assert isinstance(sac, SubAgentConfig)
        assert sac.task_queue == "sub-q"
        assert sac.execution_timeout_seconds == 900.0

    def test_inject_preserves_existing_config(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        agent = TemporalDeepAgent(mock_graph, mock_client)

        existing_config: dict[str, Any] = {
            "configurable": {"thread_id": "my-thread"},
            "recursion_limit": 50,
        }
        config = agent._inject_temporal_config(existing_config)
        assert config["configurable"]["thread_id"] == "my-thread"
        assert config["recursion_limit"] == 50
        assert "subagent_config" in config["configurable"]

    def test_inject_no_affinity_when_disabled(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        agent = TemporalDeepAgent(mock_graph, mock_client)

        config = agent._inject_temporal_config(None)
        assert "use_worker_affinity" not in config["configurable"]


class TestTemporalDeepAgentDelegation:
    def setup_method(self) -> None:
        GraphRegistry.reset()

    @pytest.mark.asyncio
    async def test_ainvoke_delegates(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_graph.interrupt_before_nodes = []
        mock_graph.interrupt_after_nodes = []
        mock_client = AsyncMock()
        mock_client.execute_workflow = AsyncMock(
            return_value=WorkflowOutput(
                channel_values={"messages": ["result"]},
                step=1,
            )
        )

        agent = TemporalDeepAgent(mock_graph, mock_client)
        result = await agent.ainvoke(
            {"messages": ["hello"]},
            {"configurable": {"thread_id": "test-thread"}},
        )

        assert result == {"messages": ["result"]}
        mock_client.execute_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_delegates(self) -> None:
        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        mock_handle = AsyncMock()
        mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)

        agent = TemporalDeepAgent(mock_graph, mock_client)

        config = {"configurable": {"thread_id": "test-thread"}}
        await agent.resume(config, "approved")
        mock_handle.signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_state_delegates(self) -> None:
        from langgraph.temporal.config import StateQueryResult

        from deepagent_temporal.agent import TemporalDeepAgent

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        mock_handle = AsyncMock()
        mock_handle.query = AsyncMock(
            return_value=StateQueryResult(
                channel_values={"messages": ["hi"]},
                channel_versions={},
                versions_seen={},
                step=1,
                status="running",
                interrupts=[],
            )
        )
        mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)

        agent = TemporalDeepAgent(mock_graph, mock_client)

        config = {"configurable": {"thread_id": "test-thread"}}
        state = await agent.get_state(config)
        assert state["values"] == {"messages": ["hi"]}


class TestCreateTemporalDeepAgent:
    def setup_method(self) -> None:
        GraphRegistry.reset()

    def test_factory_function(self) -> None:
        from deepagent_temporal.agent import (
            TemporalDeepAgent,
            create_temporal_deep_agent,
        )

        mock_graph = MagicMock()
        mock_graph.name = "test"
        mock_client = MagicMock()

        agent = create_temporal_deep_agent(
            mock_graph,
            mock_client,
            task_queue="my-queue",
            use_worker_affinity=True,
        )

        assert isinstance(agent, TemporalDeepAgent)
        assert agent._use_worker_affinity is True
