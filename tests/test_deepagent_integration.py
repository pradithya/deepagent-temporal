"""Integration tests for Deep Agent Temporal features.

Tests the full flow: Activity -> context var -> NodeActivityOutput ->
Workflow -> Child Workflow -> result back.

All tests require a local Temporal server.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.temporal.config import (
    NodeActivityOutput,
    SubAgentConfig,
    WorkflowInput,
    WorkflowOutput,
)
from langgraph.temporal.converter import GraphRegistry
from langgraph.temporal.workflow import LangGraphWorkflow


@pytest.mark.integration
class TestChildWorkflowDispatch:
    """Test that child_workflow_requests in Activity output triggers
    Child Workflow dispatch."""

    def setup_method(self) -> None:
        GraphRegistry.reset()

    @pytest.mark.asyncio
    async def test_process_child_workflow_requests_dispatches(self) -> None:
        """When results contain child_workflow_requests, the workflow
        dispatches Child Workflows and injects results back."""
        wf = LangGraphWorkflow.__new__(LangGraphWorkflow)
        wf.step = 5
        wf._sticky_task_queue = None
        wf._subagent_config = SubAgentConfig(
            task_queue="sub-queue",
            execution_timeout_seconds=60.0,
        )

        results = [
            NodeActivityOutput(
                node_name="tools",
                writes=[("messages", "existing")],
                child_workflow_requests=[
                    {
                        "subagent_type": "researcher",
                        "instruction": "Research AI",
                        "tool_call_id": "tc-1",
                        "initial_state": {"messages": []},
                        "graph_definition_ref": "subagent:researcher",
                    }
                ],
            ),
        ]

        mock_output = WorkflowOutput(
            channel_values={"messages": ["Research complete"]},
            step=3,
        )

        with patch.object(
            wf,
            "_dispatch_child_workflow",
            new_callable=AsyncMock,
            return_value=mock_output,
        ):
            wf_input = WorkflowInput(
                graph_definition_ref="ref",
                subagent_config=SubAgentConfig(task_queue="sub-queue"),
            )
            await wf._process_child_workflow_requests(results, wf_input)

        # Original write preserved + tool message injected
        assert len(results[0].writes) == 2
        assert results[0].writes[0] == ("messages", "existing")
        tool_msg = results[0].writes[1]
        assert tool_msg[0] == "messages"
        assert tool_msg[1]["type"] == "tool"
        assert tool_msg[1]["tool_call_id"] == "tc-1"
        assert "Research complete" in tool_msg[1]["content"]


@pytest.mark.integration
class TestChildWorkflowFailure:
    """Test that Child Workflow failures are converted to error messages."""

    def setup_method(self) -> None:
        GraphRegistry.reset()

    @pytest.mark.asyncio
    async def test_child_workflow_failure_becomes_error_message(self) -> None:
        wf = LangGraphWorkflow.__new__(LangGraphWorkflow)
        wf.step = 1
        wf._sticky_task_queue = None
        wf._subagent_config = SubAgentConfig(task_queue="sub-queue")

        results = [
            NodeActivityOutput(
                node_name="tools",
                writes=[],
                child_workflow_requests=[
                    {
                        "subagent_type": "flaky-agent",
                        "instruction": "Do something",
                        "tool_call_id": "tc-fail",
                        "initial_state": {},
                        "graph_definition_ref": "subagent:flaky",
                    }
                ],
            ),
        ]

        with patch.object(
            wf,
            "_dispatch_child_workflow",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Child workflow crashed"),
        ):
            wf_input = WorkflowInput(graph_definition_ref="ref")
            await wf._process_child_workflow_requests(results, wf_input)

        assert len(results[0].writes) == 1
        tool_msg = results[0].writes[0]
        assert tool_msg[0] == "messages"
        assert tool_msg[1]["type"] == "tool"
        assert "failed" in tool_msg[1]["content"]
        assert "flaky-agent" in tool_msg[1]["content"]
        assert tool_msg[1]["tool_call_id"] == "tc-fail"


@pytest.mark.integration
class TestParallelChildWorkflows:
    """Test that multiple child_workflow_requests are dispatched concurrently."""

    def setup_method(self) -> None:
        GraphRegistry.reset()

    @pytest.mark.asyncio
    async def test_parallel_dispatch(self) -> None:
        wf = LangGraphWorkflow.__new__(LangGraphWorkflow)
        wf.step = 2
        wf._sticky_task_queue = None
        wf._subagent_config = SubAgentConfig(task_queue="sub-queue")

        results = [
            NodeActivityOutput(
                node_name="tools",
                writes=[],
                child_workflow_requests=[
                    {
                        "subagent_type": "researcher",
                        "instruction": "Research",
                        "tool_call_id": "tc-1",
                        "initial_state": {},
                        "graph_definition_ref": "ref-1",
                    },
                    {
                        "subagent_type": "coder",
                        "instruction": "Code",
                        "tool_call_id": "tc-2",
                        "initial_state": {},
                        "graph_definition_ref": "ref-2",
                    },
                ],
            ),
        ]

        call_order: list[str] = []

        async def mock_dispatch(
            req: dict[str, Any], wf_input: WorkflowInput, idx: int
        ) -> WorkflowOutput:
            call_order.append(req["subagent_type"])
            return WorkflowOutput(
                channel_values={"messages": [f"Result from {req['subagent_type']}"]},
                step=1,
            )

        with patch.object(wf, "_dispatch_child_workflow", side_effect=mock_dispatch):
            wf_input = WorkflowInput(graph_definition_ref="ref")
            await wf._process_child_workflow_requests(results, wf_input)

        # Both dispatched
        assert "researcher" in call_order
        assert "coder" in call_order

        # Both results injected
        assert len(results[0].writes) == 2
        tool_msgs = [w[1] for w in results[0].writes]
        tool_call_ids = [m["tool_call_id"] for m in tool_msgs]
        assert "tc-1" in tool_call_ids
        assert "tc-2" in tool_call_ids


@pytest.mark.integration
class TestWorkerAffinityRouting:
    """Test that discovered sticky_task_queue overrides per-node routing."""

    def setup_method(self) -> None:
        GraphRegistry.reset()

    def test_discovered_queue_overrides_per_node(self) -> None:
        """After discovery, _sticky_task_queue overrides per-node queues."""
        wf = LangGraphWorkflow.__new__(LangGraphWorkflow)
        wf._node_task_queues = {"gpu_node": "gpu-queue"}
        wf._node_activity_options = {}
        wf._node_retry_policies = {}
        wf._sticky_task_queue = "discovered-worker-queue"
        wf._subagent_config = None

        assert wf._task_queue_for_node("gpu_node") == "discovered-worker-queue"
        assert wf._task_queue_for_node("any_other") == "discovered-worker-queue"

    def test_no_affinity_uses_per_node(self) -> None:
        """Without affinity, per-node overrides work normally."""
        wf = LangGraphWorkflow.__new__(LangGraphWorkflow)
        wf._node_task_queues = {"gpu_node": "gpu-queue"}
        wf._node_activity_options = {}
        wf._node_retry_policies = {}
        wf._sticky_task_queue = None
        wf._subagent_config = None

        assert wf._task_queue_for_node("gpu_node") == "gpu-queue"
        assert wf._task_queue_for_node("other") is None

    def test_use_worker_affinity_in_workflow_input(self) -> None:
        """WorkflowInput carries use_worker_affinity flag."""
        wi = WorkflowInput(
            graph_definition_ref="ref",
            use_worker_affinity=True,
        )
        assert wi.use_worker_affinity is True
        assert wi.sticky_task_queue is None  # discovered at runtime


@pytest.mark.integration
class TestAffinityQueueContinueAsNew:
    """Test that discovered queue survives continue-as-new."""

    def test_discovered_queue_in_restored_state(self) -> None:
        from langgraph.temporal.config import RestoredState

        rs = RestoredState(
            checkpoint={"v": 1},
            step=500,
            sticky_task_queue="discovered-worker-abc123",
        )
        assert rs.sticky_task_queue == "discovered-worker-abc123"

    def test_continue_as_new_preserves_discovered_queue(self) -> None:
        """Verify continue-as-new WorkflowInput includes the discovered
        sticky_task_queue and use_worker_affinity."""
        from langgraph.temporal.config import RestoredState

        sac = SubAgentConfig(task_queue="sub-q")
        wi = WorkflowInput(
            graph_definition_ref="ref",
            input_data=None,
            recursion_limit=100,
            sticky_task_queue="discovered-q",
            use_worker_affinity=True,
            subagent_config=sac,
            restored_state=RestoredState(
                checkpoint={"v": 1},
                step=500,
                sticky_task_queue="discovered-q",
            ),
        )
        assert wi.sticky_task_queue == "discovered-q"
        assert wi.use_worker_affinity is True
        assert wi.subagent_config is not None
        assert wi.subagent_config.task_queue == "sub-q"
        assert wi.restored_state is not None
        assert wi.restored_state.sticky_task_queue == "discovered-q"
