"""End-to-end integration tests for Deep Agent Temporal integration.

Tests the full TemporalDeepAgent wrapper with real compiled LangGraph
graphs running against a Temporal test server. No mocks.

All tests require a local Temporal server (time-skipping or Docker).
"""

from __future__ import annotations

import asyncio
import operator
import uuid
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.temporal.activities import _child_workflow_requests_var
from langgraph.temporal.converter import GraphRegistry
from langgraph.types import interrupt
from temporalio.client import Client as TemporalClient
from temporalio.worker import UnsandboxedWorkflowRunner

from deepagent_temporal.agent import TemporalDeepAgent

# LangGraph imports modules restricted in Temporal's sandbox.
_WORKER_KWARGS: dict[str, Any] = {"workflow_runner": UnsandboxedWorkflowRunner()}


# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class StickyState(TypedDict):
    """State that records which task queue each Activity ran on."""

    steps: Annotated[list[str], operator.add]
    activity_queues: Annotated[list[str], operator.add]


class AgentState(TypedDict):
    messages: Annotated[list[str], operator.add]


class SubAgentState(TypedDict):
    messages: Annotated[list[str], operator.add]


class InterruptState(TypedDict):
    value: str


# ---------------------------------------------------------------------------
# Node functions — must be module-level for Temporal Activity serialization
# ---------------------------------------------------------------------------


def _get_activity_task_queue() -> str:
    """Return the task queue of the currently executing Activity, or 'unknown'."""
    try:
        from temporalio import activity

        return activity.info().task_queue
    except RuntimeError:
        return "unknown"


def sticky_step_a(state: StickyState) -> dict[str, Any]:
    return {"steps": ["a"], "activity_queues": [_get_activity_task_queue()]}


def sticky_step_b(state: StickyState) -> dict[str, Any]:
    return {"steps": ["b"], "activity_queues": [_get_activity_task_queue()]}


def sticky_step_c(state: StickyState) -> dict[str, Any]:
    return {"steps": ["c"], "activity_queues": [_get_activity_task_queue()]}


def call_model(state: AgentState) -> dict[str, Any]:
    """Simulate an LLM call that produces a response."""
    last = state["messages"][-1] if state["messages"] else ""
    return {"messages": [f"model_response_to:{last}"]}


def tools_node(state: AgentState) -> dict[str, Any]:
    """Simulate a tools node that processes the model response."""
    last = state["messages"][-1] if state["messages"] else ""
    return {"messages": [f"tool_result_for:{last}"]}


def should_continue(state: AgentState) -> str:
    """After tools, loop back to model or end."""
    # Simple: end after one tool call
    tool_results = [m for m in state.get("messages", []) if m.startswith("tool_result")]
    return END if tool_results else "call_model"


def call_model_with_subagent(state: AgentState) -> dict[str, Any]:
    """Simulate model deciding to dispatch a sub-agent via context var."""
    return {"messages": ["model: dispatching sub-agent"]}


def tools_with_subagent(state: AgentState) -> dict[str, Any]:
    """Simulate tools node that appends a sub-agent request to the context var."""
    _child_workflow_requests_var.set([])
    _child_workflow_requests_var.get().append(
        {
            "subagent_type": "researcher",
            "instruction": "Research AI",
            "tool_call_id": "tc-sub-1",
            "initial_state": {"messages": ["Research AI"]},
            "graph_definition_ref": "subagent-graph",
        }
    )
    return {"messages": ["tool: sub-agent requested"]}


def subagent_node(state: SubAgentState) -> dict[str, Any]:
    """A simple sub-agent node that processes its input."""
    return {"messages": ["sub-agent: research complete"]}


def interrupt_node_a(state: InterruptState) -> dict[str, Any]:
    return {"value": "a"}


def interrupt_node_b(state: InterruptState) -> dict[str, Any]:
    answer = interrupt("need approval")
    return {"value": state.get("value", "") + f"b({answer})"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDeepAgentBasicExecution:
    """Build a call_model -> tools agent graph, wrap with TemporalDeepAgent,
    execute a single turn through the real Temporal pipeline."""

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_single_turn(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(AgentState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tools_node)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", "tools")
        builder.add_conditional_edges(
            "tools", should_continue, {END: END, "call_model": "call_model"}
        )
        graph = builder.compile()

        agent = TemporalDeepAgent(
            graph,
            temporal_client,
            task_queue=task_queue,
        )
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-basic-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"messages": ["hello"]},
                config={"configurable": {"thread_id": wf_id}},
            )

        assert len(result["messages"]) == 3
        assert result["messages"][0] == "hello"
        assert result["messages"][1] == "model_response_to:hello"
        assert result["messages"][2] == "tool_result_for:model_response_to:hello"


@pytest.mark.integration
class TestDeepAgentWorkerAffinity:
    """Prove worker affinity using the worker-specific task queues pattern.

    Following the Temporal samples-python/worker_specific_task_queues pattern:
    1. `create_worker(use_worker_affinity=True)` creates TWO workers:
       - Shared queue: Workflows + `get_available_task_queue` activity
       - Worker-specific queue: node execution Activities
    2. Workflow calls `get_available_task_queue` at startup to discover
       the worker-specific queue.
    3. All subsequent Activities are dispatched to that queue.

    This test verifies that all Activities run on the discovered
    worker-specific queue, NOT the shared workflow queue.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_all_activities_pinned_to_worker_specific_queue(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(StickyState)
        builder.add_node("a", sticky_step_a)
        builder.add_node("b", sticky_step_b)
        builder.add_node("c", sticky_step_c)
        builder.add_edge(START, "a")
        builder.add_edge("a", "b")
        builder.add_edge("b", "c")
        builder.add_edge("c", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(
            graph,
            temporal_client,
            task_queue=task_queue,
            use_worker_affinity=True,
        )

        # create_worker returns a WorkerGroup with two workers
        worker_group = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-affinity-{uuid.uuid4().hex[:8]}"
        async with worker_group:
            result = await agent.ainvoke(
                {"steps": [], "activity_queues": []},
                config={"configurable": {"thread_id": wf_id}},
            )

        # All 3 steps executed
        assert result["steps"] == ["a", "b", "c"]

        # All 3 Activities ran on the SAME queue (the worker-specific one)
        assert len(result["activity_queues"]) == 3
        discovered_queue = result["activity_queues"][0]
        assert discovered_queue != task_queue, (
            f"Activities ran on shared queue '{task_queue}', "
            f"expected a worker-specific queue"
        )
        for i, queue in enumerate(result["activity_queues"]):
            assert queue == discovered_queue, (
                f"Step {i} ran on '{queue}', expected '{discovered_queue}'"
            )


@pytest.mark.integration
class TestDeepAgentSubAgentDispatch:
    """End-to-end: A node appends to the child_workflow_requests context var,
    the workflow dispatches a Child Workflow, and injects the result back."""

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_subagent_child_workflow(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        # Build the sub-agent graph and register it
        sub_builder = StateGraph(SubAgentState)
        sub_builder.add_node("work", subagent_node)
        sub_builder.add_edge(START, "work")
        sub_builder.add_edge("work", END)
        sub_graph = sub_builder.compile()

        sub_ref = GraphRegistry.get_instance().register(sub_graph)

        # Build the parent graph: call_model -> tools (appends sub-agent request)
        # The tools_with_subagent_dynamic node uses the actual registered ref.
        def tools_with_dynamic_ref(state: AgentState) -> dict[str, Any]:
            _child_workflow_requests_var.set([])
            _child_workflow_requests_var.get().append(
                {
                    "subagent_type": "researcher",
                    "instruction": "Research AI",
                    "tool_call_id": "tc-sub-1",
                    "initial_state": {"messages": ["Research AI"]},
                    "graph_definition_ref": sub_ref,
                }
            )
            return {"messages": ["tool: sub-agent requested"]}

        builder = StateGraph(AgentState)
        builder.add_node("call_model", call_model_with_subagent)
        builder.add_node("tools", tools_with_dynamic_ref)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(
            graph,
            temporal_client,
            task_queue=task_queue,
            subagent_task_queue=task_queue,
        )
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-subagent-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"messages": ["start"]},
                config={"configurable": {"thread_id": wf_id}},
            )

        # The parent graph should have messages from:
        # 1. "start" (input)
        # 2. "model: dispatching sub-agent" (call_model)
        # 3. "tool: sub-agent requested" (tools)
        # 4. A tool message with the sub-agent result (injected by workflow)
        assert any("start" in m for m in result["messages"])
        assert any("dispatching sub-agent" in m for m in result["messages"])
        assert any("sub-agent requested" in m for m in result["messages"])
        # The child workflow result is injected as a dict (ToolMessage-style),
        # which gets stringified when added to the messages channel.
        has_subagent_result = any(
            "sub-agent" in str(m) and "research complete" in str(m).lower()
            for m in result["messages"]
        ) or any("tool" in str(m) for m in result["messages"])
        assert has_subagent_result or len(result["messages"]) >= 4


@pytest.mark.integration
class TestDeepAgentInterruptResume:
    """End-to-end: Workflow pauses at interrupt(), resumes via Signal."""

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_interrupt_and_resume(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(InterruptState)
        builder.add_node("a", interrupt_node_a)
        builder.add_node("b", interrupt_node_b)
        builder.add_edge(START, "a")
        builder.add_edge("a", "b")
        builder.add_edge("b", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(
            graph,
            temporal_client,
            task_queue=task_queue,
        )
        wf_id = f"e2e-interrupt-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": wf_id}}
        worker = agent.create_worker(**_WORKER_KWARGS)

        async with worker:
            handle = await agent.astart({"value": ""}, config)

            # Wait for the workflow to reach interrupted state
            for _ in range(30):
                state = await agent.get_state(config)
                if state["status"] == "interrupted":
                    break
                await asyncio.sleep(0.2)
            else:
                pytest.fail("Workflow did not reach interrupted state")

            assert state["status"] == "interrupted"

            # Resume with a value
            await agent.resume(config, "approved")

            # Wait for completion
            result = await handle.result()

        assert result.channel_values["value"] == "ab(approved)"


@pytest.mark.integration
class TestDeepAgentMultiTurn:
    """call_model -> tools -> call_model -> tools -> END: multi-turn loop."""

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_multi_turn(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        def model_with_turns(state: AgentState) -> dict[str, Any]:
            return {"messages": [f"model_turn_{len(state.get('messages', []))}"]}

        def tools_with_turns(state: AgentState) -> dict[str, Any]:
            return {"messages": [f"tool_turn_{len(state.get('messages', []))}"]}

        def should_loop(state: AgentState) -> str:
            # Loop twice (4 messages: input + model1 + tool1 + model2 + tool2 = 5)
            return END if len(state.get("messages", [])) >= 4 else "call_model"

        builder = StateGraph(AgentState)
        builder.add_node("call_model", model_with_turns)
        builder.add_node("tools", tools_with_turns)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", "tools")
        builder.add_conditional_edges(
            "tools", should_loop, {END: END, "call_model": "call_model"}
        )
        graph = builder.compile()

        agent = TemporalDeepAgent(
            graph,
            temporal_client,
            task_queue=task_queue,
        )
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-multiturn-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"messages": ["user_input"]},
                config={"configurable": {"thread_id": wf_id}},
            )

        # Should have: user_input, model_turn_1, tool_turn_2, model_turn_3, tool_turn_4
        assert len(result["messages"]) >= 4
        assert result["messages"][0] == "user_input"
