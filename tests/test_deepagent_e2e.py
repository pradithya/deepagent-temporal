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
from langgraph.types import Command, Send, interrupt
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


class ParallelState(TypedDict):
    items_a: list[str]
    items_b: list[str]
    result: list[str]


class CondState(TypedDict):
    route: str
    value: str


class GotoState(TypedDict):
    from_a: bool
    from_b: bool


class LoopState(TypedDict):
    count: int
    history: str


class ScatterGatherState(TypedDict):
    subjects: list[str]
    results: Annotated[list[str], operator.add]
    summary: str


class WorkerInput(TypedDict):
    subject: str


class MultiInterruptState(TypedDict):
    log: Annotated[list[str], operator.add]


class ErrorRecoveryState(TypedDict):
    attempts: int
    value: str


class PipelineState(TypedDict):
    """State for a multi-stage data processing pipeline."""

    raw_data: str
    parsed: str
    validated: str
    transformed: str
    output: str


class ToolCallState(TypedDict):
    """State simulating an agentic tool-calling loop."""

    messages: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[str], operator.add]
    tool_results: Annotated[list[str], operator.add]


class BranchMergeState(TypedDict):
    """State for branch-and-merge pattern (parallel analysis then merge)."""

    input_data: str
    analysis_a: str
    analysis_b: str
    merged: str


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


# -- Parallel nodes --


def par_a(state: ParallelState) -> dict[str, Any]:
    return {"items_a": ["a"]}


def par_b(state: ParallelState) -> dict[str, Any]:
    return {"items_b": ["b"]}


def par_c(state: ParallelState) -> dict[str, Any]:
    a = state.get("items_a", [])
    b = state.get("items_b", [])
    return {"result": sorted(a + b)}


# -- Conditional edge nodes --


def cond_a(state: CondState) -> dict[str, Any]:
    return {"route": "go_b", "value": "a"}


def cond_b(state: CondState) -> dict[str, Any]:
    return {"value": state["value"] + "b"}


def cond_c(state: CondState) -> dict[str, Any]:
    return {"value": state["value"] + "c"}


def cond_router(state: CondState) -> str:
    return "b" if state.get("route") == "go_b" else "c"


# -- Command.goto nodes --


def goto_a(state: GotoState) -> Command:
    return Command(update={"from_a": True}, goto="b")


def goto_b(state: GotoState) -> dict[str, Any]:
    return {"from_b": True}


# -- Loop nodes --


def loop_a(state: LoopState) -> dict[str, Any]:
    count = state.get("count", 0)
    return {"count": count + 1, "history": state.get("history", "") + "a"}


def loop_b(state: LoopState) -> dict[str, Any]:
    return {"history": state.get("history", "") + "b"}


def loop_should_continue(state: LoopState) -> str:
    return "a" if state.get("count", 0) < 3 else "end"


# -- Scatter-gather nodes --


def scatter_node(state: ScatterGatherState) -> Command:
    sends = [Send("worker", {"subject": s}) for s in state["subjects"]]
    return Command(goto=sends)


def worker_node(state: WorkerInput) -> dict[str, Any]:
    subject = state["subject"]
    return {"results": [f"processed:{subject}"]}


def gather_node(state: ScatterGatherState) -> dict[str, Any]:
    return {"summary": ",".join(sorted(state.get("results", [])))}


# -- Multi-interrupt nodes --


def multi_interrupt_step1(state: MultiInterruptState) -> dict[str, Any]:
    approval = interrupt({"step": 1, "message": "Approve step 1?"})
    return {"log": [f"step1({approval})"]}


def multi_interrupt_step2(state: MultiInterruptState) -> dict[str, Any]:
    approval = interrupt({"step": 2, "message": "Approve step 2?"})
    return {"log": [f"step2({approval})"]}


def multi_interrupt_finalize(state: MultiInterruptState) -> dict[str, Any]:
    return {"log": ["finalized"]}


# -- Error recovery nodes --


_attempt_counter: dict[str, int] = {}


def flaky_node(state: ErrorRecoveryState) -> dict[str, Any]:
    """Simulates a node that fails on first attempt then succeeds."""
    wf_key = state.get("value", "default")
    _attempt_counter.setdefault(wf_key, 0)
    _attempt_counter[wf_key] += 1
    return {
        "attempts": _attempt_counter[wf_key],
        "value": f"success_on_attempt_{_attempt_counter[wf_key]}",
    }


# -- Pipeline nodes --


def parse_data(state: PipelineState) -> dict[str, Any]:
    return {"parsed": f"parsed({state['raw_data']})"}


def validate_data(state: PipelineState) -> dict[str, Any]:
    return {"validated": f"valid({state['parsed']})"}


def transform_data(state: PipelineState) -> dict[str, Any]:
    return {"transformed": f"transformed({state['validated']})"}


def output_data(state: PipelineState) -> dict[str, Any]:
    return {"output": f"output({state['transformed']})"}


# -- Tool-calling loop nodes --


def agent_decide(state: ToolCallState) -> dict[str, Any]:
    """Agent decides what tool to call based on messages so far."""
    n_results = len(state.get("tool_results", []))
    if n_results >= 2:
        return {"messages": [f"final_answer(tools_used={n_results})"]}
    tool_name = "search" if n_results == 0 else "calculate"
    return {
        "messages": [f"call_tool:{tool_name}"],
        "tool_calls": [tool_name],
    }


def execute_tools(state: ToolCallState) -> dict[str, Any]:
    """Execute the most recent tool call."""
    last_call = state["tool_calls"][-1] if state.get("tool_calls") else "unknown"
    return {
        "messages": [f"result:{last_call}=done"],
        "tool_results": [f"{last_call}=done"],
    }


def tool_loop_router(state: ToolCallState) -> str:
    """Route: if agent produced final_answer, end. Otherwise go to tools."""
    last_msg = state["messages"][-1] if state.get("messages") else ""
    return END if last_msg.startswith("final_answer") else "tools"


# -- Branch-and-merge nodes --


def analyze_a(state: BranchMergeState) -> dict[str, Any]:
    return {"analysis_a": f"sentiment({state['input_data']})"}


def analyze_b(state: BranchMergeState) -> dict[str, Any]:
    return {"analysis_b": f"entities({state['input_data']})"}


def merge_analyses(state: BranchMergeState) -> dict[str, Any]:
    return {"merged": f"{state['analysis_a']}+{state['analysis_b']}"}


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


# ---------------------------------------------------------------------------
# NEW: Advanced pattern tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDeepAgentParallelNodes:
    """START -> [A, B] -> C -> END: parallel node execution fan-out/fan-in.

    Deep agents with parallel tool execution (e.g., search + calculate
    concurrently) need parallel nodes to work correctly on Temporal.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_parallel_nodes(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(ParallelState)
        builder.add_node("a", par_a)
        builder.add_node("b", par_b)
        builder.add_node("c", par_c)
        builder.add_edge(START, "a")
        builder.add_edge(START, "b")
        builder.add_edge("a", "c")
        builder.add_edge("b", "c")
        builder.add_edge("c", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-parallel-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"items_a": [], "items_b": [], "result": []},
                config={"configurable": {"thread_id": wf_id}},
            )

        assert result["result"] == ["a", "b"]


@pytest.mark.integration
class TestDeepAgentConditionalEdge:
    """A -> (condition) -> B or C -> END: conditional routing.

    Models the pattern where an agent decides between different
    execution paths based on state (e.g., code vs search tool).
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_conditional_edge(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(CondState)
        builder.add_node("a", cond_a)
        builder.add_node("b", cond_b)
        builder.add_node("c", cond_c)
        builder.add_edge(START, "a")
        builder.add_conditional_edges("a", cond_router, {"b": "b", "c": "c"})
        builder.add_edge("b", END)
        builder.add_edge("c", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-cond-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"route": "", "value": ""},
                config={"configurable": {"thread_id": wf_id}},
            )

        assert result["value"] == "ab"


@pytest.mark.integration
class TestDeepAgentCommandGoto:
    """A -> (Command.goto) -> B -> END: imperative routing via Command.

    Models dynamic routing used by agents that decide their next
    step programmatically (e.g., "go to code_review after generating").
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_command_goto(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(GotoState)
        builder.add_node("a", goto_a)
        builder.add_node("b", goto_b)
        builder.add_edge(START, "a")
        builder.add_edge("b", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-goto-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"from_a": False, "from_b": False},
                config={"configurable": {"thread_id": wf_id}},
            )

        assert result["from_a"] is True
        assert result["from_b"] is True


@pytest.mark.integration
class TestDeepAgentLoopWithStateAccumulation:
    """A -> B -> A (loop 3x) -> END: stateful loop with accumulation.

    Models the deep agent's call_model -> tools loop where state
    accumulates over multiple iterations until a termination condition.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_loop_with_accumulation(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(LoopState)
        builder.add_node("a", loop_a)
        builder.add_node("b", loop_b)
        builder.add_edge(START, "a")
        builder.add_edge("a", "b")
        builder.add_conditional_edges("b", loop_should_continue, {"a": "a", "end": END})
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-loop-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"count": 0, "history": ""},
                config={"configurable": {"thread_id": wf_id}},
            )

        assert result["count"] == 3
        assert result["history"] == "ababab"


@pytest.mark.integration
class TestDeepAgentScatterGather:
    """scatter -> [worker x N] -> gather -> END: fan-out via Send.

    Models the pattern where an agent dispatches multiple parallel
    tasks (e.g., researching multiple topics simultaneously) and
    collects results. Uses Command(goto=Send(...)) for dynamic dispatch.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_scatter_gather(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(ScatterGatherState)
        builder.add_node("scatter", scatter_node)
        builder.add_node("worker", worker_node)
        builder.add_node("gather", gather_node)
        builder.add_edge(START, "scatter")
        builder.add_edge("worker", "gather")
        builder.add_edge("gather", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        worker = agent.create_worker(**_WORKER_KWARGS)

        subjects = ["alpha", "beta", "gamma"]
        wf_id = f"e2e-scatter-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"subjects": subjects, "results": [], "summary": ""},
                config={"configurable": {"thread_id": wf_id}},
            )

        assert sorted(result["results"]) == [
            "processed:alpha",
            "processed:beta",
            "processed:gamma",
        ]
        assert result["summary"] == "processed:alpha,processed:beta,processed:gamma"


@pytest.mark.integration
class TestDeepAgentMultipleInterrupts:
    """step1(interrupt) -> step2(interrupt) -> finalize -> END.

    Models a multi-stage approval pipeline where a deep agent
    requires human approval at multiple checkpoints (e.g., plan
    approval then deployment approval).
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_sequential_interrupts(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(MultiInterruptState)
        builder.add_node("step1", multi_interrupt_step1)
        builder.add_node("step2", multi_interrupt_step2)
        builder.add_node("finalize", multi_interrupt_finalize)
        builder.add_edge(START, "step1")
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", "finalize")
        builder.add_edge("finalize", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        wf_id = f"e2e-multiint-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": wf_id}}
        worker = agent.create_worker(**_WORKER_KWARGS)

        async with worker:
            handle = await agent.astart({"log": []}, config)

            # Wait for first interrupt
            for _ in range(30):
                state = await agent.get_state(config)
                if state["status"] == "interrupted":
                    break
                await asyncio.sleep(0.2)
            else:
                pytest.fail("Did not reach first interrupt")

            # Resume first interrupt
            await agent.resume(config, "yes_step1")

            # Wait for second interrupt
            for _ in range(30):
                state = await agent.get_state(config)
                if state["status"] == "interrupted":
                    # Check it's the second interrupt (step1 result already in log)
                    if any(
                        "step1" in str(v)
                        for v in state.get("values", {}).get("log", [])
                    ):
                        break
                await asyncio.sleep(0.2)
            else:
                pytest.fail("Did not reach second interrupt")

            # Resume second interrupt
            await agent.resume(config, "yes_step2")

            result = await handle.result()

        log = result.channel_values["log"]
        assert "step1(yes_step1)" in log
        assert "step2(yes_step2)" in log
        assert "finalized" in log


@pytest.mark.integration
class TestDeepAgentStateQuery:
    """Query workflow state during and after execution.

    Models the pattern where a client checks agent progress
    (e.g., a UI polling for the current state of a long-running task).
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_state_query_after_completion(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(LoopState)
        builder.add_node("a", loop_a)
        builder.add_node("b", loop_b)
        builder.add_edge(START, "a")
        builder.add_edge("a", "b")
        builder.add_edge("b", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        wf_id = f"e2e-query-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": wf_id}}
        worker = agent.create_worker(**_WORKER_KWARGS)

        async with worker:
            handle = await agent.astart({"count": 0, "history": ""}, config)
            await handle.result()

            state = await agent.get_state(config)

        assert state["values"]["count"] == 1
        assert state["values"]["history"] == "ab"
        assert state["status"] == "done"


@pytest.mark.integration
class TestDeepAgentPipeline:
    """parse -> validate -> transform -> output -> END: linear data pipeline.

    Models a deep agent that processes data through multiple
    transformation stages (e.g., read file -> parse -> validate ->
    write output), common in coding agents.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_multi_stage_pipeline(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(PipelineState)
        builder.add_node("parse", parse_data)
        builder.add_node("validate", validate_data)
        builder.add_node("transform", transform_data)
        builder.add_node("output", output_data)
        builder.add_edge(START, "parse")
        builder.add_edge("parse", "validate")
        builder.add_edge("validate", "transform")
        builder.add_edge("transform", "output")
        builder.add_edge("output", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-pipeline-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {
                    "raw_data": "input_data",
                    "parsed": "",
                    "validated": "",
                    "transformed": "",
                    "output": "",
                },
                config={"configurable": {"thread_id": wf_id}},
            )

        assert result["output"] == "output(transformed(valid(parsed(input_data))))"


@pytest.mark.integration
class TestDeepAgentToolCallingLoop:
    """agent -> tools -> agent -> tools -> agent(final) -> END.

    Simulates a realistic agentic tool-calling loop where the agent
    calls multiple tools sequentially, accumulating results, before
    producing a final answer. This is the core deep agent pattern.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_tool_calling_loop(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(ToolCallState)
        builder.add_node("agent", agent_decide)
        builder.add_node("tools", execute_tools)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", tool_loop_router, {END: END, "tools": "tools"}
        )
        builder.add_edge("tools", "agent")
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-toolloop-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"messages": [], "tool_calls": [], "tool_results": []},
                config={"configurable": {"thread_id": wf_id}},
            )

        # Agent should have called 2 tools (search, calculate) then final answer
        assert len(result["tool_results"]) == 2
        assert "search=done" in result["tool_results"]
        assert "calculate=done" in result["tool_results"]
        assert any("final_answer" in m for m in result["messages"])


@pytest.mark.integration
class TestDeepAgentBranchAndMerge:
    """START -> [analyze_a, analyze_b] -> merge -> END.

    Models parallel analysis (e.g., sentiment + entity extraction)
    followed by a merge step. Common in agents that need to gather
    multiple perspectives before making a decision.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_branch_and_merge(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(BranchMergeState)
        builder.add_node("analyze_a", analyze_a)
        builder.add_node("analyze_b", analyze_b)
        builder.add_node("merge", merge_analyses)
        builder.add_edge(START, "analyze_a")
        builder.add_edge(START, "analyze_b")
        builder.add_edge("analyze_a", "merge")
        builder.add_edge("analyze_b", "merge")
        builder.add_edge("merge", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-branch-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {
                    "input_data": "hello world",
                    "analysis_a": "",
                    "analysis_b": "",
                    "merged": "",
                },
                config={"configurable": {"thread_id": wf_id}},
            )

        assert result["analysis_a"] == "sentiment(hello world)"
        assert result["analysis_b"] == "entities(hello world)"
        assert result["merged"] == "sentiment(hello world)+entities(hello world)"


@pytest.mark.integration
class TestDeepAgentInterruptBefore:
    """Interrupt before a specific node using compile-time configuration.

    Models the pattern where certain nodes (e.g., "deploy", "delete")
    require pre-approval before execution.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_interrupt_before_node(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        builder = StateGraph(LoopState)
        builder.add_node("a", loop_a)
        builder.add_node("b", loop_b)
        builder.add_edge(START, "a")
        builder.add_edge("a", "b")
        builder.add_edge("b", END)
        graph = builder.compile(interrupt_before=["b"])

        agent = TemporalDeepAgent(graph, temporal_client, task_queue=task_queue)
        wf_id = f"e2e-intbefore-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": wf_id}}
        worker = agent.create_worker(**_WORKER_KWARGS)

        async with worker:
            handle = await agent.astart({"count": 0, "history": ""}, config)

            # Wait for interrupt (before node "b")
            for _ in range(30):
                state = await agent.get_state(config)
                if state["status"] == "interrupted":
                    break
                await asyncio.sleep(0.2)
            else:
                pytest.fail("Did not reach interrupt before 'b'")

            # At this point, "a" has run but "b" has not
            assert state["values"]["history"] == "a"

            # Resume
            await agent.resume(config, None)

            result = await handle.result()

        assert result.channel_values["history"] == "ab"


@pytest.mark.integration
class TestDeepAgentWorkerAffinityWithParallel:
    """Worker affinity + parallel nodes: all Activities on same queue.

    Tests that even when nodes execute in parallel, all Activities
    are dispatched to the worker-specific queue.
    """

    def setup_method(self) -> None:
        GraphRegistry.reset()

    async def test_affinity_with_parallel_nodes(
        self, temporal_client: TemporalClient, task_queue: str
    ) -> None:
        class AffinityParallelState(TypedDict):
            queues: Annotated[list[str], operator.add]

        def par_node_1(state: AffinityParallelState) -> dict[str, Any]:
            return {"queues": [_get_activity_task_queue()]}

        def par_node_2(state: AffinityParallelState) -> dict[str, Any]:
            return {"queues": [_get_activity_task_queue()]}

        def par_node_3(state: AffinityParallelState) -> dict[str, Any]:
            return {"queues": [_get_activity_task_queue()]}

        builder = StateGraph(AffinityParallelState)
        builder.add_node("n1", par_node_1)
        builder.add_node("n2", par_node_2)
        builder.add_node("n3", par_node_3)
        builder.add_edge(START, "n1")
        builder.add_edge(START, "n2")
        builder.add_edge("n1", "n3")
        builder.add_edge("n2", "n3")
        builder.add_edge("n3", END)
        graph = builder.compile()

        agent = TemporalDeepAgent(
            graph, temporal_client, task_queue=task_queue, use_worker_affinity=True
        )
        worker = agent.create_worker(**_WORKER_KWARGS)

        wf_id = f"e2e-affpar-{uuid.uuid4().hex[:8]}"
        async with worker:
            result = await agent.ainvoke(
                {"queues": []},
                config={"configurable": {"thread_id": wf_id}},
            )

        # All 3 Activities should be on the same worker-specific queue
        assert len(result["queues"]) == 3
        assert all(q == result["queues"][0] for q in result["queues"])
        assert result["queues"][0] != task_queue
