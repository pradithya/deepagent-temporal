"""Tests for TemporalSubAgentMiddleware and SubAgentRequest."""

from __future__ import annotations

from langgraph.temporal.activities import _child_workflow_requests_var

from deepagent_temporal.middleware import (
    SubAgentRequest,
    TemporalSubAgentMiddleware,
    collect_pending_requests,
)


class TestSubAgentRequest:
    def test_to_dict(self) -> None:
        req = SubAgentRequest(
            subagent_type="researcher",
            instruction="Research AI trends",
            tool_call_id="tc-1",
            initial_state={"messages": [{"role": "user", "content": "Research AI"}]},
            graph_definition_ref="subagent:researcher",
        )
        d = req.to_dict()
        assert d["subagent_type"] == "researcher"
        assert d["instruction"] == "Research AI trends"
        assert d["tool_call_id"] == "tc-1"
        assert d["graph_definition_ref"] == "subagent:researcher"

    def test_from_dict(self) -> None:
        d = {
            "subagent_type": "coder",
            "instruction": "Write tests",
            "tool_call_id": "tc-2",
            "initial_state": {"messages": []},
            "graph_definition_ref": "subagent:coder",
        }
        req = SubAgentRequest.from_dict(d)
        assert req.subagent_type == "coder"
        assert req.instruction == "Write tests"
        assert req.tool_call_id == "tc-2"
        assert req.graph_definition_ref == "subagent:coder"

    def test_round_trip(self) -> None:
        original = SubAgentRequest(
            subagent_type="analyst",
            instruction="Analyze data",
            tool_call_id="tc-3",
            initial_state={"key": "value", "messages": []},
            graph_definition_ref="subagent:analyst",
        )
        reconstructed = SubAgentRequest.from_dict(original.to_dict())
        assert reconstructed.subagent_type == original.subagent_type
        assert reconstructed.instruction == original.instruction
        assert reconstructed.tool_call_id == original.tool_call_id
        assert reconstructed.initial_state == original.initial_state
        assert reconstructed.graph_definition_ref == original.graph_definition_ref


class TestContextVarIsolation:
    def test_isolation_per_set(self) -> None:
        """Each .set() call creates an independent list."""
        # First context: set and append
        _child_workflow_requests_var.set([])
        _child_workflow_requests_var.get().append({"type": "a"})
        result_a = _child_workflow_requests_var.get()

        # Second context: fresh set
        _child_workflow_requests_var.set([])
        result_b = _child_workflow_requests_var.get()

        assert len(result_a) == 1
        assert result_a[0]["type"] == "a"
        assert len(result_b) == 0


class TestTemporalSubAgentMiddleware:
    def setup_method(self) -> None:
        # Reset the context var for each test
        _child_workflow_requests_var.set([])

    def test_build_task_tool_returns_callable(self) -> None:
        middleware = TemporalSubAgentMiddleware()
        tool = middleware.build_task_tool()
        assert callable(tool)

    def test_tool_appends_to_context_var(self) -> None:
        middleware = TemporalSubAgentMiddleware(
            subagent_specs={"researcher": "subagent:researcher"},
        )
        tool = middleware.build_task_tool()

        result = tool(
            instruction="Research AI trends",
            subagent_type="researcher",
            tool_call_id="tc-1",
        )

        assert "dispatched" in result
        assert "researcher" in result

        pending = _child_workflow_requests_var.get([])
        assert len(pending) == 1
        assert pending[0]["subagent_type"] == "researcher"
        assert pending[0]["instruction"] == "Research AI trends"
        assert pending[0]["tool_call_id"] == "tc-1"
        assert pending[0]["graph_definition_ref"] == "subagent:researcher"

    def test_multiple_tool_calls(self) -> None:
        middleware = TemporalSubAgentMiddleware(
            subagent_specs={
                "researcher": "subagent:researcher",
                "coder": "subagent:coder",
            },
        )
        tool = middleware.build_task_tool()

        tool(instruction="Research", subagent_type="researcher", tool_call_id="tc-1")
        tool(instruction="Code", subagent_type="coder", tool_call_id="tc-2")

        pending = _child_workflow_requests_var.get([])
        assert len(pending) == 2
        assert pending[0]["subagent_type"] == "researcher"
        assert pending[1]["subagent_type"] == "coder"

    def test_default_graph_ref(self) -> None:
        middleware = TemporalSubAgentMiddleware(
            default_graph_ref="default-agent",
        )
        tool = middleware.build_task_tool()

        tool(
            instruction="Do something",
            subagent_type="unknown-type",
            tool_call_id="tc-1",
        )

        pending = _child_workflow_requests_var.get([])
        assert len(pending) == 1
        assert pending[0]["graph_definition_ref"] == "default-agent"

    def test_tool_sets_initial_state(self) -> None:
        middleware = TemporalSubAgentMiddleware()
        tool = middleware.build_task_tool()

        tool(instruction="Test instruction", tool_call_id="tc-1")

        pending = _child_workflow_requests_var.get([])
        assert len(pending) == 1
        state = pending[0]["initial_state"]
        assert "messages" in state
        assert state["messages"][0]["content"] == "Test instruction"


class TestCollectPendingRequests:
    def setup_method(self) -> None:
        _child_workflow_requests_var.set([])

    def test_collect_returns_and_resets(self) -> None:
        _child_workflow_requests_var.set([{"type": "test"}])

        collected = collect_pending_requests()
        assert len(collected) == 1
        assert collected[0]["type"] == "test"

        # After collection, the var should be empty
        assert _child_workflow_requests_var.get([]) == []

    def test_collect_empty(self) -> None:
        collected = collect_pending_requests()
        assert collected == []
