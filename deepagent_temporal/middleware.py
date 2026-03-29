"""Temporal-aware sub-agent middleware for Deep Agents.

Provides `TemporalSubAgentMiddleware` which intercepts sub-agent invocations
and stores them as `SubAgentRequest` objects in a context variable. The
Activity collects these after node execution and returns them in
`NodeActivityOutput.child_workflow_requests`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.temporal.activities import _child_workflow_requests_var


@dataclass
class SubAgentRequest:
    """Request for a sub-agent Child Workflow dispatch.

    Stored in context variable during tool execution; collected by the
    Activity and returned in `NodeActivityOutput.child_workflow_requests`.

    All fields must be JSON-serializable for Temporal payload conversion.

    Attributes:
        subagent_type: Type/name of the sub-agent to dispatch.
        instruction: The instruction/prompt for the sub-agent.
        tool_call_id: The tool call ID to map the result back.
        initial_state: Initial state for the sub-agent (excluded keys removed).
        graph_definition_ref: Reference to the pre-registered sub-agent graph.
    """

    subagent_type: str
    instruction: str
    tool_call_id: str
    initial_state: dict[str, Any]
    graph_definition_ref: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict for Temporal payload."""
        return {
            "subagent_type": self.subagent_type,
            "instruction": self.instruction,
            "tool_call_id": self.tool_call_id,
            "initial_state": self.initial_state,
            "graph_definition_ref": self.graph_definition_ref,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubAgentRequest:
        """Deserialize from dict."""
        return cls(
            subagent_type=d["subagent_type"],
            instruction=d["instruction"],
            tool_call_id=d["tool_call_id"],
            initial_state=d["initial_state"],
            graph_definition_ref=d["graph_definition_ref"],
        )


class TemporalSubAgentMiddleware:
    """Temporal-aware replacement for SubAgentMiddleware.

    Instead of invoking sub-agents in-process, the `task` tool stores a
    `SubAgentRequest` in the `_child_workflow_requests_var` context variable
    and returns a placeholder string. The Activity collects pending requests
    after execution and includes them in the output.

    This class is standalone (does not subclass deepagents middleware) to
    avoid a hard dependency on the deepagents package during unit testing.

    Args:
        subagent_specs: Mapping of sub-agent type names to their graph
            definition references in GraphRegistry.
        default_graph_ref: Default graph reference for sub-agents not in
            `subagent_specs`.
    """

    def __init__(
        self,
        subagent_specs: dict[str, str] | None = None,
        default_graph_ref: str | None = None,
    ) -> None:
        self._subagent_specs = subagent_specs or {}
        self._default_graph_ref = default_graph_ref

    def build_task_tool(self) -> Any:
        """Build a `task` tool function that stores SubAgentRequests.

        Returns a callable that, when invoked with an instruction and
        sub-agent type, appends a `SubAgentRequest` to the context
        variable and returns a placeholder string.
        """
        specs = self._subagent_specs
        default_ref = self._default_graph_ref

        def task_tool(
            instruction: str,
            subagent_type: str = "general-purpose",
            tool_call_id: str = "",
        ) -> str:
            """Dispatch a sub-agent as a Temporal Child Workflow.

            Args:
                instruction: The instruction/prompt for the sub-agent.
                subagent_type: Type of sub-agent to dispatch.
                tool_call_id: The tool call ID for result mapping.

            Returns:
                A placeholder string indicating the sub-agent was dispatched.
            """
            graph_ref = specs.get(subagent_type, default_ref or "")

            req = SubAgentRequest(
                subagent_type=subagent_type,
                instruction=instruction,
                tool_call_id=tool_call_id,
                initial_state={"messages": [{"role": "user", "content": instruction}]},
                graph_definition_ref=graph_ref,
            )

            # Append to context variable (initialized by _execute_node_impl)
            pending = _child_workflow_requests_var.get([])
            pending.append(req.to_dict())
            _child_workflow_requests_var.set(pending)

            return f"[Sub-agent '{subagent_type}' dispatched as Child Workflow]"

        return task_tool


def collect_pending_requests() -> list[dict[str, Any]]:
    """Collect pending SubAgentRequests from the context variable.

    Returns the list of serialized requests and resets the context variable.
    """
    pending = _child_workflow_requests_var.get([])
    _child_workflow_requests_var.set([])
    return pending
