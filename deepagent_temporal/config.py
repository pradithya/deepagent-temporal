"""Configuration for the Deep Agent Temporal integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class SubAgentSpec:
    """Specification for a sub-agent type.

    Attributes:
        name: Identifier for this sub-agent type (e.g., "researcher", "coder").
        graph_definition_ref: Reference to the pre-registered sub-agent graph
            in GraphRegistry.
        task_queue: Optional task queue override for this sub-agent type.
        execution_timeout: Maximum execution time for this sub-agent.
        description: Human-readable description of the sub-agent's purpose.
    """

    name: str
    graph_definition_ref: str | None = None
    task_queue: str | None = None
    execution_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    description: str = ""
