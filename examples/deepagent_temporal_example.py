"""Example: Deep Agent with Temporal durable execution.

Shows before/after comparison of migrating from vanilla Deep Agent
to Temporal-backed execution. Uses mock/stub LLM for reproducibility.

Prerequisites:
    pip install langgraph-temporal deepagents

Usage:
    # Start a Temporal server (e.g., via Docker or Temporal CLI)
    # Then run:
    python examples/deepagent_temporal_example.py
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

# ============================================================
# BEFORE: Vanilla Deep Agent (no durability)
# ============================================================
#
# from deepagents import create_deep_agent
#
# agent = create_deep_agent(
#     model=ChatAnthropic(model="claude-sonnet-4-20250514"),
#     tools=[read_file, write_file, execute],
#     system_prompt="You are a helpful coding assistant.",
#     backend=FilesystemBackend(root_dir="/workspace"),
# )
#
# # No durability -- if process crashes, all progress is lost
# result = await agent.ainvoke(
#     {"messages": [HumanMessage(content="Fix the bug in main.py")]},
#     config={"configurable": {"thread_id": "task-123"}},
# )

# ============================================================
# AFTER: Temporal-backed Deep Agent (durable execution)
# ============================================================


async def main() -> None:
    """Demonstrate TemporalDeepAgent usage with mock components."""
    from deepagent_temporal.agent import TemporalDeepAgent

    # In real usage, this would be a compiled graph from create_deep_agent()
    mock_graph = MagicMock()
    mock_graph.name = "coding-agent"
    mock_graph.interrupt_before_nodes = []
    mock_graph.interrupt_after_nodes = []

    # In real usage, this would be: await Client.connect("localhost:7233")
    mock_client = AsyncMock()
    mock_client.execute_workflow = AsyncMock(
        return_value=MagicMock(
            channel_values={"messages": ["Bug fixed in main.py!"]},
            step=5,
        )
    )

    # Wrap the agent for Temporal execution
    temporal_agent = TemporalDeepAgent(
        mock_graph,
        mock_client,
        # Task queue for this agent's Activities
        task_queue="coding-agents",
        # Enable worker affinity — the Workflow discovers a worker-specific
        # queue at startup and pins all Activities to that worker
        # (important for FilesystemBackend affinity)
        use_worker_affinity=True,
        # Sub-agents get their own task queue
        subagent_task_queue="coding-agents-sub",
        subagent_execution_timeout=timedelta(minutes=15),
        # Workflow-level timeouts
        workflow_execution_timeout=timedelta(hours=2),
    )

    # Execute with durability -- survives process crashes!
    result = await temporal_agent.ainvoke(
        {"messages": ["Fix the bug in main.py"]},
        config={"configurable": {"thread_id": "task-123"}},
    )
    print(f"Result: {result}")

    # ---- Worker setup (separate process) ----
    #
    # In a real deployment, run this on a worker machine:
    #
    # worker = temporal_agent.create_worker()
    # async with worker:
    #     await worker.run()

    # ---- Sub-agent middleware setup ----
    #
    # For agents with sub-agent support, use TemporalSubAgentMiddleware:
    #
    # from deepagent_temporal.middleware import TemporalSubAgentMiddleware
    #
    # middleware = TemporalSubAgentMiddleware(
    #     subagent_specs={
    #         "researcher": "subagent:researcher",
    #         "coder": "subagent:coder",
    #     },
    # )
    # task_tool = middleware.build_task_tool()
    #
    # When the task_tool is called during graph execution, it appends
    # a SubAgentRequest to the context variable. The Activity collects
    # these and the Workflow dispatches them as Child Workflows.

    # ---- Local development ----
    #
    # For testing without a Temporal server:
    #
    # temporal_agent = await TemporalDeepAgent.local(
    #     mock_graph,
    #     task_queue="test-queue",
    # )


if __name__ == "__main__":
    asyncio.run(main())
