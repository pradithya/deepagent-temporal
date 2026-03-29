"""Deep Agent Temporal integration for durable execution of AI agents.

This package provides Temporal integration for Deep Agents (from
`langchain-ai/deepagents`), enabling durable execution, sub-agent
dispatch via Child Workflows, and worker affinity via sticky task queues.
"""

# Re-export from langgraph-temporal for convenience
from langgraph.temporal.config import RetryPolicyConfig

from deepagent_temporal.agent import TemporalDeepAgent, create_temporal_deep_agent
from deepagent_temporal.config import SubAgentSpec
from deepagent_temporal.middleware import SubAgentRequest, TemporalSubAgentMiddleware
from deepagent_temporal.serialization import validate_payload_size

__all__ = [
    "RetryPolicyConfig",
    "SubAgentRequest",
    "SubAgentSpec",
    "TemporalDeepAgent",
    "TemporalSubAgentMiddleware",
    "create_temporal_deep_agent",
    "validate_payload_size",
]
