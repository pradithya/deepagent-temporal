"""Deep Agent Temporal integration for durable execution of AI agents.

This package provides Temporal integration for Deep Agents (from
`langchain-ai/deepagents`), enabling durable execution, sub-agent
dispatch via Child Workflows, and worker affinity via sticky task queues.
"""

# Re-export from langgraph-temporal for convenience
from langgraph.temporal.config import RetryPolicyConfig

from deepagent_temporal.activity import StreamingNodeWrapper, wrap_graph_for_streaming
from deepagent_temporal.agent import TemporalDeepAgent, create_temporal_deep_agent
from deepagent_temporal.config import SubAgentSpec
from deepagent_temporal.middleware import SubAgentRequest, TemporalSubAgentMiddleware
from deepagent_temporal.serialization import validate_payload_size
from deepagent_temporal.streaming import (
    RedisStreamBackend,
    TokenCapturingHandler,
    TokenEvent,
)
from deepagent_temporal.worker import create_streaming_worker

__all__ = [
    "RedisStreamBackend",
    "RetryPolicyConfig",
    "StreamingNodeWrapper",
    "SubAgentRequest",
    "SubAgentSpec",
    "TemporalDeepAgent",
    "TemporalSubAgentMiddleware",
    "TokenCapturingHandler",
    "TokenEvent",
    "create_streaming_worker",
    "create_temporal_deep_agent",
    "validate_payload_size",
    "wrap_graph_for_streaming",
]
