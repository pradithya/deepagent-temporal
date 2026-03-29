"""Tests for token-level streaming infrastructure.

Tests the callback-based token capture (Phase 1) and the
StreamingNodeWrapper that injects callbacks into node execution.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from deepagent_temporal.streaming import (
    TokenCapturingHandler,
    TokenEvent,
    create_token_handler,
)


class TestTokenEvent:
    def test_to_dict(self) -> None:
        event = TokenEvent(
            token="hello",
            node_name="call_model",
            index=0,
            is_final=False,
            attempt=1,
        )
        d = event.to_dict()
        assert d["type"] == "token"
        assert d["token"] == "hello"
        assert d["node_name"] == "call_model"
        assert d["index"] == 0
        assert d["is_final"] == "0"
        assert d["attempt"] == 1

    def test_final_event(self) -> None:
        event = TokenEvent(
            token="",
            node_name="call_model",
            index=5,
            is_final=True,
        )
        d = event.to_dict()
        assert d["is_final"] == "1"
        assert d["token"] == ""


class TestTokenCapturingHandler:
    def test_captures_tokens(self) -> None:
        captured: list[TokenEvent] = []
        handler = TokenCapturingHandler("call_model", captured.append)

        run_id = uuid4()
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="Hello"),
        )

        handler.on_llm_new_token("Hello", chunk=chunk, run_id=run_id)

        assert len(captured) == 1
        assert captured[0].token == "Hello"
        assert captured[0].node_name == "call_model"
        assert captured[0].index == 0
        assert captured[0].is_final is False

    def test_increments_index(self) -> None:
        captured: list[TokenEvent] = []
        handler = TokenCapturingHandler("call_model", captured.append)

        run_id = uuid4()
        for _i, token in enumerate(["Hello", " ", "world"]):
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=token),
            )
            handler.on_llm_new_token(token, chunk=chunk, run_id=run_id)

        assert len(captured) == 3
        assert [e.index for e in captured] == [0, 1, 2]
        assert [e.token for e in captured] == ["Hello", " ", "world"]

    def test_ignores_non_chat_chunks(self) -> None:
        captured: list[TokenEvent] = []
        handler = TokenCapturingHandler("call_model", captured.append)

        handler.on_llm_new_token("test", chunk=None, run_id=uuid4())

        assert len(captured) == 0

    def test_emits_final_event_on_llm_end(self) -> None:
        captured: list[TokenEvent] = []
        handler = TokenCapturingHandler("call_model", captured.append)

        handler.on_llm_end(MagicMock(), run_id=uuid4())

        assert len(captured) == 1
        assert captured[0].is_final is True
        assert captured[0].token == ""

    def test_token_count_property(self) -> None:
        handler = TokenCapturingHandler("call_model", lambda _: None)
        assert handler.token_count == 0

        run_id = uuid4()
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="tok"),
        )
        handler.on_llm_new_token("tok", chunk=chunk, run_id=run_id)
        assert handler.token_count == 1

    def test_heartbeat_called_at_interval(self) -> None:
        heartbeat_mock = MagicMock()
        handler = TokenCapturingHandler(
            "call_model",
            lambda _: None,
            heartbeat_fn=heartbeat_mock,
            heartbeat_interval=3,
        )

        run_id = uuid4()
        for i in range(7):
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=f"t{i}"),
            )
            handler.on_llm_new_token(f"t{i}", chunk=chunk, run_id=run_id)

        # Heartbeat at token 3 and 6 (0-indexed: after 3rd and 6th token)
        assert heartbeat_mock.call_count == 2

    def test_heartbeat_error_suppressed(self) -> None:
        def failing_heartbeat(*args: Any) -> None:
            raise RuntimeError("Not in Activity context")

        captured: list[TokenEvent] = []
        handler = TokenCapturingHandler(
            "call_model",
            captured.append,
            heartbeat_fn=failing_heartbeat,
            heartbeat_interval=1,
        )

        run_id = uuid4()
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="test"),
        )
        # Should not raise
        handler.on_llm_new_token("test", chunk=chunk, run_id=run_id)
        assert len(captured) == 1

    def test_tracks_llm_call_count(self) -> None:
        handler = TokenCapturingHandler("call_model", lambda _: None)

        handler.on_chat_model_start({}, [[]], run_id=uuid4())
        assert handler._llm_call_count == 1

        handler.on_chat_model_start({}, [[]], run_id=uuid4())
        assert handler._llm_call_count == 2

    def test_resets_index_on_new_llm_call(self) -> None:
        captured: list[TokenEvent] = []
        handler = TokenCapturingHandler("call_model", captured.append)

        run_id1 = uuid4()
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="a"),
        )
        handler.on_llm_new_token("a", chunk=chunk, run_id=run_id1)
        handler.on_llm_new_token("b", chunk=chunk, run_id=run_id1)

        # New LLM call (e.g., SummarizationMiddleware triggers second call)
        handler.on_chat_model_start({}, [[]], run_id=uuid4())

        run_id2 = uuid4()
        handler.on_llm_new_token("c", chunk=chunk, run_id=run_id2)

        assert captured[-1].index == 0  # Reset to 0 for new call

    def test_attempt_number_passed_through(self) -> None:
        captured: list[TokenEvent] = []
        handler = TokenCapturingHandler("call_model", captured.append, attempt=3)

        run_id = uuid4()
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="test"),
        )
        handler.on_llm_new_token("test", chunk=chunk, run_id=run_id)

        assert captured[0].attempt == 3


class TestCreateTokenHandler:
    def test_factory_creates_handler(self) -> None:
        handler = create_token_handler(
            "call_model",
            lambda _: None,
            heartbeat_interval=100,
            attempt=2,
        )
        assert isinstance(handler, TokenCapturingHandler)
        assert handler._node_name == "call_model"
        assert handler._heartbeat_interval == 100
        assert handler._attempt == 2


class TestStreamingNodeWrapper:
    def test_injects_callback_and_delegates(self) -> None:
        from deepagent_temporal.activity import StreamingNodeWrapper

        # Mock the original runnable
        original = MagicMock()
        captured_config: dict[str, Any] = {}

        async def mock_ainvoke(
            input: Any, config: Any, **kwargs: Any
        ) -> dict[str, str]:
            captured_config.update(config)
            return {"result": "ok"}

        original.ainvoke = mock_ainvoke

        wrapper = StreamingNodeWrapper(original, "call_model")

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            wrapper.ainvoke({"x": 1}, {})
        )

        assert result == {"result": "ok"}
        assert "callbacks" in captured_config
        assert any(
            isinstance(cb, TokenCapturingHandler) for cb in captured_config["callbacks"]
        )

    def test_preserves_existing_callbacks(self) -> None:
        from deepagent_temporal.activity import StreamingNodeWrapper

        original = MagicMock()
        captured_config: dict[str, Any] = {}

        async def mock_ainvoke(
            input: Any, config: Any, **kwargs: Any
        ) -> dict[str, str]:
            captured_config.update(config)
            return {"result": "ok"}

        original.ainvoke = mock_ainvoke

        existing_callback = MagicMock()
        wrapper = StreamingNodeWrapper(original, "call_model")

        import asyncio

        asyncio.get_event_loop().run_until_complete(
            wrapper.ainvoke({"x": 1}, {"callbacks": [existing_callback]})
        )

        callbacks = captured_config["callbacks"]
        assert existing_callback in callbacks
        assert len(callbacks) == 2  # existing + TokenCapturingHandler

    def test_delegates_attributes(self) -> None:
        from deepagent_temporal.activity import StreamingNodeWrapper

        original = MagicMock()
        original.name = "test_runnable"
        original.some_attr = "some_value"

        wrapper = StreamingNodeWrapper(original, "call_model")
        assert wrapper.name == "test_runnable"
        assert wrapper.some_attr == "some_value"


class TestWrapGraphForStreaming:
    def test_wraps_all_nodes(self) -> None:
        from typing import TypedDict

        # Create a simple graph
        from langgraph.graph import StateGraph

        from deepagent_temporal.activity import (
            StreamingNodeWrapper,
            wrap_graph_for_streaming,
        )

        class S(TypedDict):
            x: str

        g = StateGraph(S)
        g.add_node("a", lambda s: {"x": "a"})
        g.add_node("b", lambda s: {"x": "b"})
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.set_finish_point("b")
        compiled = g.compile()

        wrap_graph_for_streaming(compiled)

        for name in ["a", "b"]:
            assert isinstance(compiled.nodes[name].bound, StreamingNodeWrapper)

    def test_wraps_specific_nodes(self) -> None:
        from typing import TypedDict

        from langgraph.graph import StateGraph

        from deepagent_temporal.activity import (
            StreamingNodeWrapper,
            wrap_graph_for_streaming,
        )

        class S(TypedDict):
            x: str

        g = StateGraph(S)
        g.add_node("a", lambda s: {"x": "a"})
        g.add_node("b", lambda s: {"x": "b"})
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.set_finish_point("b")
        compiled = g.compile()

        wrap_graph_for_streaming(compiled, node_names=["a"])

        assert isinstance(compiled.nodes["a"].bound, StreamingNodeWrapper)
        assert not isinstance(compiled.nodes["b"].bound, StreamingNodeWrapper)

    def test_idempotent_wrapping(self) -> None:
        from typing import TypedDict

        from langgraph.graph import StateGraph

        from deepagent_temporal.activity import (
            StreamingNodeWrapper,
            wrap_graph_for_streaming,
        )

        class S(TypedDict):
            x: str

        g = StateGraph(S)
        g.add_node("a", lambda s: {"x": "a"})
        g.set_entry_point("a")
        g.set_finish_point("a")
        compiled = g.compile()

        wrap_graph_for_streaming(compiled)
        wrap_graph_for_streaming(compiled)  # Should not double-wrap

        wrapper = compiled.nodes["a"].bound
        assert isinstance(wrapper, StreamingNodeWrapper)
        assert not isinstance(wrapper._original, StreamingNodeWrapper)


class TestStreamingNodeWrapperE2E:
    """End-to-end test: node with a mock LLM that fires callbacks."""

    @pytest.mark.asyncio
    async def test_callback_captures_tokens_during_ainvoke(self) -> None:
        """Verify that tokens are captured when a chat model fires callbacks."""
        from deepagent_temporal.activity import StreamingNodeWrapper

        captured_tokens: list[TokenEvent] = []

        # Create a fake runnable that simulates a chat model firing callbacks
        class FakeRunnable:
            name = "fake"

            async def ainvoke(
                self, input: Any, config: Any, **kwargs: Any
            ) -> dict[str, str]:
                # Simulate the chat model firing callbacks
                callbacks = config.get("callbacks", [])
                run_id = uuid4()
                for cb in callbacks:
                    if hasattr(cb, "on_llm_new_token"):
                        for token in ["Hello", " ", "World"]:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=token),
                            )
                            cb.on_llm_new_token(token, chunk=chunk, run_id=run_id)
                        cb.on_llm_end(MagicMock(), run_id=run_id)
                return {"result": "Hello World"}

        wrapper = StreamingNodeWrapper(
            FakeRunnable(),
            "call_model",
            token_sink=captured_tokens.append,
        )

        result = await wrapper.ainvoke({"messages": []}, {})

        assert result == {"result": "Hello World"}
        # 3 tokens + 1 final event
        assert len(captured_tokens) == 4
        assert captured_tokens[0].token == "Hello"
        assert captured_tokens[1].token == " "
        assert captured_tokens[2].token == "World"
        assert captured_tokens[3].is_final is True
        assert [e.index for e in captured_tokens[:3]] == [0, 1, 2]
