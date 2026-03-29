"""Tests for payload size validation."""

from __future__ import annotations

import logging

import pytest

from deepagent_temporal.serialization import (
    PayloadTooLargeError,
    estimate_payload_size,
    validate_payload_size,
)


class TestEstimatePayloadSize:
    def test_empty_state(self) -> None:
        size = estimate_payload_size({})
        assert size == 2  # "{}"

    def test_simple_state(self) -> None:
        state = {"messages": ["hello", "world"]}
        size = estimate_payload_size(state)
        assert size > 0

    def test_large_state(self) -> None:
        state = {"messages": ["x" * 1000 for _ in range(1000)]}
        size = estimate_payload_size(state)
        assert size > 1_000_000


class TestValidatePayloadSize:
    def test_small_state_passes(self) -> None:
        state = {"messages": ["hello"]}
        size = validate_payload_size(state)
        assert size > 0

    def test_warns_on_large_state(self, caplog: pytest.LogCaptureFixture) -> None:
        state = {"data": "x" * 1_500_000}
        with caplog.at_level(logging.WARNING):
            size = validate_payload_size(
                state, warn_bytes=1_000_000, error_bytes=5_000_000
            )
        assert size > 1_000_000
        assert "approaching Temporal event limit" in caplog.text

    def test_errors_on_oversized_state(self) -> None:
        state = {"data": "x" * 3_000_000}
        with pytest.raises(PayloadTooLargeError, match="exceeds the"):
            validate_payload_size(state, error_bytes=2_000_000)

    def test_custom_thresholds(self) -> None:
        state = {"data": "x" * 500}
        with pytest.raises(PayloadTooLargeError):
            validate_payload_size(state, error_bytes=100)

    def test_no_warning_below_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        state = {"messages": ["hello"]}
        with caplog.at_level(logging.WARNING):
            validate_payload_size(state, warn_bytes=1_000_000)
        assert "approaching" not in caplog.text
