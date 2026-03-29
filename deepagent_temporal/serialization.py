"""Payload size validation for Temporal event history.

Temporal has a hard limit on event payload size (~2 MB default per event,
configurable per namespace). LangGraph agent state — especially accumulated
message history — can exceed this limit during long conversations.

This module provides a ``validate_payload_size`` guard that checks
serialized state size and raises before Temporal silently rejects or
corrupts the payload.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

#: Default warning threshold in bytes (1 MB).
DEFAULT_WARN_BYTES: int = 1_000_000

#: Default error threshold in bytes (2 MB — Temporal's default per-event limit).
DEFAULT_ERROR_BYTES: int = 2_000_000


class PayloadTooLargeError(Exception):
    """Raised when serialized state exceeds the configured error threshold."""


def estimate_payload_size(state: dict[str, Any]) -> int:
    """Estimate the serialized size of a state dict in bytes.

    Uses ``json.dumps`` as a proxy for Temporal's payload serialization.
    The actual Temporal payload (protobuf + codec) may differ slightly,
    but JSON length is a reasonable lower bound.

    Falls back to ``sys.getsizeof`` for non-JSON-serializable state.
    """
    try:
        return len(json.dumps(state, default=str).encode())
    except (TypeError, ValueError, OverflowError):
        return sys.getsizeof(state)


def validate_payload_size(
    state: dict[str, Any],
    *,
    warn_bytes: int = DEFAULT_WARN_BYTES,
    error_bytes: int = DEFAULT_ERROR_BYTES,
) -> int:
    """Check that ``state`` fits within Temporal's payload limits.

    Args:
        state: The state dict to validate.
        warn_bytes: Log a warning when size exceeds this threshold.
        error_bytes: Raise ``PayloadTooLargeError`` when size exceeds this.

    Returns:
        The estimated size in bytes.

    Raises:
        PayloadTooLargeError: When the estimated size exceeds ``error_bytes``.
    """
    size = estimate_payload_size(state)

    if size >= error_bytes:
        raise PayloadTooLargeError(
            f"State payload ({size:,} bytes) exceeds the {error_bytes:,}-byte "
            f"Temporal event limit. Consider using the claim-check pattern: "
            f"offload large state (e.g., message history) to external storage "
            f"(S3, Redis) and pass only a reference through the workflow. "
            f"See docs/serialization.md for details."
        )

    if size >= warn_bytes:
        logger.warning(
            "State payload (%s bytes) approaching Temporal event limit "
            "(%s bytes). Consider summarizing conversation history or "
            "offloading large state to external storage.",
            f"{size:,}",
            f"{error_bytes:,}",
        )

    return size
