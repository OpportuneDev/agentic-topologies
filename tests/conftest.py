"""Pytest configuration shared by all test modules.

We never let real LLM traffic leave the test process. The Anthropic client is
lazily constructed inside `shared.llm.client()`, and every test patches the
boundary (`shared.llm.call` or the module-local `call` / `client` symbols) so
that real calls cannot happen. We still set a dummy API key so that any
accidental construction of `anthropic.Anthropic()` would fail loudly later
rather than failing at import time.
"""

from __future__ import annotations

import os
from typing import Any


# A clearly fake key — never a real one. Any real network call would 401, but
# tests should mock the boundary before that happens.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")


def fake_llm_result(
    text: str = "fake response",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> dict[str, Any]:
    """Standard shape returned by `shared.llm.call`. Used to mock specialists."""
    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": (input_tokens * 3 + output_tokens * 15) / 1_000_000,
        "latency_s": 0.01,
        "raw": None,
    }
