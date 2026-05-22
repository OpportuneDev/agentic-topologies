"""Thin Anthropic SDK wrapper with cost and latency tracking."""

from __future__ import annotations

import time
from typing import Any

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"

# Per-million-token prices in USD for the default model.
# Update if the model or pricing changes.
PRICE_INPUT_PER_MTOK = 3.0
PRICE_OUTPUT_PER_MTOK = 15.0

_client: Anthropic | None = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * PRICE_INPUT_PER_MTOK
        + output_tokens * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000


def call(
    messages: list[dict[str, Any]],
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Single LLM call. Returns text plus token / cost / latency metadata."""
    t0 = time.time()

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools

    response = client().messages.create(**kwargs)
    latency = time.time() - t0

    text_parts = [block.text for block in response.content if block.type == "text"]
    text = "\n".join(text_parts) if text_parts else ""

    return {
        "text": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": _cost_usd(response.usage.input_tokens, response.usage.output_tokens),
        "latency_s": latency,
        "raw": response,
    }
