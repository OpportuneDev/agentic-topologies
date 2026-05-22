"""Tests for the orchestrator + specialists topology.

We mock two boundaries:

  1. `orchestrator.client()` — the Anthropic client used by the orchestrator
     loop. We script its `.messages.create()` to return a deterministic
     sequence of tool-use responses, simulating an orchestrator that asks for
     all three extractions, then composes, then finishes.
  2. `orchestrator.call` — the helper used by specialists to make their own
     LLM calls. Returns a fixed fake result.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import orchestrator
from tests.conftest import fake_llm_result


def _tool_use_block(name: str, tool_input: dict[str, Any], block_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _fake_response(
    content_blocks: list[Any],
    stop_reason: str = "tool_use",
    input_tokens: int = 200,
    output_tokens: int = 80,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content_blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_scripted_client(responses: list[SimpleNamespace]) -> MagicMock:
    """Build a mock Anthropic client whose `.messages.create` returns the
    scripted responses in order."""
    iterator = iter(responses)

    def create(*args, **kwargs):
        return next(iterator)

    mock = MagicMock()
    mock.messages.create.side_effect = create
    return mock


def test_orchestrator_happy_path():
    """Three extractions in one turn, compose in turn 2, finish in turn 3."""
    responses = [
        _fake_response(
            content_blocks=[
                _tool_use_block("extract_claim", {}, "tu1"),
                _tool_use_block("extract_evidence", {}, "tu2"),
                _tool_use_block("extract_limitations", {}, "tu3"),
            ]
        ),
        _fake_response(
            content_blocks=[
                _tool_use_block(
                    "compose_summary",
                    {"claim": "c", "evidence": "e", "limitations": "l"},
                    "tu4",
                ),
            ]
        ),
        _fake_response(
            content_blocks=[
                _tool_use_block("finish", {"summary": "Final 300-word summary."}, "tu5"),
            ]
        ),
    ]
    mock_client = _make_scripted_client(responses)

    with patch("orchestrator.client", return_value=mock_client), patch(
        "orchestrator.call", return_value=fake_llm_result(text="specialist output")
    ) as mock_specialist:
        result = orchestrator.run()

    assert result["topology"] == "orchestrator+specialists"
    assert result["summary"] == "Final 300-word summary."
    # 3 orchestrator turns + 3 extraction specialists + 1 compose specialist = 7
    assert result["llm_calls"] == 7
    # 4 specialist calls actually happened.
    assert mock_specialist.call_count == 4


def test_orchestrator_accumulates_costs_from_both_layers():
    """Token counts must include orchestrator turns AND specialist calls."""
    responses = [
        _fake_response(
            content_blocks=[_tool_use_block("extract_claim", {}, "tu1")],
            input_tokens=1000,
            output_tokens=100,
        ),
        _fake_response(
            content_blocks=[
                _tool_use_block("finish", {"summary": "done"}, "tu2"),
            ],
            input_tokens=1100,
            output_tokens=50,
        ),
    ]
    mock_client = _make_scripted_client(responses)

    specialist_result = fake_llm_result(input_tokens=500, output_tokens=200)
    with patch("orchestrator.client", return_value=mock_client), patch(
        "orchestrator.call", return_value=specialist_result
    ):
        result = orchestrator.run()

    # Orchestrator: 1000+1100 in, 100+50 out. Specialist: 500 in, 200 out.
    assert result["total_input_tokens"] == 1000 + 1100 + 500
    assert result["total_output_tokens"] == 100 + 50 + 200


def test_orchestrator_stops_when_max_turns_reached():
    """If the model never calls `finish`, we must terminate at max_turns."""
    # Always return one harmless tool_use → loop will run forever without a cap.
    always_extract = _fake_response(
        content_blocks=[_tool_use_block("extract_claim", {}, "tu_x")]
    )

    def infinite_create(*args, **kwargs):
        return always_extract

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = infinite_create

    with patch("orchestrator.client", return_value=mock_client), patch(
        "orchestrator.call", return_value=fake_llm_result()
    ):
        result = orchestrator.run(max_turns=4)

    # 4 orchestrator turns × (1 turn + 1 specialist) = 8 calls
    assert result["llm_calls"] == 8
    assert "no final summary produced" in result["summary"]


def test_orchestrator_stops_on_end_turn():
    """A stop_reason of 'end_turn' with no tools should break the loop early."""
    responses = [
        _fake_response(content_blocks=[], stop_reason="end_turn"),
    ]
    mock_client = _make_scripted_client(responses)

    with patch("orchestrator.client", return_value=mock_client), patch(
        "orchestrator.call", return_value=fake_llm_result()
    ) as mock_specialist:
        result = orchestrator.run()

    assert result["llm_calls"] == 1
    assert mock_specialist.call_count == 0


def test_orchestrator_unknown_tool_raises():
    """Catch a misnamed tool early — silently returning empty would mask bugs."""
    responses = [
        _fake_response(
            content_blocks=[_tool_use_block("not_a_real_tool", {}, "tu_bad")]
        ),
    ]
    mock_client = _make_scripted_client(responses)

    with patch("orchestrator.client", return_value=mock_client), patch(
        "orchestrator.call", return_value=fake_llm_result()
    ):
        try:
            orchestrator.run(max_turns=2)
        except ValueError as e:
            assert "Unknown tool" in str(e)
        else:
            raise AssertionError("expected ValueError for unknown tool")


def test_orchestrator_specialist_dispatch():
    """Each specialist tool name routes to its corresponding Python function."""
    paper = "fake paper text"
    fake = fake_llm_result(text="dispatched")

    with patch("orchestrator.call", return_value=fake):
        text, meta = orchestrator._dispatch("extract_claim", {}, paper)
        assert text == "dispatched"
        assert meta["input_tokens"] == fake["input_tokens"]

        text, meta = orchestrator._dispatch("extract_evidence", {}, paper)
        assert text == "dispatched"

        text, meta = orchestrator._dispatch("extract_limitations", {}, paper)
        assert text == "dispatched"

        text, meta = orchestrator._dispatch(
            "compose_summary",
            {"claim": "c", "evidence": "e", "limitations": "l"},
            paper,
        )
        assert text == "dispatched"
