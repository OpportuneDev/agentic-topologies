"""Tests for the pipeline topology.

The pipeline is the simplest case: a fixed sequence of three LLM calls. We
verify that the three steps fire in order, that token / cost / call-count
accounting is correct, and that the final summary is the output of step 3.
"""

from __future__ import annotations

from unittest.mock import patch

import pipeline
from tests.conftest import fake_llm_result


def test_pipeline_makes_exactly_three_calls():
    with patch("pipeline.call", return_value=fake_llm_result()) as mock_call:
        result = pipeline.run()
    assert mock_call.call_count == 3
    assert result["llm_calls"] == 3


def test_pipeline_result_shape():
    with patch("pipeline.call", return_value=fake_llm_result()):
        result = pipeline.run()

    assert result["topology"] == "pipeline"
    for key in (
        "summary",
        "llm_calls",
        "total_cost_usd",
        "total_input_tokens",
        "total_output_tokens",
        "total_latency_s",
    ):
        assert key in result, f"missing key: {key}"


def test_pipeline_aggregates_token_counts():
    # Three identical calls of 100 in / 50 out → 300 in / 150 out.
    with patch("pipeline.call", return_value=fake_llm_result(input_tokens=100, output_tokens=50)):
        result = pipeline.run()

    assert result["total_input_tokens"] == 300
    assert result["total_output_tokens"] == 150


def test_pipeline_final_summary_comes_from_last_step():
    responses = [
        fake_llm_result(text="STEP1: claim and mechanism"),
        fake_llm_result(text="STEP2: evidence and limitations"),
        fake_llm_result(text="STEP3: the final 300-word summary"),
    ]
    with patch("pipeline.call", side_effect=responses):
        result = pipeline.run()

    assert result["summary"] == "STEP3: the final 300-word summary"


def test_pipeline_returns_three_steps_with_metrics():
    with patch("pipeline.call", return_value=fake_llm_result(input_tokens=10, output_tokens=5)):
        result = pipeline.run()

    assert "steps" in result
    assert len(result["steps"]) == 3
    for step in result["steps"]:
        assert {"label", "kind", "input_tokens", "output_tokens", "cost_usd", "latency_s"} <= step.keys()
        assert step["kind"] == "llm"
        assert step["input_tokens"] == 10
        assert step["output_tokens"] == 5
    # Step totals should sum to the run totals.
    assert sum(s["input_tokens"] for s in result["steps"]) == result["total_input_tokens"]
    assert sum(s["output_tokens"] for s in result["steps"]) == result["total_output_tokens"]


def test_pipeline_accepts_custom_paper():
    """`run(paper=...)` should pass the custom text through, not load from disk."""
    custom = "PAPER_MARKER_xyz123 — this is a fake paper."
    with patch("pipeline.call", return_value=fake_llm_result()) as mock_call:
        pipeline.run(paper=custom)

    # Steps 1 and 2 receive the raw paper; step 3 receives prior outputs.
    first_call_prompt = mock_call.call_args_list[0].kwargs["messages"][0]["content"]
    second_call_prompt = mock_call.call_args_list[1].kwargs["messages"][0]["content"]
    assert "PAPER_MARKER_xyz123" in first_call_prompt
    assert "PAPER_MARKER_xyz123" in second_call_prompt


def test_pipeline_step3_receives_outputs_of_step1_and_step2():
    """Step 3 must compose using the previous two outputs, not the raw paper."""
    responses = [
        fake_llm_result(text="CLAIM_BLOCK"),
        fake_llm_result(text="EVIDENCE_BLOCK"),
        fake_llm_result(text="final summary"),
    ]
    with patch("pipeline.call", side_effect=responses) as mock_call:
        pipeline.run()

    # Inspect the third call.
    third_call_kwargs = mock_call.call_args_list[2].kwargs
    composed_prompt = third_call_kwargs["messages"][0]["content"]
    assert "CLAIM_BLOCK" in composed_prompt
    assert "EVIDENCE_BLOCK" in composed_prompt
