"""Unit tests for the shared utilities. No LLM calls happen here."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared import llm
from shared.task import DEFAULT_PAPER_PATH, TASK_BRIEF, load_paper


def test_cost_calculation_zero_tokens():
    assert llm._cost_usd(0, 0) == 0.0


def test_cost_calculation_input_only():
    # 1M input tokens at $3/Mtok = $3.00
    assert llm._cost_usd(1_000_000, 0) == pytest.approx(3.0)


def test_cost_calculation_output_only():
    # 1M output tokens at $15/Mtok = $15.00
    assert llm._cost_usd(0, 1_000_000) == pytest.approx(15.0)


def test_cost_calculation_mixed():
    # 1000 input + 500 output = 1000*3/1M + 500*15/1M = 0.003 + 0.0075 = 0.0105
    assert llm._cost_usd(1000, 500) == pytest.approx(0.0105)


def test_default_paper_exists():
    assert DEFAULT_PAPER_PATH.exists(), "bundled sample paper must ship with the repo"


def test_load_paper_default_nonempty():
    paper = load_paper()
    assert len(paper) > 500, "the bundled paper should be a substantial document"
    assert "Depth-Aware" in paper, "bundled paper looks wrong"


def test_load_paper_custom_path(tmp_path: Path):
    custom = tmp_path / "custom.md"
    custom.write_text("# Custom paper\n\nHello world.")
    assert load_paper(custom) == "# Custom paper\n\nHello world."


def test_task_brief_mentions_required_sections():
    # The TASK_BRIEF is shared across all three topologies; if someone edits it
    # to drop a required section, all three implementations silently drift.
    required = ["claim", "mechanism", "evidence", "limitation", "open question"]
    for term in required:
        assert term.lower() in TASK_BRIEF.lower(), f"TASK_BRIEF missing '{term}'"
