"""Tests for the flat swarm topology.

The swarm has two interesting properties to verify:
  1. Round-robin order: researcher → critic → editor, repeating.
  2. Early termination when the editor emits a `FINAL_SUMMARY:` prefix.
  3. Graceful fallback: if no FINAL_SUMMARY appears, the last editor turn wins.
"""

from __future__ import annotations

from unittest.mock import patch

import swarm
from tests.conftest import fake_llm_result


def test_swarm_runs_full_round_robin_when_no_final_summary():
    with patch("swarm.call", return_value=fake_llm_result(text="placeholder draft")):
        result = swarm.run()

    assert result["topology"] == "flat_swarm"
    assert result["llm_calls"] == swarm.MAX_TURNS
    assert result["terminated_via_final_summary_prefix"] is False


def test_swarm_round_robin_order():
    """The agent on turn N must be ORDER[N % 3]."""
    seen_agents: list[str] = []
    original_call = swarm.call

    def capture(messages, system=None, **kwargs):
        # Identify which agent's system prompt we received.
        for name, cfg in swarm.AGENTS.items():
            if cfg["system"] == system:
                seen_agents.append(name)
                break
        return fake_llm_result(text="draft")

    with patch("swarm.call", side_effect=capture):
        swarm.run()

    assert seen_agents == [swarm.ORDER[i % 3] for i in range(swarm.MAX_TURNS)]
    _ = original_call  # silence "unused" lint if it ever creeps in


def test_swarm_terminates_when_editor_emits_final_summary():
    """Editor on turn 3 emits FINAL_SUMMARY: → loop must stop there."""
    call_index = [0]

    def scripted(messages, system=None, **kwargs):
        call_index[0] += 1
        # Turn 3 is the editor (researcher=1, critic=2, editor=3).
        if call_index[0] == 3:
            return fake_llm_result(text="FINAL_SUMMARY: This is the agreed summary.")
        return fake_llm_result(text="draft contribution")

    with patch("swarm.call", side_effect=scripted):
        result = swarm.run()

    assert result["llm_calls"] == 3
    assert result["summary"] == "This is the agreed summary."
    assert result["terminated_via_final_summary_prefix"] is True


def test_swarm_falls_back_to_last_editor_turn_if_no_prefix():
    """Without FINAL_SUMMARY:, the last editor turn's text is the result."""
    # 9 turns, editors on turns 3, 6, 9. The last editor message should win.
    editor_call_count = [0]

    def scripted(messages, system=None, **kwargs):
        if system == swarm.AGENTS["editor"]["system"]:
            editor_call_count[0] += 1
            return fake_llm_result(text=f"editor draft #{editor_call_count[0]}")
        return fake_llm_result(text="other agent draft")

    with patch("swarm.call", side_effect=scripted):
        result = swarm.run()

    assert result["summary"] == "editor draft #3"
    assert result["terminated_via_final_summary_prefix"] is False


def test_swarm_scratchpad_grows_each_turn():
    """Each later turn should see input tokens that grow with the scratchpad."""
    input_sizes: list[int] = []

    def measure(messages, system=None, **kwargs):
        # Approximate the scratchpad size from the prompt length.
        prompt = messages[0]["content"]
        input_sizes.append(len(prompt))
        return fake_llm_result(text="X" * 200)

    with patch("swarm.call", side_effect=measure):
        swarm.run()

    # Each turn appends a previous-agent contribution to the prompt, so the
    # prompt length is monotonically non-decreasing.
    assert input_sizes == sorted(input_sizes), "swarm input size should grow over time"
    assert input_sizes[-1] > input_sizes[0], "last prompt should be bigger than first"
