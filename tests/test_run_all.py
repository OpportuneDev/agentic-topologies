"""Smoke test for run_all.py.

We mock all three topologies' `run()` functions and verify that run_all writes
the expected JSON files and prints the comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import run_all


def _fake_run_result(topology: str, calls: int, cost: float, latency: float) -> dict:
    return {
        "topology": topology,
        "summary": f"summary from {topology}",
        "llm_calls": calls,
        "total_cost_usd": cost,
        "total_input_tokens": calls * 1000,
        "total_output_tokens": calls * 200,
        "total_latency_s": latency,
    }


def test_run_all_writes_three_json_files(tmp_path: Path, capsys, monkeypatch):
    # Redirect run_all's RESULTS_DIR to a tmp path so we don't clobber the
    # repo's checked-in results.
    monkeypatch.setattr(run_all, "RESULTS_DIR", tmp_path)

    fake_results = {
        "pipeline": _fake_run_result("pipeline", 3, 0.01, 2.0),
        "orchestrator": _fake_run_result("orchestrator+specialists", 7, 0.05, 5.0),
        "swarm": _fake_run_result("flat_swarm", 9, 0.20, 12.0),
    }

    def fake_run_module(module_path: str) -> dict:
        return fake_results[module_path]

    with patch.object(run_all, "_run_module", side_effect=fake_run_module):
        run_all.main()

    for name in ("pipeline.json", "orchestrator.json", "swarm.json"):
        path = tmp_path / name
        assert path.exists(), f"{name} not written"
        data = json.loads(path.read_text())
        assert "topology" in data
        assert "summary" in data

    out = capsys.readouterr().out
    assert "Comparison" in out
    assert "Multipliers vs pipeline" in out
    assert "5.00x" in out or "5.0x" in out  # orchestrator cost: 0.05 / 0.01
