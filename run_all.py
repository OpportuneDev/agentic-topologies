"""Run all three topologies and print a comparison table.

Each topology solves the same task (summarize the bundled paper in 300 words)
so cost, latency, and call-count numbers are directly comparable.

Outputs:
  results/pipeline.json
  results/orchestrator.json
  results/swarm.json
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _run_module(module_path: str) -> dict:
    module = import_module(module_path)
    return module.run()


def _format_row(name: str, r: dict) -> str:
    return (
        f"{name:<28} | {r['llm_calls']:>4} | "
        f"${r['total_cost_usd']:>7.4f} | "
        f"{r['total_input_tokens']:>7} | "
        f"{r['total_output_tokens']:>7} | "
        f"{r['total_latency_s']:>6.2f}s"
    )


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n=== Running pipeline ===")
    p = _run_module("pipeline")
    (RESULTS_DIR / "pipeline.json").write_text(json.dumps(p, indent=2))

    print("\n=== Running orchestrator + specialists ===")
    o = _run_module("orchestrator")
    (RESULTS_DIR / "orchestrator.json").write_text(json.dumps(o, indent=2))

    print("\n=== Running flat swarm ===")
    s = _run_module("swarm")
    (RESULTS_DIR / "swarm.json").write_text(json.dumps(s, indent=2))

    print("\n" + "=" * 80)
    print("Comparison")
    print("=" * 80)
    header = f"{'Topology':<28} | {'Calls':>4} | {'Cost USD':>8} | {'In tok':>7} | {'Out tok':>7} | {'Latency':>7}"
    print(header)
    print("-" * len(header))
    print(_format_row("pipeline", p))
    print(_format_row("orchestrator+specialists", o))
    print(_format_row("flat_swarm", s))

    # Multipliers vs pipeline
    print("\nMultipliers vs pipeline:")
    for name, r in [("orchestrator+specialists", o), ("flat_swarm", s)]:
        cost_mult = r["total_cost_usd"] / p["total_cost_usd"] if p["total_cost_usd"] else float("inf")
        latency_mult = r["total_latency_s"] / p["total_latency_s"] if p["total_latency_s"] else float("inf")
        print(f"  {name:<28} cost {cost_mult:.2f}x   latency {latency_mult:.2f}x")


if __name__ == "__main__":
    main()
