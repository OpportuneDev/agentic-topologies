"""Flat swarm topology.

Three peer agents — researcher, critic, editor — take turns contributing to a
shared scratchpad. There is no central coordinator. Each agent sees the full
prior scratchpad on every turn.

This is a deliberately fair implementation, not a strawman. The failure modes
are emergent rather than scripted:

  - Input tokens grow linearly per turn because each agent sees the full
    scratchpad. By turn 8 the input cost has compounded substantially.
  - Agents agree with each other in ways that look thoughtful but don't move
    the work forward.
  - The editor must guess when the team is done, because no one is in charge
    of "are we done."

The final summary is whatever the editor produced last. If the editor never
prefixes a final version with `FINAL_SUMMARY:`, we take their last turn's
content as the result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from shared.llm import call
from shared.task import TASK_BRIEF, load_paper

AGENTS = {
    "researcher": {
        "system": (
            "You are the researcher in a 3-person team summarizing a paper. "
            "Your job is to draft content drawn from the paper itself. Read the "
            "scratchpad, see what the team needs, and either contribute new "
            "content or refine based on critic feedback. Keep each contribution "
            "focused. The team is producing a 300-word summary covering claim, "
            "mechanism, evidence, limitations, and one open question."
        ),
    },
    "critic": {
        "system": (
            "You are the critic in a 3-person team. Review the scratchpad and "
            "point out specific gaps, inaccuracies, or vague language. Be terse "
            "and specific. Do not write the summary yourself; that is the editor's "
            "job. The team is producing a 300-word summary covering claim, "
            "mechanism, evidence, limitations, and one open question."
        ),
    },
    "editor": {
        "system": (
            "You are the editor in a 3-person team. Assemble the team's "
            "contributions into a clean 300-word summary covering claim, "
            "mechanism, evidence, limitations, and one open question. When you "
            "produce a final version you are satisfied with, prefix it with "
            "FINAL_SUMMARY: on its own line. Until then, share drafts for the "
            "critic to review."
        ),
    },
}

ORDER = ["researcher", "critic", "editor"]
MAX_TURNS = 9


def _format_scratchpad(scratchpad: list[tuple[str, str]]) -> str:
    if not scratchpad:
        return "(no prior contributions yet)"
    return "\n\n".join(f"[{name}]:\n{content}" for name, content in scratchpad)


def run() -> dict:
    t0 = time.time()
    paper = load_paper()

    scratchpad: list[tuple[str, str]] = []
    total_cost = 0.0
    total_input = 0
    total_output = 0
    final_summary: str | None = None

    for turn in range(MAX_TURNS):
        agent_name = ORDER[turn % len(ORDER)]
        agent = AGENTS[agent_name]

        history = _format_scratchpad(scratchpad)

        prompt = (
            f"Paper to summarize:\n---\n{paper}\n---\n\n"
            f"Task: {TASK_BRIEF}\n\n"
            f"Team scratchpad so far:\n{history}\n\n"
            f"It is your turn ({agent_name}). Contribute."
        )

        result = call(
            system=agent["system"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )

        total_cost += result["cost_usd"]
        total_input += result["input_tokens"]
        total_output += result["output_tokens"]
        scratchpad.append((agent_name, result["text"]))

        print(
            f"→ Turn {turn + 1} [{agent_name}]: "
            f"{result['input_tokens']} in / {result['output_tokens']} out, "
            f"${result['cost_usd']:.4f}"
        )

        if agent_name == "editor" and "FINAL_SUMMARY:" in result["text"]:
            final_summary = result["text"].split("FINAL_SUMMARY:", 1)[1].strip()
            break

    if final_summary is None:
        editor_contributions = [c for n, c in scratchpad if n == "editor"]
        final_summary = (
            editor_contributions[-1]
            if editor_contributions
            else "(no editor output produced)"
        )

    total_latency = time.time() - t0

    return {
        "topology": "flat_swarm",
        "summary": final_summary,
        "llm_calls": len(scratchpad),
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_latency_s": round(total_latency, 2),
        "agent_order": ORDER,
        "max_turns": MAX_TURNS,
        "terminated_via_final_summary_prefix": "FINAL_SUMMARY:" in (final_summary or ""),
    }


def main() -> None:
    result = run()
    print("\n" + "=" * 60)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "summary"},
            indent=2,
        )
    )
    print("\nSUMMARY:")
    print(result["summary"])

    out = Path(__file__).resolve().parent / "results" / "swarm.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
