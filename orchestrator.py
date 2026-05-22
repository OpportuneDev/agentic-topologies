"""Orchestrator + specialists topology.

One main agent holds context and routes to bounded specialists via Anthropic
tool use. Specialists are deterministic Python functions that themselves make
LLM calls. The orchestrator decides which specialist to call when, and when to
finish.

This earns its complexity over a pipeline when the task structure genuinely
varies at runtime. For a static task like ours, you can see it produce a
result comparable to the pipeline at higher cost. That's the point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from shared.llm import DEFAULT_MODEL, call, client
from shared.task import TASK_BRIEF, load_paper

# ---------------------------------------------------------------------------
# Specialists: each does one thing and returns a string.
# ---------------------------------------------------------------------------


def specialist_extract_claim(paper: str) -> dict:
    return call(
        system="You extract specific claims, not generalities.",
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the central claim in one sentence and the mechanism "
                    f"in 2-3 sentences.\n\n---\n{paper}"
                ),
            }
        ],
        max_tokens=400,
    )


def specialist_extract_evidence(paper: str) -> dict:
    return call(
        system="You extract empirical evidence with numbers.",
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the evidence supporting the main claims in 2-3 sentences. "
                    f"Include numbers where present.\n\n---\n{paper}"
                ),
            }
        ],
        max_tokens=400,
    )


def specialist_extract_limitations(paper: str) -> dict:
    return call(
        system="You extract authors' acknowledged limitations.",
        messages=[
            {
                "role": "user",
                "content": f"Extract the limitations the authors acknowledge in 2-3 sentences.\n\n---\n{paper}",
            }
        ],
        max_tokens=300,
    )


def specialist_compose(claim: str, evidence: str, limitations: str) -> dict:
    return call(
        system="You write tight, accurate research summaries. Hit the word count.",
        messages=[
            {
                "role": "user",
                "content": (
                    f"{TASK_BRIEF}\n\n"
                    f"CLAIM AND MECHANISM:\n{claim}\n\n"
                    f"EVIDENCE:\n{evidence}\n\n"
                    f"LIMITATIONS:\n{limitations}"
                ),
            }
        ],
        max_tokens=700,
    )


# ---------------------------------------------------------------------------
# Tool definitions exposed to the orchestrator.
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "extract_claim",
        "description": "Extract the paper's central claim and mechanism. Usually called first.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "extract_evidence",
        "description": "Extract evidence supporting the paper's main claims.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "extract_limitations",
        "description": "Extract the limitations the authors acknowledge.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "compose_summary",
        "description": "Compose the 300-word summary. Requires claim, evidence, and limitations to be already extracted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "Output from extract_claim."},
                "evidence": {"type": "string", "description": "Output from extract_evidence."},
                "limitations": {"type": "string", "description": "Output from extract_limitations."},
            },
            "required": ["claim", "evidence", "limitations"],
        },
    },
    {
        "name": "finish",
        "description": "Submit the final 300-word summary and end the task.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


ORCHESTRATOR_SYSTEM = (
    "You are an orchestrator coordinating specialists to produce a 300-word summary "
    "of a research paper. The summary must cover claim, mechanism, evidence, "
    "limitations, and one open question. Call specialists in a sensible order. "
    "When you have the final summary, call `finish` with the summary text."
)


# ---------------------------------------------------------------------------
# Orchestrator loop.
# ---------------------------------------------------------------------------


def _dispatch(tool_name: str, tool_input: dict[str, Any], paper: str) -> tuple[str, dict]:
    """Call the named specialist. Returns (textual_result, llm_metadata).

    `finish` is handled by the caller.
    """
    if tool_name == "extract_claim":
        r = specialist_extract_claim(paper)
        return r["text"], r
    if tool_name == "extract_evidence":
        r = specialist_extract_evidence(paper)
        return r["text"], r
    if tool_name == "extract_limitations":
        r = specialist_extract_limitations(paper)
        return r["text"], r
    if tool_name == "compose_summary":
        r = specialist_compose(
            tool_input.get("claim", ""),
            tool_input.get("evidence", ""),
            tool_input.get("limitations", ""),
        )
        return r["text"], r
    raise ValueError(f"Unknown tool: {tool_name}")


def run(max_turns: int = 12) -> dict:
    t0 = time.time()
    paper = load_paper()

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Paper to summarize:\n\n---\n{paper}\n---\n\n{TASK_BRIEF}\n\nCoordinate the specialists to produce the summary.",
        }
    ]

    total_cost = 0.0
    total_input = 0
    total_output = 0
    llm_calls = 0
    final_summary: str | None = None

    cli = client()

    for turn in range(max_turns):
        t_call = time.time()
        response = cli.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2000,
            system=ORCHESTRATOR_SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        llm_calls += 1
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        total_cost += (
            response.usage.input_tokens * 3 + response.usage.output_tokens * 15
        ) / 1_000_000

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        print(f"→ Turn {turn + 1}: stop_reason={response.stop_reason}, tools_called={[t.name for t in tool_uses]}")

        if response.stop_reason == "end_turn" or not tool_uses:
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            if tu.name == "finish":
                final_summary = tu.input.get("summary", "")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": "Task complete.",
                    }
                )
            else:
                text, meta = _dispatch(tu.name, tu.input, paper)
                llm_calls += 1
                total_input += meta["input_tokens"]
                total_output += meta["output_tokens"]
                total_cost += meta["cost_usd"]
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": text}
                )

        messages.append({"role": "user", "content": tool_results})

        if final_summary is not None:
            break

    total_latency = time.time() - t0

    return {
        "topology": "orchestrator+specialists",
        "summary": final_summary or "(no final summary produced)",
        "llm_calls": llm_calls,
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_latency_s": round(total_latency, 2),
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

    out = Path(__file__).resolve().parent / "results" / "orchestrator.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
