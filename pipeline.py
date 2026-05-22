"""Pipeline topology.

A fixed sequence of three LLM calls. Each step does one thing and the output
flows linearly to the next step. No runtime control flow.

Step 1: extract the claim and mechanism.
Step 2: extract evidence and limitations.
Step 3: compose the final 300-word summary.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from shared.llm import call
from shared.task import TASK_BRIEF, load_paper


def step_extract_claim_and_mechanism(paper: str) -> dict:
    return call(
        system="You are a careful research reader. Extract specific claims, not generalities.",
        messages=[
            {
                "role": "user",
                "content": (
                    "From this paper, extract:\n"
                    "(a) the central claim in one sentence,\n"
                    "(b) the mechanism or method in 2-3 sentences.\n"
                    "Be specific.\n\n---\n"
                    f"{paper}"
                ),
            }
        ],
        max_tokens=400,
    )


def step_extract_evidence_and_limitations(paper: str) -> dict:
    return call(
        system="You are a careful research reader. Quote numbers where present.",
        messages=[
            {
                "role": "user",
                "content": (
                    "From this paper, extract:\n"
                    "(a) the evidence supporting the central claim in 2-3 sentences with numbers where present,\n"
                    "(b) the limitations the authors acknowledge in 1-2 sentences.\n\n---\n"
                    f"{paper}"
                ),
            }
        ],
        max_tokens=400,
    )


def step_compose_summary(claim_block: str, evidence_block: str) -> dict:
    return call(
        system="You write tight, accurate research summaries. Hit the word count.",
        messages=[
            {
                "role": "user",
                "content": (
                    f"{TASK_BRIEF}\n\n"
                    f"CLAIM AND MECHANISM:\n{claim_block}\n\n"
                    f"EVIDENCE AND LIMITATIONS:\n{evidence_block}"
                ),
            }
        ],
        max_tokens=700,
    )


def run() -> dict:
    t0 = time.time()
    paper = load_paper()

    print("→ Step 1: extract claim and mechanism")
    s1 = step_extract_claim_and_mechanism(paper)

    print("→ Step 2: extract evidence and limitations")
    s2 = step_extract_evidence_and_limitations(paper)

    print("→ Step 3: compose summary")
    s3 = step_compose_summary(s1["text"], s2["text"])

    total_cost = s1["cost_usd"] + s2["cost_usd"] + s3["cost_usd"]
    total_input = s1["input_tokens"] + s2["input_tokens"] + s3["input_tokens"]
    total_output = s1["output_tokens"] + s2["output_tokens"] + s3["output_tokens"]
    total_latency = time.time() - t0

    return {
        "topology": "pipeline",
        "summary": s3["text"],
        "llm_calls": 3,
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

    out = Path(__file__).resolve().parent / "results" / "pipeline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
