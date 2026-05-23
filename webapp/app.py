"""Streamlit BYOK demo of the three agentic topologies.

Users paste their own Anthropic API key, choose which topologies to run, and
get a side-by-side comparison of cost, latency, and output quality — plus a
per-step breakdown so they can see exactly why each topology costs what it
costs.

Run locally:
    streamlit run webapp/app.py

Deploy to Streamlit Cloud by pointing at this file.

PRIVACY: the API key never leaves the Streamlit Python process except in
outbound calls to api.anthropic.com. We do not write it to disk, do not log
it, and do not set it as an OS env var (which would leak across sessions in
a shared multi-tenant runtime like Streamlit Cloud). The key is installed
directly into the `anthropic.Anthropic(api_key=...)` client and then
discarded when the script reruns.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so we can pull in the topology modules.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402
from anthropic import Anthropic  # noqa: E402

from shared import llm  # noqa: E402
from shared.task import TASK_BRIEF, load_paper  # noqa: E402

REPO_URL = "https://github.com/OpportuneDev/agentic-topologies"

st.set_page_config(
    page_title="Agentic Topologies — interactive demo",
    page_icon="🔀",
    layout="wide",
)


# ============================================================================
# Helpers
# ============================================================================
def _fmt_money(x: float) -> str:
    return f"${x:.4f}"


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _render_steps_table(steps: list[dict]) -> None:
    """Per-step breakdown for one topology's run."""
    if not steps:
        st.caption("No step detail recorded.")
        return
    rows = []
    for i, s in enumerate(steps, start=1):
        rows.append(
            {
                "#": i,
                "Step": s["label"],
                "In tok": _fmt_int(s["input_tokens"]),
                "Out tok": _fmt_int(s["output_tokens"]),
                "Cost": _fmt_money(s["cost_usd"]),
                "Latency": f"{s['latency_s']:.2f}s",
            }
        )
    st.table(rows)


PIPELINE_FLOW_MD = """
**Flow** — 3 LLM calls, fully linear:

```
📄 Paper
   ↓
[1] LLM: extract claim + mechanism  ──► claim_text
   ↓
[2] LLM: extract evidence + limitations  ──► evidence_text
   ↓
[3] LLM: compose 300-word summary  ←── claim_text + evidence_text
   ↓
📝 Final summary
```

**Why it's cheap:** the paper is read twice (steps 1+2), then step 3 sees only
the short structured outputs of 1+2 — not the paper again. No retries, no
re-planning, no shared context bloat.
"""

ORCHESTRATOR_FLOW_MD = """
**Flow** — one orchestrator LLM in a tool-use loop:

```
📄 Paper
   ↓
┌──────────────────────────────────────────────┐
│ Orchestrator turn 1: paper + task in context │
│   "I'll call all three extractors"           │
└──────────────────────────────────────────────┘
   ↓ parallel tool calls
[specialist: extract_claim]      ──┐
[specialist: extract_evidence]   ──┤── results returned to orchestrator
[specialist: extract_limitations]──┘
   ↓
┌──────────────────────────────────────────────┐
│ Orchestrator turn 2: sees results            │
│   "Time to compose"                          │
└──────────────────────────────────────────────┘
   ↓
[specialist: compose_summary]
   ↓
┌──────────────────────────────────────────────┐
│ Orchestrator turn 3: receives summary        │
│   calls `finish(summary=...)`                │
└──────────────────────────────────────────────┘
   ↓
📝 Final summary
```

**Why it costs more:** the orchestrator turn includes the **entire paper + the
tools schema + tool results so far** in every turn's input. Three orchestrator
turns × full-paper-in-context + four specialist calls = ~7 LLM calls and
~2–3× the input tokens of the pipeline.
"""

SWARM_FLOW_MD = """
**Flow** — round-robin on a shared scratchpad, no central coordinator:

```
📄 Paper
   ↓
Turn 1 [researcher] reads {paper} ─► writes draft to scratchpad
Turn 2 [critic]     reads {paper + scratchpad} ─► critique
Turn 3 [editor]     reads {paper + scratchpad} ─► draft
Turn 4 [researcher] reads {paper + scratchpad} ─► refines
Turn 5 [critic]     reads {paper + scratchpad} ─► critique
Turn 6 [editor]     reads {paper + scratchpad} ─► draft
Turn 7 [researcher] reads {paper + scratchpad}
Turn 8 [critic]     reads {paper + scratchpad}
Turn 9 [editor]     reads {paper + scratchpad} ─► final
```

**Why it's expensive:** every agent re-reads the **paper + every prior turn's
output** as input. By turn 9 the scratchpad alone is ~5000 tokens, on top of
the ~1500-token paper. Input tokens compound quadratically while the output
quality plateaus by turn 4 or 5.
"""

FLOW_BY_KEY = {
    "pipeline": PIPELINE_FLOW_MD,
    "orchestrator": ORCHESTRATOR_FLOW_MD,
    "swarm": SWARM_FLOW_MD,
}


# ============================================================================
# Sidebar: config
# ============================================================================
with st.sidebar:
    st.header("Config")

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-...",
        help="Get one at https://console.anthropic.com/.",
    )
    st.caption(
        "🔒 Used only for this browser session. Sent only to "
        "`api.anthropic.com` — never written to disk, never logged, never "
        "stored as an OS environment variable."
    )

    st.divider()
    st.markdown("**Topologies to run**")
    run_pipeline = st.checkbox(
        "Pipeline",
        value=True,
        help="3 sequential LLM calls. ~$0.03, ~20s.",
    )
    run_orchestrator = st.checkbox(
        "Orchestrator + specialists",
        value=True,
        help="5–8 LLM calls via tool use. ~$0.08, ~40s.",
    )
    run_swarm = st.checkbox(
        "Flat swarm",
        value=False,
        help="9 LLM calls on a shared scratchpad. ~$0.20, ~100s. Slow on purpose.",
    )

    selected_count = sum([run_pipeline, run_orchestrator, run_swarm])

    st.divider()
    st.markdown("**Input paper**")
    paper_choice = st.radio(
        "Source",
        ["Bundled sample paper", "Paste your own"],
        label_visibility="collapsed",
    )

    if paper_choice == "Paste your own":
        paper_text = st.text_area(
            "Paper text (Markdown ok)",
            height=200,
            max_chars=80_000,
            placeholder="Paste the full paper text here…",
        )
    else:
        paper_text = load_paper()
        with st.expander("Preview bundled paper"):
            st.markdown(paper_text)

    st.divider()
    can_run = (
        bool(api_key.strip())
        and bool(paper_text.strip())
        and selected_count > 0
    )

    if selected_count:
        button_label = f"Run {selected_count} topolog" + ("ies" if selected_count != 1 else "y")
    else:
        button_label = "Select a topology"
    run_button = st.button(
        button_label,
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    )

    if not can_run:
        if not api_key.strip():
            st.caption("Enter your API key to enable Run.")
        elif not paper_text.strip():
            st.caption("Paper text is empty.")
        elif selected_count == 0:
            st.caption("Select at least one topology.")

    st.divider()
    st.caption(f"[View source on GitHub]({REPO_URL})")


# ============================================================================
# Header
# ============================================================================
st.title("Agentic Topologies")
st.caption(
    "Same task, three architectures. Make the cost, latency, and quality "
    "differences observable rather than asserted."
)


# ============================================================================
# Run handler
# ============================================================================
if run_button:
    # Install the user-provided key directly into the Anthropic client.
    # We do NOT touch os.environ — that would persist the key in process-wide
    # state and could leak across concurrent sessions on a shared host.
    llm._client = Anthropic(api_key=api_key.strip())

    # Lazy import — keeps a bad install error out of the top-of-script path.
    import orchestrator as orchestrator_mod  # noqa: E402
    import pipeline as pipeline_mod  # noqa: E402
    import swarm as swarm_mod  # noqa: E402

    new_results: dict[str, dict] = {}

    def _render_complete(status, r: dict) -> None:
        status.update(
            label=(
                f"{r['topology']} complete · {r['llm_calls']} calls · "
                f"${r['total_cost_usd']:.4f} · {r['total_latency_s']:.1f}s"
            ),
            state="complete",
            expanded=False,
        )

    if run_pipeline:
        with st.status(
            "Running pipeline — 3 sequential LLM calls (≈20s)…",
            expanded=True,
        ) as status:
            st.markdown(PIPELINE_FLOW_MD)
            try:
                new_results["pipeline"] = pipeline_mod.run(paper=paper_text)
                _render_complete(status, new_results["pipeline"])
            except Exception as e:
                status.update(label=f"Pipeline failed: {e}", state="error")

    if run_orchestrator:
        with st.status(
            "Running orchestrator + specialists — multi-turn tool use (≈40s)…",
            expanded=True,
        ) as status:
            st.markdown(ORCHESTRATOR_FLOW_MD)
            try:
                new_results["orchestrator"] = orchestrator_mod.run(paper=paper_text)
                _render_complete(status, new_results["orchestrator"])
            except Exception as e:
                status.update(label=f"Orchestrator failed: {e}", state="error")

    if run_swarm:
        with st.status(
            "Running flat swarm — 9 turns of researcher / critic / editor on a "
            "shared scratchpad (≈100s, slow on purpose)…",
            expanded=True,
        ) as status:
            st.markdown(SWARM_FLOW_MD)
            try:
                new_results["swarm"] = swarm_mod.run(paper=paper_text)
                _render_complete(status, new_results["swarm"])
            except Exception as e:
                status.update(label=f"Swarm failed: {e}", state="error")

    # Drop the client reference so the next click rebuilds it fresh.
    llm._client = None

    if new_results:
        st.session_state["last_results"] = new_results
        st.toast("Done — comparison below.", icon="✅")


# ============================================================================
# Results
# ============================================================================
results = st.session_state.get("last_results")

if results:
    st.markdown("---")
    st.subheader("Comparison")

    rows = []
    for key, r in results.items():
        rows.append(
            {
                "Topology": r["topology"],
                "LLM calls": r["llm_calls"],
                "Cost (USD)": _fmt_money(r["total_cost_usd"]),
                "Input tokens": _fmt_int(r["total_input_tokens"]),
                "Output tokens": _fmt_int(r["total_output_tokens"]),
                "Latency": f"{r['total_latency_s']:.2f}s",
            }
        )
    st.table(rows)

    if "pipeline" in results and len(results) > 1:
        p = results["pipeline"]
        st.markdown("**Multipliers vs pipeline:**")
        for key in ("orchestrator", "swarm"):
            if key in results:
                r = results[key]
                cm = (
                    r["total_cost_usd"] / p["total_cost_usd"]
                    if p["total_cost_usd"]
                    else float("inf")
                )
                lm = (
                    r["total_latency_s"] / p["total_latency_s"]
                    if p["total_latency_s"]
                    else float("inf")
                )
                st.markdown(
                    f"- **{r['topology']}** — cost **{cm:.2f}×**, latency **{lm:.2f}×**"
                )

    st.markdown("---")
    st.subheader("Per-run breakdown")

    tab_labels = []
    tab_keys = []
    for key in ("pipeline", "orchestrator", "swarm"):
        if key in results:
            tab_labels.append(results[key]["topology"])
            tab_keys.append(key)

    tabs = st.tabs(tab_labels)
    for tab, key in zip(tabs, tab_keys):
        with tab:
            r = results[key]
            wc = len(r["summary"].split())

            # Header metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LLM calls", r["llm_calls"])
            m2.metric("Cost", _fmt_money(r["total_cost_usd"]))
            m3.metric("Latency", f"{r['total_latency_s']:.1f}s")
            m4.metric("Output words", wc)

            # Per-step breakdown
            st.markdown("**Step-by-step cost & tokens**")
            _render_steps_table(r.get("steps", []))

            # Flow context
            with st.expander("How this topology is structured"):
                st.markdown(FLOW_BY_KEY[key])

            # Final summary
            st.markdown("**Generated summary**")
            st.markdown(r["summary"])


# ============================================================================
# Idle state — when there are no results yet, show the framing + per-topology
# flows so users know what they're about to spend on
# ============================================================================
else:
    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("How it works")
        st.markdown(
            """
All three topologies solve the **same task** and use the **same model**
(Claude Sonnet 4.6). The shared task asks for a 300-word summary covering
claim, mechanism, evidence, limitations, and one open question.

**Pipeline** — fixed sequence, each step does one thing.

**Orchestrator + specialists** — one main agent decides which bounded subagent
to call via tool use.

**Flat swarm** — peer agents take turns on a shared scratchpad with no central
coordinator. Demos beautifully, ships almost nowhere.
"""
        )

    with col2:
        st.subheader("Typical results")
        st.table(
            [
                {"Topology": "Pipeline", "Calls": "3", "Cost": "$0.03", "Latency": "~20s"},
                {"Topology": "Orchestrator", "Calls": "5–8", "Cost": "$0.08", "Latency": "~40s"},
                {"Topology": "Flat swarm", "Calls": "9–15", "Cost": "$0.20", "Latency": "~100s"},
            ]
        )
        st.caption("Pricing: Claude Sonnet 4.6 at $3 / $15 per Mtok.")

    st.markdown("---")
    st.subheader("Flow of control for each topology")
    st.caption(
        "Each topology solves the same task differently. Open each section to "
        "see exactly which LLM calls happen, in what order, with what input — "
        "and why that drives the cost."
    )

    with st.expander("Pipeline — 3 LLM calls, linear", expanded=True):
        st.markdown(PIPELINE_FLOW_MD)

    with st.expander("Orchestrator + specialists — orchestrator loop with tool use"):
        st.markdown(ORCHESTRATOR_FLOW_MD)

    with st.expander("Flat swarm — round-robin on shared scratchpad"):
        st.markdown(SWARM_FLOW_MD)

    st.markdown("---")
    with st.expander("Decision tree — when should you build each one?"):
        st.code(
            """Does the workflow need an LLM at all?
  No  → Write a script. You don't need an agent.
  Yes ↓

Does the task structure vary based on input?
  No  → Pipeline. Stop here.
  Yes ↓

Can the variation be expressed as "which specialist to call next"?
  No  → You haven't decomposed the problem yet. Don't build a swarm.
  Yes ↓

How often does the routing decision actually fire per task?
  Less than 3 times → Pipeline with a branch is enough.
  More than 3 times → Orchestrator + specialists earns its keep.

Still tempted to build a swarm?
  Answer: "could a pipeline have done this?"
  If you cannot answer convincingly, the answer is yes.""",
            language="text",
        )


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption(
    f"BYOK · your API key stays in this browser session and is sent only to "
    f"`api.anthropic.com`. Source on [GitHub]({REPO_URL}). "
    f"Shared task: _{TASK_BRIEF[:140]}…_"
)
