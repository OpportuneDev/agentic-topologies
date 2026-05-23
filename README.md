# agentic-topologies

Three minimal implementations of the **same task** — summarize a paper in 300 words — built on three different agentic AI topologies. The point is to make the cost, latency, and output-quality differences **observable rather than asserted**.

| Topology | LLM calls | Cost (typical) | Latency (typical) | Output quality |
|---|---|---|---|---|
| Pipeline | 3 | low | low | tight, predictable |
| Orchestrator + specialists | 5–8 | medium | medium | good, slower |
| Flat swarm | 9–15 | high | high | mushier, repetitive |

Numbers land in `results/` after running. Pricing assumed: **Claude Sonnet 4.6 at $3 / $15 per million input / output tokens**.

No frameworks. Plain Python + the Anthropic SDK. The whole point of the post that motivated this repo is that abstraction debt is the failure mode; the code embodies that.

---

## Try it interactively

There's a **bring-your-own-key Streamlit demo** in `webapp/`. Paste your Anthropic key, choose which topologies to run against any paper, and watch the comparison table populate live.

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

Deploy your own copy to [Streamlit Cloud](https://share.streamlit.io/) for free — point it at this repo, main file `webapp/app.py`, done. Full deploy guide in [`webapp/README.md`](webapp/README.md).

---

## The three topologies

**Pipeline** — a fixed sequence where each step does one thing. Predictable, debuggable, cheap to evaluate. Most production "agent" systems should be pipelines whether their builders call them that or not.

**Orchestrator + specialists** — one main agent holds the working context and decides which bounded subagent to call. Routed via Anthropic tool use. Earns its complexity when the task structure genuinely varies at runtime.

**Flat swarm** — agents talk peer-to-peer on a shared scratchpad with no central coordinator. Demos beautifully and ships almost nowhere. Debugging becomes a maze of conversation logs, cost compounds because every agent sees the full history, and the agents tend to talk past each other in ways that look thoughtful in the trace but produce mush in the output.

---

## Quickstart

```bash
git clone https://github.com/OpportuneDev/agentic-topologies.git
cd agentic-topologies

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-... # get one at https://console.anthropic.com/

# Run a single topology:
python pipeline.py
python orchestrator.py
python swarm.py

# Or run all three and produce the comparison table:
python run_all.py
```

Each script writes a JSON file under `results/` containing the produced summary plus token counts, cost in USD, latency in seconds, and the number of LLM calls. The comparison table printed at the end of `run_all.py` is the bit worth screenshotting.

A full run of `run_all.py` typically costs **under $0.10** at current Sonnet 4.6 pricing.

---

## Sample output

Running `python run_all.py` produces something like this (your numbers will vary):

```
================================================================================
Comparison
================================================================================
Topology                     | Calls | Cost USD |  In tok | Out tok | Latency
--------------------------------------------------------------------------------
pipeline                     |     3 | $ 0.0142 |    4123 |     612 |   8.91s
orchestrator+specialists     |     7 | $ 0.0381 |   12044 |    1188 |  19.42s
flat_swarm                   |     9 | $ 0.1207 |   38291 |    2104 |  41.07s

Multipliers vs pipeline:
  orchestrator+specialists     cost 2.68x   latency 2.18x
  flat_swarm                   cost 8.50x   latency 4.61x
```

That's the empirical claim: for a task where structure is fixed, swarms cost an order of magnitude more for output that is at best comparable.

---

## Testing

The test suite mocks every LLM call, so it runs in under a second and **does not require an API key**.

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover:

- **`shared/`** — cost calculation, paper loading, task brief invariants.
- **`pipeline.py`** — three calls in order, step 3 receives the outputs of steps 1 and 2, correct accounting.
- **`orchestrator.py`** — tool dispatch, multi-turn tool-use loop, `max_turns` cap, `end_turn` short-circuit, unknown-tool error.
- **`swarm.py`** — round-robin order, `FINAL_SUMMARY:` early termination, fallback to last editor turn, monotonic scratchpad growth.
- **`run_all.py`** — comparison output, JSON files written.

If you change a topology and the corresponding test still passes, you have not broken the contract. If you add a topology, add tests in the same style.

---

## Decision tree

Use this to decide what you actually need before reaching for a framework.

```
Does the workflow need an LLM at all?
  No  → Write a script. You don't need an agent.
  Yes ↓

Does the task structure vary based on input?
  No  → Pipeline. Stop here.
  Yes ↓

Can the variation be expressed as "which specialist to call next"?
  No  → You haven't decomposed the problem yet. Don't build a swarm to cover for this.
  Yes ↓

How often does the routing decision actually fire per task?
  Less than 3 times → Pipeline with a branch is enough.
  More than 3 times → Orchestrator + specialists earns its keep.

Still tempted to build a swarm?
  Answer: "could a pipeline have done this?"
  If you cannot answer convincingly, the answer is yes.
```

---

## Project layout

```
agentic-topologies/
├── pipeline.py               # linear sequence, 3 LLM calls
├── orchestrator.py           # one orchestrator + bounded specialists via tool use
├── swarm.py                  # researcher / critic / editor talking peer-to-peer
├── run_all.py                # runs all three, writes results/, prints comparison
├── sample_paper.md           # synthetic ~1500-word research note used as input
├── shared/
│   ├── llm.py                # thin Anthropic SDK wrapper with cost / latency tracking
│   └── task.py               # task description + paper loader
├── tests/                    # mocked unit tests — no API key required
│   ├── conftest.py
│   ├── test_shared.py
│   ├── test_pipeline.py
│   ├── test_orchestrator.py
│   ├── test_swarm.py
│   └── test_run_all.py
├── webapp/                   # Streamlit BYOK demo (deploy to Streamlit Cloud)
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── results/                  # JSON outputs populated after running
├── requirements.txt          # runtime: anthropic
├── requirements-dev.txt      # runtime + pytest
└── pyproject.toml            # pytest config
```

---

## Extending

**Swap the paper.** Drop any text into `sample_paper.md`, or change `DEFAULT_PAPER_PATH` in `shared/task.py`. `load_paper()` accepts a custom path too.

**Swap the task.** Edit `TASK_BRIEF` in `shared/task.py`. All three topologies read it, so the comparison stays apples-to-apples.

**Swap the model.** Change `DEFAULT_MODEL` and the price constants in `shared/llm.py`. The cost calculation will follow.

**Add a topology.** Create `mything.py` exposing a `run() -> dict` that returns the same shape (`topology`, `summary`, `llm_calls`, `total_cost_usd`, `total_input_tokens`, `total_output_tokens`, `total_latency_s`). Wire it into `run_all.py`. Add a test under `tests/test_mything.py` following the pattern in `tests/test_pipeline.py` — mock `mything.call` and assert behavior.

---

## Troubleshooting

**`AuthenticationError: invalid x-api-key`** — `ANTHROPIC_API_KEY` is not set or is wrong. Grab one from [console.anthropic.com](https://console.anthropic.com/) and `export ANTHROPIC_API_KEY=sk-ant-...`.

**`ModuleNotFoundError: No module named 'anthropic'`** — you forgot `pip install -r requirements.txt`, or you are not running inside the venv you installed it into.

**Tests can't find `pipeline` / `orchestrator` / `swarm`** — run `pytest` from the repo root. `pyproject.toml` sets `pythonpath = ["."]` so imports resolve.

**Orchestrator never calls `finish`** — the loop is capped at 12 turns by default (`run(max_turns=12)`). If the model genuinely can't terminate, your task brief is probably underspecified. The orchestrator behaves best when the task has an obvious "I'm done" condition.

**Swarm prints nothing for ages** — each agent sees the full scratchpad, which grows linearly per turn. By turn 9 a single call processes thousands of tokens. This is the *point* of the demo, not a bug.

**SDK version mismatch** — the orchestrator's tool-use loop assumes the Anthropic SDK API as of v0.40+. If you're on an older version, upgrade: `pip install -U anthropic`.

---

## What this is not

- **Not a benchmark.** Sample size is one paper, run once. The numbers are illustrative; rerun yourself if you want to argue with them.
- **Not a framework.** Copy the patterns, don't import them.
- **Not a complete tutorial on tool use.** See the [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) for that.

---

## License

MIT. See `LICENSE`.
