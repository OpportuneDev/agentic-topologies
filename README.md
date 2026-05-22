# agentic-topologies

Three minimal implementations of the same task — summarize a paper in 300 words — built on three different agentic AI topologies. The point is to make the cost, latency, and output-quality differences observable rather than asserted.

| Topology | LLM calls | Cost (typical) | Latency (typical) | Output quality |
|---|---|---|---|---|
| Pipeline | 3 | low | low | tight, predictable |
| Orchestrator + specialists | 5–8 | medium | medium | good, slower |
| Flat swarm | 10–15 | high | high | mushier, repetitive |

Numbers in `results/` after running. Pricing assumed: Claude Sonnet 4.6 at $3 / $15 per million input / output tokens.

## The three topologies

**Pipeline** — a fixed sequence where each step does one thing. Predictable, debuggable, cheap to evaluate. Most production "agent" systems should be pipelines whether their builders call them that or not.

**Orchestrator + specialists** — one main agent holds the working context and decides which bounded subagent to call. Earns its complexity when the task structure genuinely varies at runtime.

**Flat swarm** — agents talk peer-to-peer with no central coordinator. Demos beautifully and ships almost nowhere. Debugging becomes a maze of conversation logs, cost compounds quickly, and the agents tend to talk past each other in ways that look thoughtful in the trace but produce mush in the output.

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python pipeline.py
python orchestrator.py
python swarm.py

# Or run all three and produce a comparison table:
python run_all.py
```

Results land in `results/*.json`. The console output for `run_all.py` includes the comparison table you'd put in a blog post or PR.

## Decision tree

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

## Files

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
└── results/                  # JSON outputs populated after running
```

No frameworks. Plain Python + the Anthropic SDK. The whole point of the post that motivated this repo is that abstraction debt is the failure mode; the code embodies that.

## What this is not

- Not a benchmark — sample size is one paper, run once. The numbers are illustrative.
- Not a framework — copy the patterns, don't import them.
- Not a complete tutorial on tool use — see the [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) for that.

## License

MIT. See `LICENSE`.
