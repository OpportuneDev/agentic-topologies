# results/

Empty until you run the scripts. After running:

```bash
python run_all.py
```

You'll see:

- `pipeline.json` — output and metrics from `01_pipeline.py`
- `orchestrator.json` — output and metrics from `02_orchestrator.py`
- `swarm.json` — output and metrics from `03_swarm.py`

Each JSON contains the produced summary plus token counts, cost in USD, latency in seconds, and the number of LLM calls. The comparison table printed at the end of `run_all.py` is the bit worth screenshotting for a blog post.
