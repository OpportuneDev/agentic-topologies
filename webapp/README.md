# webapp — Streamlit BYOK demo

A bring-your-own-key web UI for the three agentic topologies. Users paste an Anthropic API key, pick which topologies to run against a paper, and get a side-by-side comparison of cost, latency, and output quality.

## Why BYOK

A public demo with a server-held API key gets abused within hours. BYOK shifts the cost and rate limits onto each visitor: you pay zero per request, visitors pay for their own usage at their own console pricing. The key lives in Streamlit session state for the duration of the browser tab — never persisted, never logged.

## Run locally

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r webapp/requirements.txt

streamlit run webapp/app.py
```

A browser tab opens at `http://localhost:8501`. Paste your key, pick topologies, hit Run.

## Deploy to Streamlit Cloud (free)

1. Make sure the repo is on GitHub (it is — `OpportuneDev/agentic-topologies`).
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with the GitHub account that has access.
3. **New app** → pick this repo, branch `main`, **Main file path:** `webapp/app.py`.
4. Click **Deploy**. Streamlit Cloud detects `webapp/requirements.txt` and installs deps. First boot takes ~2 minutes; subsequent loads are instant.
5. The app gets a public URL like `https://opportunedev-agentic-topologies.streamlit.app`. Paste it into the LinkedIn post.

No environment variables to configure — the key is BYOK per visitor.

## Architecture

`webapp/app.py` imports the three topology modules (`pipeline.py`, `orchestrator.py`, `swarm.py`) directly from the repo root. Each topology's `run()` accepts an optional `paper: str` argument, which the webapp uses to inject either the bundled sample paper or text pasted by the visitor — no monkey-patching, no temp files.

The API key is set via `os.environ["ANTHROPIC_API_KEY"]` and the cached Anthropic client in `shared/llm.py` is reset on each Run click so the new key takes effect immediately.

## Notes

- The swarm topology is intentionally slow (~100s). It is off by default in the UI so first-time visitors don't burn $0.20 and 2 minutes unintentionally.
- Streamlit reruns the script on every interaction. Results are persisted via `st.session_state["last_results"]` so they survive checkbox toggles.
- No telemetry, no logging of inputs or outputs. The only network call is to `api.anthropic.com`.
