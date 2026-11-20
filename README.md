# Lumo ✦

**An AI agent built 100% from scratch — no provider APIs, no API keys, no cloud calls.**

Lumo is a complete, self-contained agent system that runs entirely in one process:

| Layer | What it is | Where |
|---|---|---|
| 🧠 Model | A ~485K-parameter GPT-style transformer (causal self-attention, GELU FFN, LayerNorm, tied embeddings) with **hand-written backprop + Adam** | `lumo/model.py` |
| 📚 Corpus | A locally generated training set: single- and multi-turn tool-use trajectories + factual passages (~266K tokens) | `lumo/corpus.py` |
| 🏋️ Trainer | Background training thread with cosine LR schedule, live loss telemetry, checkpointing | `lumo/train.py` |
| 🛠 Tools | `calculator` (safe AST eval), `time`, `remember`/`recall` (persistent JSON memory), `search` (TF-IDF knowledge base) | `lumo/tools.py` |
| 🤖 Agent | ReAct-style loop: *thought → action → observation → finish*. Neural planner with a deterministic rules fallback | `lumo/engine.py` |
| 🌐 Web UI | Chat with streamed traces, live loss chart, training controls — vanilla HTML/JS, no frameworks | `web/` |
| 🚀 Server | FastAPI + SSE endpoints | `lumo/server.py` |

## Why "no provider API" holds

- The model is trained **in-process on your CPU** from a corpus generated at startup.
- Inference is NumPy matmuls — no OpenAI / Anthropic / Gemini / Ollama / llama.cpp.
- The knowledge base, memory, and clock are local files and system time.
- There are zero outbound network calls in the codebase. Weights never leave the machine.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m lumo.server        # http://localhost:8000
```

On first boot the server **auto-starts a quick training run** (~1 minute, 300 steps).
Watch the loss curve drop in the sidebar, then chat. The engine badge on each reply
tells you whether the **neural** planner or the deterministic **rules** planner
answered — `auto` mode uses neural once trained and falls back to rules whenever
the model's output can't be parsed into valid tool calls.

## Training modes

- **quick** — 300 steps / ≤3 min: format + intent learning (default on boot)
- **full** — 2500 steps / ≤20 min: solid tool-calling competence
- **extreme** — 6000 steps / ≤60 min: push the loss further

Checkpoints save to `checkpoints/lumo.npz` and reload automatically on restart.
