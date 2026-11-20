# Lumo ✦

**An AI agent built 100% from scratch — no provider APIs, no API keys, no cloud calls.**
It **trains itself continuously (autopilot)** and learns from your conversations
even when you're not there.

| Layer | What it is | Where |
|---|---|---|
| 🧠 Model | A ~560K-parameter GPT-style transformer (causal self-attention, GELU FFN, LayerNorm, tied embeddings) with **hand-written backprop + Adam** | `lumo/model.py` |
| 📚 Corpus | Locally generated training set: single/multi-turn tool trajectories, unit conversions, factual passages (~327K tokens, 121 passages) | `lumo/corpus.py` |
| 🏋️ Trainer | Background training with cosine LR, live loss telemetry, checkpoints — plus **autopilot mode that trains forever** | `lumo/train.py` |
| 📝 Experience replay | Every chat you have is written to `data/experiences.jsonl` and mixed into future training batches | `lumo/experience.py` |
| 🛠 Tools | `calculator` (safe AST eval + unit conversions), `time`, `remember`/`recall` (persistent JSON memory), `search` (TF-IDF knowledge base) | `lumo/tools.py` |
| 🤖 Agent | ReAct loop: *thought → action → observation → finish*; neural planner with deterministic rules fallback and result-reflection quality gates | `lumo/engine.py` |
| 🌐 Web UI | Chat with streamed traces, live loss chart, **autopilot toggle**, memory panel — vanilla HTML/JS | `web/` |
| 🚀 Server | FastAPI + SSE endpoints | `lumo/server.py` |

## Why "no provider API" holds

- The model is trained **in-process on your CPU** from a corpus generated at startup.
- Inference is NumPy matmuls — no OpenAI / Anthropic / Gemini / Ollama / llama.cpp.
- The knowledge base, memory, and clock are local files and system time.
- Zero outbound network calls in the codebase. Weights never leave the machine.

## How to run it

```bash
# one-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# start the website — autopilot training begins immediately (ON by default)
.venv/bin/python -m lumo.server
# → open http://localhost:8000
```

**That's it.** Leave the server running and it trains itself in the background —
close the browser tab, walk away, come back tomorrow: the loss chart keeps
dropping and checkpoints keep saving to `checkpoints/lumo.npz` (reloaded on restart).

### Train with NO website running (offline / overnight)

```bash
.venv/bin/python -m lumo.selftrain            # train forever, logs to data/train.log
.venv/bin/python -m lumo.selftrain --steps 5000   # bounded run
.venv/bin/python -m lumo.selftrain --status   # checkpoint + experience summary
```

## How to test it

1. **Watch it learn (no interaction needed)**
   - Open the site → the sidebar loss chart streams live; the pill shows
     `autopilot: step N · loss L` and the progress bar pulses.
   - Or from a terminal: `tail -f data/train.log`
   - Or: `curl -s localhost:8000/api/status | python3 -m json.tool` →
     `training.running == true`, `training.loss` falling, `autopilot.enabled == true`

2. **Chat now vs. later** (engine badge on each reply)
   - First minutes: replies run on `rules` (deterministic) while the model warms up.
   - After ~250 steps (~1–2 min) the `neural` planner engages; quality gates
     keep answers honest — failures visibly fall back to `rules`.
   - Try: `what is 17 * 23 ?` · `convert 5 km to miles` ·
     `remember that my favorite color is teal` · `what is my favorite color ?` ·
     `tell me about black holes` · `what time is it ?`

3. **Experience replay** — chat a few times, then check
   - `experiences recorded: N` in the training panel, and
   - `data/experiences.jsonl` (each line is a training block the autopilot
     replays — the model literally studies *your* questions while you're away).

4. **Autopilot toggle** — hit `🤖 autopilot` in the training panel to stop/start
   continuous training; the choice persists across restarts
   (`data/autopilot.json`).

## Training modes

- **autopilot** — trains forever (default on boot), 5000-step cosine cycles,
  checkpoints every 50 steps, mixes in your recorded conversations
- **quick** — 300 steps / ≤3 min burst
- **full** — 2500 steps / ≤20 min
- **extreme** — 6000 steps / ≤60 min
