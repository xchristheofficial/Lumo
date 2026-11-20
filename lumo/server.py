"""FastAPI server for Lumo — serves the web UI and the agent/training APIs.

Everything runs locally: no provider API keys, no outbound model calls.
"""
from __future__ import annotations

import asyncio
import json
import queue
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .corpus import PASSAGES, build_bundle
from .engine import build_agent
from .model import TinyGPT
from .tools import MEMORY_PATH, TOOLS, init_knowledge

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
CKPT_DIR = ROOT / "checkpoints"
CKPT_PATH = str(CKPT_DIR / "lumo.npz")

# --------------------------------------------------------------------------- #
# App state
# --------------------------------------------------------------------------- #

app = FastAPI(title="Lumo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state: dict = {}


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ChatIn(BaseModel):
    message: str
    engine: str = "auto"  # auto | neural | rules


class TrainIn(BaseModel):
    mode: str = "quick"  # quick | full | extreme


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #

@app.on_event("startup")
def _startup() -> None:
    from .train import Trainer

    t0 = time.time()
    bundle = build_bundle()
    init_knowledge(PASSAGES)

    model: TinyGPT | None = None
    ckpt_meta = None
    ckpt = Path(CKPT_PATH)
    if ckpt.exists():
        try:
            import numpy as np

            z = np.load(CKPT_PATH, allow_pickle=False)
            cfg = {k[4:]: int(z[k]) for k in z.files if k.startswith("cfg_")}
            model = TinyGPT(**cfg)
            for k in z.files:
                if k.startswith("p_"):
                    model.P[k[2:]] = z[k].astype(np.float32)
            ckpt_meta = {k[5:]: z[k].item() if z[k].ndim == 0 else str(z[k])
                         for k in z.files if k.startswith("meta_")}
        except Exception as e:  # noqa: BLE001
            print(f"[lumo] failed to load checkpoint: {e}")
            model = None

    agent = build_agent(model, bundle.tokenizer)
    trainer = Trainer(model if model is not None else TinyGPT(bundle.tokenizer.vocab_size),
                      bundle.tokens, CKPT_PATH)
    # if we had no checkpoint, the trainer owns the fresh model → hand it to the agent
    if model is None:
        agent.reload_model(trainer.model)

    # Engage the neural planner only after real training progress.
    if ckpt_meta and int(ckpt_meta.get("step", 0)) >= agent.neural.min_steps:
        agent.neural.trained = True

    def _on_progress(step: int, _loss: float) -> None:
        if step >= agent.neural.min_steps:
            agent.neural.trained = True

    def _on_saved() -> None:
        state["ckpt_meta"] = {"step": trainer.state.get("step", 0)}

    trainer.on_progress = _on_progress
    trainer.on_saved = _on_saved
    trainer.starts = bundle.starts  # trajectory-start anchors for sampling

    state.update({
        "bundle": bundle,
        "trainer": trainer,
        "agent": agent,
        "model": trainer.model,
        "started": time.time(),
        "ckpt_meta": ckpt_meta,
        "boot_ms": int((time.time() - t0) * 1000),
    })

    # Train right away on first boot so the user sees learning ASAP.
    if ckpt_meta is None:
        trainer.start("quick")
        print("[lumo] no checkpoint found — auto-started quick training")
    print(f"[lumo] ready in {state['boot_ms']} ms "
          f"(vocab={bundle.tokenizer.vocab_size}, tokens={bundle.tokens.size:,})")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status():
    trainer = state["trainer"]
    bundle = state["bundle"]
    model = state["model"]
    memory = {}
    if MEMORY_PATH.exists():
        try:
            memory = json.loads(MEMORY_PATH.read_text())
        except Exception:  # noqa: BLE001
            memory = {}
    return {
        "app": "lumo",
        "version": "0.1.0",
        "model": {
            "ready": True,
            "neural_ready": state["agent"].neural.ready,
            "params": model.num_params(),
            "vocab": bundle.tokenizer.vocab_size,
            "seq_len": model.cfg["seq_len"],
            "layers": model.cfg["n_layer"],
            "heads": model.cfg["n_head"],
            "d_model": model.cfg["d_model"],
            "corpus_tokens": int(bundle.tokens.size),
            "checkpoint": state.get("ckpt_meta"),
        },
        "training": dict(trainer.state),
        "tools": [
            {"name": n, "desc": t["desc"], "emoji": t["emoji"]}
            for n, t in TOOLS.items()
        ],
        "knowledge_passages": len(PASSAGES),
        "memory": memory,
        "boot_ms": state.get("boot_ms"),
        "provider_apis": 0,
    }


@app.post("/api/reset")
def reset():
    state["agent"].history = []
    return {"ok": True, "message": "conversation cleared"}


class ResetMem(BaseModel):
    key: str | None = None


@app.post("/api/memory/forget")
def forget(req: ResetMem):
    if not MEMORY_PATH.exists():
        return {"ok": True, "memory": {}}
    try:
        mem = json.loads(MEMORY_PATH.read_text())
        if req.key:
            mem.pop(req.key, None)
        else:
            mem = {}
        MEMORY_PATH.write_text(json.dumps(mem, indent=2, sort_keys=True))
        return {"ok": True, "memory": mem}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.post("/api/chat")
def chat(req: ChatIn):
    agent = state["agent"]
    msg = (req.message or "").strip()
    if not msg:
        async def empty():
            yield sse({"type": "done", "answer": "", "engine": "none"})

        return StreamingResponse(empty(), media_type="text/event-stream")

    def gen():
        try:
            for ev in agent.respond(msg, engine_pref=req.engine):
                yield sse(ev)
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "message": str(e)})
            yield sse({"type": "done", "answer": None, "engine": "error"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/train")
def train(req: TrainIn):
    mode = req.mode if req.mode in ("quick", "full", "extreme") else "quick"
    return state["trainer"].start(mode)


@app.post("/api/train/stop")
def train_stop():
    return state["trainer"].stop()


@app.get("/api/train/events")
async def train_events(request: Request):
    trainer = state["trainer"]
    q = trainer.subscribe()

    def gen():
        try:
            while True:
                try:
                    payload = q.get(timeout=15.0)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                yield sse(payload)
                if payload.get("type") in ("done", "error"):
                    # keep the stream open for the next run; snapshot arrives on demand
                    pass
        finally:
            trainer.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# static assets
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def main():
    import os

    import uvicorn

    port = int(os.environ.get("LUMO_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
