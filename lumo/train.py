"""Background training loop for Lumo's tiny transformer.

Trains entirely in-process on the locally generated corpus. Emits step/loss
events that the web UI streams live over SSE.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .model import TinyGPT

# cosine learning-rate schedule helpers
def _lr_at(step: int, total: int, base: float = 6e-4, warmup: int = 20) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    import math

    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 * base + 0.9 * base * 0.5 * (1.0 + math.cos(math.pi * p))


class Trainer:
    """Runs training on a worker thread; subscribers get JSON-able events."""

    def __init__(self, model: TinyGPT, tokens: np.ndarray, ckpt_path: str,
                 on_saved=None, on_progress=None):
        self.model = model
        self.tokens = tokens
        self.ckpt_path = ckpt_path
        self.on_saved = on_saved
        self.on_progress = on_progress
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.subscribers: list[queue.Queue] = []
        self.state = {
            "running": False,
            "step": 0,
            "loss": None,
            "best_loss": None,
            "elapsed": 0.0,
            "tokens_per_s": 0.0,
            "mode": None,
            "max_steps": 0,
            "done": False,
            "message": "idle",
            "history": [],  # recent (step, loss) for late joiners
        }

    # ------------------------------ events -------------------------------- #

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self.lock:
            self.subscribers.append(q)
            snapshot = dict(self.state)
            history = list(self.state["history"])
        try:
            q.put_nowait({"type": "snapshot", "state": snapshot, "history": history})
        except queue.Full:
            pass
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _emit(self, payload: dict) -> None:
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    # ------------------------------ control ------------------------------- #

    @property
    def running(self) -> bool:
        return self.state["running"]

    def start(self, mode: str = "quick") -> dict:
        with self.lock:
            if self.state["running"]:
                return {"ok": False, "message": "training already running"}
            self.stop_flag.clear()
            # quick = fast first pass for the demo; full = longer run
            if mode == "full":
                max_steps, max_seconds = 2500, 1200.0
            elif mode == "extreme":
                max_steps, max_seconds = 6000, 3600.0
            else:
                max_steps, max_seconds = 300, 180.0
            self.state.update({
                "running": True, "step": 0, "loss": None, "best_loss": None,
                "elapsed": 0.0, "tokens_per_s": 0.0, "mode": mode,
                "max_steps": max_steps, "done": False,
                "message": f"training ({mode})",
                "history": [],
            })
        self.thread = threading.Thread(
            target=self._run, args=(max_steps, max_seconds), daemon=True)
        self.thread.start()
        return {"ok": True, "message": f"training started ({mode})"}

    def stop(self) -> dict:
        if self.state["running"]:
            self.stop_flag.set()
            return {"ok": True, "message": "stopping after current step"}
        return {"ok": False, "message": "not running"}

    # ------------------------------ loop ---------------------------------- #

    def _run(self, max_steps: int, max_seconds: float) -> None:
        rng = np.random.default_rng(1234)
        model = self.model
        tokens = self.tokens
        seq = model.cfg["seq_len"]
        batch = 16
        n = int(tokens.size)
        traj_starts = getattr(self, "starts", None)
        t0 = time.time()
        loss_ema = None
        step = 0
        try:
            while not self.stop_flag.is_set() and step < max_steps:
                if time.time() - t0 > max_seconds:
                    break
                # Half the windows are anchored at trajectory starts so the
                # model always sees question → action → observation chains
                # together; the rest are random windows over the packed stream.
                starts = np.empty(batch, dtype=np.int64)
                half = batch // 2
                if traj_starts is not None and len(traj_starts):
                    anchors = traj_starts[rng.integers(0, len(traj_starts), size=half)]
                    starts[:half] = np.minimum(anchors, max(0, n - seq - 2))
                starts[half:] = rng.integers(0, max(1, n - seq - 1), size=batch - half)
                idx = np.stack([tokens[s: s + seq] for s in starts]).astype(np.int32)
                tgt = np.stack([tokens[s + 1: s + seq + 1] for s in starts]).astype(np.int32)
                lr = _lr_at(step, max_steps)
                loss = model.train_step(idx, tgt, lr=lr)
                step += 1
                loss_ema = loss if loss_ema is None else 0.92 * loss_ema + 0.08 * loss
                elapsed = time.time() - t0
                tps = (step * batch * seq) / max(elapsed, 1e-6)

                with self.lock:
                    self.state.update({
                        "step": step, "loss": round(loss_ema, 4),
                        "best_loss": round(min(self.state["best_loss"] or 1e9, loss_ema), 4),
                        "elapsed": round(elapsed, 1),
                        "tokens_per_s": int(tps),
                    })
                    self.state["history"].append((step, round(loss_ema, 4)))
                    if len(self.state["history"]) > 2000:
                        self.state["history"] = self.state["history"][-2000:]
                self._emit({"type": "progress", "step": step,
                            "loss": round(loss_ema, 4),
                            "raw_loss": round(loss, 4),
                            "elapsed": round(elapsed, 1),
                            "tokens_per_s": int(tps), "lr": round(lr, 6)})
                if self.on_progress:
                    try:
                        self.on_progress(step, loss_ema)
                    except Exception:
                        pass

                # periodic checkpoint so progress is never lost
                if step % 50 == 0:
                    self._save(step, loss_ema)

            # final save
            self._save(step, loss_ema)
            msg = "training complete" if step >= max_steps else (
                "time limit reached" if self.stop_flag.is_set() is False and step > 0
                else "stopped")
            if self.stop_flag.is_set():
                msg = "stopped"
            with self.lock:
                self.state.update({"running": False, "done": True, "message": msg})
            self._emit({"type": "done", "step": step,
                        "loss": round(loss_ema, 4) if loss_ema is not None else None,
                        "message": msg})
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.state.update({"running": False, "done": True,
                                   "message": f"error: {e}"})
            self._emit({"type": "error", "message": str(e)})

    def _save(self, step: int, loss) -> None:
        try:
            self.model.save(self.ckpt_path, meta={
                "step": step,
                "loss": float(loss) if loss is not None else -1.0,
            })
            if self.on_saved:
                self.on_saved()
        except Exception:
            pass
