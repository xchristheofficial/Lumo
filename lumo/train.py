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
                 on_saved=None, on_progress=None, tokenizer=None,
                 log_path: str | None = None, val: "object | None" = None):
        self.model = model
        self.tokens = tokens
        self.ckpt_path = ckpt_path
        self.on_saved = on_saved
        self.on_progress = on_progress
        self.tokenizer = tokenizer
        self.log_path = log_path
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.subscribers: list[queue.Queue] = []
        self.state = {
            "running": False,
            "step": 0,
            "loss": None,
            "val_loss": None,
            "best_loss": None,
            "elapsed": 0.0,
            "tokens_per_s": 0.0,
            "mode": None,
            "max_steps": 0,
            "done": False,
            "message": "idle",
            "history": [],      # recent (step, loss) for late joiners
            "val_history": [],  # recent (step, val_loss)
        }
        self.val_batches = self._build_val_batches(val)

    # ---------------------------- validation ------------------------------ #

    def _build_val_batches(self, val, k: int = 6):
        """Fixed held-out batches so val loss is comparable step to step."""
        if val is None:
            return None
        try:
            tokens = val.tokens
            seq = self.model.cfg["seq_len"]
            rng = np.random.default_rng(4242)
            n = int(tokens.size)
            if n <= seq + 2:
                return None
            anchors = getattr(val, "starts", None)
            rows, tgts = [], []
            half = k // 2
            if anchors is not None and len(anchors):
                valid_a = anchors[anchors <= n - seq - 2]
                picks = valid_a[rng.integers(0, len(valid_a), size=half)]
            else:
                picks = rng.integers(0, n - seq - 1, size=half)
            rand = rng.integers(0, n - seq - 1, size=k - half)
            for s in list(picks) + list(rand):
                s = int(s)
                rows.append(tokens[s: s + seq])
                tgts.append(tokens[s + 1: s + seq + 1])
            return (np.stack(rows).astype(np.int32),
                    np.stack(tgts).astype(np.int32))
        except Exception:  # noqa: BLE001
            return None

    def _val_loss(self) -> float | None:
        if self.val_batches is None:
            return None
        idx, tgt = self.val_batches
        _, loss, _ = self.model.forward(idx, tgt)
        return round(float(loss), 4)

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
            # quick = fast first pass; full/extreme = longer runs;
            # auto = autopilot — trains forever until you stop it
            if mode == "auto":
                max_steps, max_seconds = 10**9, float("inf")
            elif mode == "full":
                max_steps, max_seconds = 2500, 1200.0
            elif mode == "extreme":
                max_steps, max_seconds = 6000, 3600.0
            else:
                max_steps, max_seconds = 300, 180.0
            self.state.update({
                "running": True, "step": 0, "loss": None, "best_loss": None,
                "elapsed": 0.0, "tokens_per_s": 0.0, "mode": mode,
                "max_steps": max_steps, "done": False,
                "message": ("autopilot training" if mode == "auto"
                            else f"training ({mode})"),
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

    # --------------------------- experience pool --------------------------- #

    def _refresh_experiences(self, force: bool = False) -> None:
        """(Re)encode recorded conversations for experience replay."""
        if self.tokenizer is None:
            return
        try:
            from . import experience

            path = experience.EXPERIENCE_PATH
            mtime = path.stat().st_mtime if path.exists() else 0.0
            if not force and getattr(self, "_exp_mtime", None) == mtime:
                return
            self._exp_mtime = mtime
            entries = experience.load_entries()
            toks: list[int] = []
            starts: list[int] = []
            off = 0
            for e in entries:
                block = f"user : {e['q']}\nlumo : {e['a']}"
                ids = self.tokenizer.encode(block)
                if len(ids) < 8:
                    continue
                starts.append(off)
                toks.extend(ids)
                off += len(ids)
            self.exp_tokens = np.asarray(toks, dtype=np.int32) if toks else None
            self.exp_starts = np.asarray(starts, dtype=np.int64) if starts else None
        except Exception:  # noqa: BLE001
            self.exp_tokens = None
            self.exp_starts = None

    def _log(self, line: str) -> None:
        if not self.log_path:
            return
        try:
            from pathlib import Path

            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
        except OSError:
            pass

    # -------------------------------- run --------------------------------- #

    def _run(self, max_steps: int, max_seconds: float) -> None:
        rng = np.random.default_rng(1234)
        model = self.model
        tokens = self.tokens
        seq = model.cfg["seq_len"]
        batch = 16
        n = int(tokens.size)
        traj_starts = getattr(self, "starts", None)
        mode = self.state.get("mode") or "quick"
        lr_total = 5000 if mode == "auto" else max_steps
        self._refresh_experiences(force=True)
        self._log(f"training start mode={mode} max_steps={max_steps} "
                  f"params={model.num_params():,} corpus_tokens={n:,}")
        t0 = time.time()
        loss_ema = None
        step = 0
        try:
            while not self.stop_flag.is_set() and step < max_steps:
                if time.time() - t0 > max_seconds:
                    break
                if step % 200 == 0:
                    self._refresh_experiences()

                # Experience replay: up to ¼ of the batch comes from the
                # agent's own recorded conversations.
                n_exp = 0
                exp_tokens = getattr(self, "exp_tokens", None)
                exp_starts = getattr(self, "exp_starts", None)
                if exp_tokens is not None and exp_starts is not None:
                    valid = exp_starts[exp_starts <= len(exp_tokens) - seq - 2]
                    if len(valid):
                        n_exp = min(batch // 4, len(valid))
                n_cor = batch - n_exp

                # Corpus windows: half anchored at trajectory starts so
                # question → action → observation chains train together.
                starts = np.empty(n_cor, dtype=np.int64)
                half = n_cor // 2
                if traj_starts is not None and len(traj_starts) and half:
                    anchors = traj_starts[rng.integers(0, len(traj_starts), size=half)]
                    starts[:half] = np.minimum(anchors, max(0, n - seq - 2))
                if n_cor - half:
                    starts[half:] = rng.integers(0, max(1, n - seq - 1),
                                                 size=n_cor - half)
                rows = [tokens[s: s + seq] for s in starts]

                if n_exp:
                    valid = exp_starts[exp_starts <= len(exp_tokens) - seq - 2]
                    e_starts = valid[rng.integers(0, len(valid), size=n_exp)]
                    rows += [exp_tokens[s: s + seq] for s in e_starts]

                idx = np.stack(rows).astype(np.int32)
                tgt = np.stack([tokens[int(s) + 1: int(s) + seq + 1]
                                for s in starts[:n_cor]] +
                               [exp_tokens[int(s) + 1: int(s) + seq + 1]
                                for s in (e_starts if n_exp else [])]).astype(np.int32)
                lr = _lr_at(step % lr_total, lr_total)
                loss = model.train_step(idx, tgt, lr=lr)
                step += 1
                loss_ema = loss if loss_ema is None else 0.92 * loss_ema + 0.08 * loss
                elapsed = time.time() - t0
                tps = (step * batch * seq) / max(elapsed, 1e-6)

                # held-out validation loss every 100 steps (and at the end)
                val_l = None
                if step % 100 == 0:
                    val_l = self._val_loss()
                    if val_l is not None:
                        with self.lock:
                            self.state["val_loss"] = val_l
                            self.state["val_history"].append((step, val_l))
                            if len(self.state["val_history"]) > 500:
                                self.state["val_history"] = \
                                    self.state["val_history"][-500:]

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
                            "val_loss": val_l,
                            "raw_loss": round(loss, 4),
                            "elapsed": round(elapsed, 1),
                            "tokens_per_s": int(tps), "lr": round(lr, 6)})
                if self.on_progress:
                    try:
                        self.on_progress(step, loss_ema)
                    except Exception:
                        pass

                # periodic checkpoint + log so progress is never lost
                if step % 50 == 0:
                    self._save(step, loss_ema)
                    n_seen = getattr(self, "exp_starts", None)
                    n_seen = 0 if n_seen is None else len(n_seen)
                    vtxt = (f" val={val_l}" if val_l is not None else "")
                    self._log(f"step={step} loss={loss_ema:.4f}{vtxt} "
                              f"raw={loss:.4f} tps={int(tps)} "
                              f"experiences={n_seen}")

            # final save
            self._save(step, loss_ema)
            msg = "training complete" if step >= max_steps else (
                "time limit reached" if self.stop_flag.is_set() is False and step > 0
                else "stopped")
            if self.stop_flag.is_set():
                msg = "stopped"
            self._log(f"training stop mode={mode} step={step} "
                      f"loss={loss_ema if loss_ema is not None else float('nan'):.4f} "
                      f"reason={msg}")
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
            lifetime = int(getattr(self, "base_step", 0)) + step
            self.model.save(self.ckpt_path, meta={
                "step": lifetime,
                "loss": float(loss) if loss is not None else -1.0,
            })
            if self.on_saved:
                self.on_saved()
        except Exception:
            pass
