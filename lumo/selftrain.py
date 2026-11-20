"""Lumo autopilot — continuous self-training with no web server required.

Leave this running overnight (or in any background process) and the model
keeps learning: synthetic trajectories + every conversation you had with
the web UI (experience replay) + checkpointing to checkpoints/lumo.npz.

Usage:
    .venv/bin/python -m lumo.selftrain            # train forever
    .venv/bin/python -m lumo.selftrain --steps 5000   # bounded run
    .venv/bin/python -m lumo.selftrain --status   # show state and exit
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from . import experience
from .corpus import build_bundle
from .model import TinyGPT

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = str(ROOT / "checkpoints" / "lumo.npz")
LOG_PATH = str(ROOT / "data" / "train.log")


def main() -> int:
    ap = argparse.ArgumentParser(description="Lumo autonomous trainer")
    ap.add_argument("--steps", type=int, default=0,
                    help="stop after N steps (0 = run forever)")
    ap.add_argument("--status", action="store_true",
                    help="print checkpoint/experience state and exit")
    args = ap.parse_args()

    if args.status:
        z = Path(CKPT_PATH)
        if z.exists():
            import numpy as np

            data = np.load(CKPT_PATH, allow_pickle=False)
            meta = {k[5:]: data[k].item() for k in data.files
                    if k.startswith("meta_") and data[k].ndim == 0}
            print(f"checkpoint: {CKPT_PATH} step={meta.get('step')} "
                  f"loss={meta.get('loss')}")
        else:
            print("checkpoint: none yet (first run will create one)")
        print(f"experiences recorded: {experience.count()}")
        print(f"training log: {LOG_PATH}")
        return 0

    print("[lumo:selftrain] building corpus + model …")
    bundle = build_bundle()
    model: TinyGPT | None = None
    if Path(CKPT_PATH).exists():
        try:
            import numpy as np

            z = np.load(CKPT_PATH, allow_pickle=False)
            cfg = {k[4:]: int(z[k]) for k in z.files if k.startswith("cfg_")}
            if cfg.get("vocab_size") == bundle.tokenizer.vocab_size:
                model = TinyGPT(**cfg)
                for k in z.files:
                    if k.startswith("p_"):
                        model.P[k[2:]] = z[k].astype(np.float32)
                print(f"[lumo:selftrain] resumed checkpoint ({cfg})")
            else:
                print("[lumo:selftrain] checkpoint vocab mismatch — fresh model")
        except Exception as e:  # noqa: BLE001
            print(f"[lumo:selftrain] checkpoint load failed: {e}")
    if model is None:
        model = TinyGPT(bundle.tokenizer.vocab_size)

    from .train import Trainer

    trainer = Trainer(model, bundle.tokens, CKPT_PATH,
                      tokenizer=bundle.tokenizer, log_path=LOG_PATH)
    trainer.starts = bundle.starts
    if Path(CKPT_PATH).exists() and model is not None:
        try:
            import numpy as np

            z = np.load(CKPT_PATH, allow_pickle=False)
            meta = {k[5:]: z[k].item() for k in z.files
                    if k.startswith("meta_") and z[k].ndim == 0}
            trainer.base_step = int(meta.get("step", 0))
        except Exception:  # noqa: BLE001
            trainer.base_step = 0

    # per-step console feedback (SSE is not available here)
    def _progress(step: int, loss: float) -> None:
        if step % 25 == 0:
            print(f"  step {step:>6}  loss {loss:.4f}  "
                  f"experiences {experience.count()}", flush=True)
    trainer.on_progress = _progress

    mode = "auto" if args.steps <= 0 else "full"
    if args.steps > 0:
        # bounded: reuse full cadence but cap steps via start override
        result = trainer.start("full")
        trainer.state["max_steps"] = args.steps
    else:
        result = trainer.start("auto")
    print(f"[lumo:selftrain] {result['message']} — Ctrl+C to stop")

    def _handle_sig(_sig, _frm):
        print("\n[lumo:selftrain] stopping …")
        trainer.stop()
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    # keep the main thread alive until training stops
    while trainer.state["running"]:
        time.sleep(1.0)
        if args.steps and trainer.state["step"] >= args.steps:
            trainer.stop()
    print(f"[lumo:selftrain] done — step={trainer.state['step']} "
          f"loss={trainer.state['loss']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
