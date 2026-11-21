"""Lumo evaluation harness — a scripted test battery with a scored report.

Runs ~30 checks (math, conversions, memory, knowledge, time, chat) through
the real Agent in `auto` mode and prints accuracy per category, which
engine answered, and latency. Read-only: it does not record experiences
and never writes checkpoints.

Usage:
    .venv/bin/python -m lumo.evaluate             # human-readable report
    .venv/bin/python -m lumo.evaluate --json      # machine-readable
    .venv/bin/python -m lumo.evaluate --min-acc 0.9
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from .corpus import PASSAGES, build_bundle
from .engine import build_agent
from .model import TinyGPT
from .tools import TOOLS, init_knowledge

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = str(ROOT / "checkpoints" / "lumo.npz")


def _contains(*subs: str):
    def pred(ans: str) -> bool:
        low = ans.lower()
        return all(s.lower() in low for s in subs)
    return pred


def _matches(rx: str):
    pat = re.compile(rx, re.IGNORECASE)

    def pred(ans: str) -> bool:
        return bool(pat.search(ans))
    return pred


# (category, question, predicate)
BATTERY: list[tuple[str, str, object]] = [
    # math
    ("math", "what is 17 * 23 ?", _contains("391")),
    ("math", "what is 12 plus 34 ?", _contains("46")),
    ("math", "what is 144 / 12 ?", _contains("12")),
    ("math", "what is 9 * 8 ?", _contains("72")),
    ("math", "what is 250 - 75 ?", _contains("175")),
    ("math", "what is 15 percent of 200 ?", _contains("30")),
    ("math", "compute 7 * 6", _contains("42")),
    ("math", "how much is 1000 / 8 ?", _contains("125")),
    # conversions
    ("convert", "convert 100 celsius to fahrenheit", _contains("212")),
    ("convert", "convert 5 km to miles", _contains("miles", "3.106")),
    ("convert", "how many miles in 10 km ?", _contains("miles", "6.213")),
    ("convert", "12 pounds to kg", _contains("kg", "5.443")),
    ("convert", "convert 2 meters to feet", _contains("feet", "6.561")),
    # time
    ("time", "what time is it ?", _contains("utc")),
    ("time", "what is today's date ?", _matches(r"\d{4}")),
    # knowledge (from the local passages)
    ("knowledge", "tell me about black holes", _contains("event horizon")),
    ("knowledge", "tell me about giraffes", _contains("vertebrae")),
    ("knowledge", "what is the capital of france", _contains("paris")),
    ("knowledge", "how does photosynthesis work", _contains("oxygen")),
    ("knowledge", "tell me about the nile river", _contains("mediterranean")),
    ("knowledge", "who was leonardo da vinci", _contains("michelangelo")),
    ("knowledge", "what is binary code", _contains("zero")),
    # small talk (rules-first by design)
    ("chat", "hello", _contains("lumo")),
    ("chat", "who are you ?", _contains("lumo")),
    ("chat", "what can you do ?", _contains("calculate")),
    ("chat", "thanks", _contains("welcome")),
    ("chat", "goodbye", _contains("goodbye")),
    ("chat", "how are you ?", _contains("well")),
]

# two-turn memory flow (order matters, appended after single-turn cases)
MEMORY_FLOW = [
    ("memory", "remember that my favorite test color is teal",
     _contains("teal")),
    ("memory", "what is my favorite test color ?", _contains("teal")),
]


def _load_model(tokenizer):
    if not Path(CKPT_PATH).exists():
        return None, None
    try:
        import numpy as np

        z = np.load(CKPT_PATH, allow_pickle=False)
        cfg = {k[4:]: int(z[k]) for k in z.files if k.startswith("cfg_")}
        if cfg.get("vocab_size") != tokenizer.vocab_size:
            return None, None
        model = TinyGPT(**cfg)
        for k in z.files:
            if k.startswith("p_"):
                model.P[k[2:]] = z[k].astype(np.float32)
        meta = {k[5:]: z[k].item() for k in z.files
                if k.startswith("meta_") and z[k].ndim == 0}
        return model, meta
    except Exception as e:  # noqa: BLE001
        print(f"[evaluate] checkpoint load failed: {e}", file=sys.stderr)
        return None, None


def run_battery(min_acc: float) -> int:
    t0 = time.time()
    bundle = build_bundle()
    init_knowledge(PASSAGES)
    model, meta = _load_model(bundle.tokenizer)
    agent = build_agent(model, bundle.tokenizer)
    agent.record = False                       # never pollute experiences
    if model is not None and meta and int(meta.get("step", 0)) >= \
            agent.neural.min_steps:
        agent.neural.trained = True

    cases = BATTERY + MEMORY_FLOW
    results = []
    for cat, msg, pred in cases:
        started = time.perf_counter()
        answer, engine = None, None
        try:
            for ev in agent.respond(msg, engine_pref="auto"):
                if ev["type"] == "done":
                    answer, engine = ev.get("answer"), ev.get("engine")
        except Exception as e:  # noqa: BLE001
            answer, engine = f"ERROR: {e}", "error"
        ms = (time.perf_counter() - started) * 1000
        ok = bool(answer) and bool(pred(answer))
        results.append({"cat": cat, "q": msg, "ok": ok,
                        "engine": engine, "ms": round(ms, 1),
                        "answer": answer})

    # cleanup test memory keys
    try:
        from .tools import MEMORY_PATH

        if MEMORY_PATH.exists():
            mem = json.loads(MEMORY_PATH.read_text())
            for k in [k for k in mem if "test" in k]:
                mem.pop(k)
            MEMORY_PATH.write_text(json.dumps(mem, indent=2, sort_keys=True))
    except Exception:  # noqa: BLE001
        pass

    # ---- report ----
    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    acc = passed / total
    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["cat"], []).append(r["ok"])
    engines: dict[str, int] = {}
    for r in results:
        engines[r["engine"] or "?"] = engines.get(r["engine"] or "?", 0) + 1
    lat = sorted(r["ms"] for r in results)
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]

    report = {
        "accuracy": round(acc, 3),
        "passed": passed,
        "total": total,
        "min_acc": min_acc,
        "by_category": {c: {"passed": sum(v), "total": len(v),
                            "acc": round(sum(v) / len(v), 3)}
                        for c, v in by_cat.items()},
        "engines": engines,
        "latency_ms": {"p50": round(p50, 1), "p99": round(lat[-1], 1)},
        "neural_ready": agent.neural.ready,
        "checkpoint": meta,
        "seconds": round(time.time() - t0, 1),
        "failures": [r for r in results if not r["ok"]],
    }

    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n═══ Lumo evaluation ═════════════════════════════")
        print(f"{'category':<12} {'passed':>8} {'acc':>8}")
        for c, v in report["by_category"].items():
            print(f"{c:<12} {v['passed']:>3}/{v['total']:<4} {v['acc']*100:>7.0f}%")
        print("─" * 46)
        print(f"overall      {passed:>3}/{total:<4} {acc*100:>7.0f}%"
              f"   (min {min_acc*100:.0f}%)")
        print(f"engines      {engines}")
        print(f"latency      p50 {p50:.0f}ms · max {lat[-1]:.0f}ms")
        print(f"model        neural_ready={agent.neural.ready} "
              f"checkpoint={meta}")
        if report["failures"]:
            print("failures:")
            for r in report["failures"]:
                print(f"  ✗ [{r['cat']}] {r['q']!r} → {r['engine']}: "
                      f"{str(r['answer'])[:90]!r}")
        verdict = "PASS" if acc >= min_acc else "FAIL"
        print(f"\n{verdict} — {report['seconds']}s\n")

    return 0 if acc >= min_acc else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Lumo evaluation battery")
    ap.add_argument("--min-acc", type=float, default=0.85)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    # --json is also inspected inside run_battery via sys.argv
    if args.json and "--json" not in sys.argv:
        sys.argv.append("--json")
    return run_battery(args.min_acc)


if __name__ == "__main__":
    sys.exit(main())
