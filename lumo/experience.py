"""Experience replay — the agent writes down its conversations so the
background trainer can learn from them while you're away.

Every successful turn (rules- or gate-verified neural) is appended to
`data/experiences.jsonl` as an exact single-turn training block:

    user : <your question>
    lumo : <thought / action / observation / finish transcript>

The trainer mixes these blocks into its batches, so the model literally
gets better at *your* questions over time.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPERIENCE_PATH = DATA_DIR / "experiences.jsonl"
MAX_ENTRIES = 5000

_lock = threading.Lock()
_seen: set[str] | None = None


def _key(user_msg: str, transcript: str) -> str:
    raw = f"{user_msg.strip().lower()}||{transcript.strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_seen() -> set[str]:
    global _seen
    if _seen is None:
        _seen = set()
        if EXPERIENCE_PATH.exists():
            try:
                for line in EXPERIENCE_PATH.read_text().splitlines():
                    if line.strip():
                        _seen.add(json.loads(line)["k"])
            except (json.JSONDecodeError, OSError, KeyError):
                pass
    return _seen


def record(user_msg: str, transcript: str, engine: str) -> bool:
    """Append one experience. Returns True if it was new."""
    user_msg = (user_msg or "").strip()
    transcript = (transcript or "").strip()
    if not user_msg or not transcript:
        return False
    k = _key(user_msg, transcript)
    with _lock:
        seen = _load_seen()
        if k in seen:
            return False
        seen.add(k)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"k": k, "q": user_msg, "a": transcript, "e": engine}
        with EXPERIENCE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # cap: rewrite the file when it grows too large
        if len(seen) > MAX_ENTRIES:
            _prune(seen)
    return True


def _prune(seen: set[str]) -> None:
    try:
        lines = EXPERIENCE_PATH.read_text().splitlines()[-MAX_ENTRIES:]
        EXPERIENCE_PATH.write_text("\n".join(lines) + "\n")
        for line in lines[:-MAX_ENTRIES // 2]:
            try:
                seen.discard(json.loads(line)["k"])
            except (json.JSONDecodeError, KeyError):
                pass
    except OSError:
        pass


def load_entries() -> list[dict]:
    """All recorded experiences (for the trainer)."""
    if not EXPERIENCE_PATH.exists():
        return []
    out: list[dict] = []
    try:
        for line in EXPERIENCE_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e.get("q") and e.get("a"):
                    out.append(e)
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def count() -> int:
    return len(load_entries())
