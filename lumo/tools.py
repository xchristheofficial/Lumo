"""Tools available to the Lumo agent — all local, no network, no provider APIs."""
from __future__ import annotations

import ast
import datetime as dt
import json
import operator
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMORY_PATH = DATA_DIR / "memory.json"

# --------------------------------------------------------------------------- #
# Calculator (safe expression evaluation via AST whitelist)
# --------------------------------------------------------------------------- #

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": lambda x: float(np.sqrt(x)) if x >= 0 else float("nan"),
    "int": int,
    "float": float,
}
_MAX_LEN = 120
_MAX_POW_EXP = 64


def safe_eval(expr: str):
    """Evaluate a arithmetic expression safely. Raises ValueError on anything odd."""
    expr = expr.strip().rstrip("?").strip()
    if not expr or len(expr) > _MAX_LEN:
        raise ValueError("empty or too-long expression")
    # allow the agent to write '17 x 23' / '17 * 23'
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", expr, flags=re.IGNORECASE)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse: {expr}") from e

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > _MAX_POW_EXP:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id.lower()
            if name in _FUNCS and len(node.args) == 1 and not node.keywords:
                return _FUNCS[name](ev(node.args[0]))
        raise ValueError("unsupported syntax")

    val = ev(tree)
    if isinstance(val, complex) or (isinstance(val, float) and not np.isfinite(val)):
        raise ValueError("result is not a real number")
    if abs(float(val)) > 1e15:
        raise ValueError("result too large")
    return val


def fmt_num(x) -> str:
    if isinstance(x, float) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def tool_calculator(arg: str) -> str:
    try:
        return fmt_num(safe_eval(arg))
    except Exception as e:  # noqa: BLE001 — surface a friendly error to the agent
        return f"error: could not compute that ({e})"


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #

def tool_time(arg: str) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S utc ( %A )").lower()


# --------------------------------------------------------------------------- #
# Memory (persists to data/memory.json)
# --------------------------------------------------------------------------- #

def _load_memory() -> dict:
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_memory(mem: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(mem, indent=2, sort_keys=True))


def tool_remember(arg: str) -> str:
    if "=" not in arg:
        return "error: use 'key = value'"
    key, val = arg.split("=", 1)
    key = key.strip().lower()
    val = val.strip()
    if not key or not val:
        return "error: empty key or value"
    mem = _load_memory()
    mem[key] = val
    _save_memory(mem)
    return f"stored {key} = {val}"


def tool_recall(arg: str) -> str:
    key = arg.strip().lower()
    mem = _load_memory()
    if key in mem:
        return mem[key]
    # fuzzy-ish fallback: substring match
    for k, v in mem.items():
        if key and (key in k or k in key):
            return v
    return f"no memory found for '{key}'"


# --------------------------------------------------------------------------- #
# Knowledge search (TF-IDF over the built-in passages)
# --------------------------------------------------------------------------- #

class KnowledgeBase:
    def __init__(self, passages: list[tuple[str, str]]):
        self.passages = [p for _, p in passages]
        self.docs = [self._tok(p) for p in self.passages]
        # df / idf
        df: dict[str, int] = {}
        for toks in self.docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = max(len(self.docs), 1)
        self.idf = {t: np.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        vocab = sorted(self.idf)
        self.col = {t: i for i, t in enumerate(vocab)}
        self.matrix = np.zeros((len(self.docs), len(vocab)), dtype=np.float32)
        for i, toks in enumerate(self.docs):
            for t in toks:
                j = self.col.get(t)
                if j is not None:
                    self.matrix[i, j] += 1.0
        # l2 normalize rows
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix /= norms

    @staticmethod
    def _tok(text: str) -> list[str]:
        # light stemming so "giraffes" matches "giraffe" (applied to both
        # the index and queries, so it stays consistent)
        out = []
        for t in re.findall(r"[a-z0-9]+", text.lower()):
            if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
                t = t[:-1]
            out.append(t)
        return out

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.col), dtype=np.float32)
        for t in self._tok(text):
            j = self.col.get(t)
            if j is not None:
                v[j] += self.idf[t]
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def search(self, query: str, top: int = 1) -> tuple[str, float]:
        if not self.passages:
            return ("knowledge base is empty", 0.0)
        q = self._vec(query)
        scores = self.matrix @ q
        order = np.argsort(-scores)[:top]
        best = int(order[0])
        return (self.passages[best], float(scores[best]))


_KB: KnowledgeBase | None = None


def init_knowledge(passages: list[tuple[str, str]]) -> None:
    global _KB
    _KB = KnowledgeBase(passages)


def tool_search(arg: str) -> str:
    if _KB is None:
        return "error: knowledge base not initialized"
    text, score = _KB.search(arg)
    if score < 0.12:
        return "no relevant passage found"
    return text


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

TOOLS: dict[str, dict] = {
    "calculator": {
        "fn": tool_calculator,
        "desc": "evaluate an arithmetic expression, e.g. '17 * 23'",
        "emoji": "🧮",
    },
    "time": {
        "fn": tool_time,
        "desc": "current UTC date and time",
        "emoji": "🕒",
    },
    "remember": {
        "fn": tool_remember,
        "desc": "store a fact, format 'key = value'",
        "emoji": "💾",
    },
    "recall": {
        "fn": tool_recall,
        "desc": "read back a stored fact by key",
        "emoji": "🔎",
    },
    "search": {
        "fn": tool_search,
        "desc": "search the built-in knowledge base",
        "emoji": "📚",
    },
}


def run_tool(name: str, arg: str) -> str:
    name = name.strip().lower()
    if name not in TOOLS:
        return f"error: unknown tool '{name}'"
    return TOOLS[name]["fn"](arg)
