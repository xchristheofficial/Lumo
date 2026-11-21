"""The Lumo agent loop.

Two planners share the same tool environment:
  * rules  — deterministic intent parsing (always available, zero model needed)
  * neural — the from-scratch transformer generates `thought/action/.../finish`
             lines in a ReAct-style loop; observations come from real tools.

Engine 'auto' tries neural first (when a checkpoint is loaded) and silently
falls back to rules if the model's output can't be parsed into valid tool calls.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .corpus import CONVERSIONS, PASSAGES, SYS_LINE, UNIT_ALIASES, Tokenizer
from .model import TinyGPT
from .tools import run_tool

Event = dict

# --------------------------------------------------------------------------- #
# Rule-based planner
# --------------------------------------------------------------------------- #

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
             "fifteen": 15, "twenty": 20, "twenty-five": 25, "fifty": 50,
             "hundred": 100, "thousand": 1000}


def _normalize_math(text: str) -> str | None:
    """Return an arithmetic expression if the message looks like a calc request."""
    t = text.lower().strip()
    # word → symbol
    t = t.replace("plus", "+").replace("minus", "-")
    t = re.sub(r"multiplied by|times", "*", t)
    t = re.sub(r"divided by|/", "/", t)
    t = re.sub(r"\bx\b", "*", t)
    # percent: "15% of 200" or "15 percent of 200"
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        return f"{m.group(2)} * ({m.group(1)} / 100)"
    # strip question wording
    t = re.sub(r"\b(what|whats|what's|calculate|compute|how much|the result of|"
               r"equals?|please|solve|find|value of|of)\b", " ", t)
    t = t.replace("?", " ").replace("=", " ")
    t = re.sub(r"[a-z]+", lambda m: str(_WORD_NUM.get(m.group(0), " ")), t)
    t = re.sub(r"[^0-9+\-*/().%]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    # must contain a digit and an operator (or %)
    if re.search(r"\d", t) and re.search(r"[+\-*/%]", t):
        return t
    if re.fullmatch(r"\d+(?:\.\d+)?", t) and re.search(r"[%]", text.lower()):
        return t
    return None


def _canon_unit(u: str) -> str | None:
    return UNIT_ALIASES.get(u.strip().lower().lstrip("°"))


def _normalize_conversion(t: str) -> tuple[str, str, str, str] | None:
    """Detect unit conversions → (expr, value, from_display, to_display)."""
    def lookup(u1: str | None, u2: str | None, v: str):
        if u1 and u2 and u1 != u2 and (u1, u2) in CONVERSIONS:
            from .corpus import _UNIT_DISPLAY

            return (CONVERSIONS[(u1, u2)].format(v=v), v,
                    _UNIT_DISPLAY[u1], _UNIT_DISPLAY[u2])
        return None

    # "convert 5 km to miles" / "5 km in miles" / "12 pounds to kg"
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([a-z]+)\s+"
                  r"(?:to|into|in)\s+(?:°\s*)?([a-z]+)", t)
    if m:
        hit = lookup(_canon_unit(m.group(2)), _canon_unit(m.group(3)), m.group(1))
        if hit:
            return hit
    # "how many miles in 5 km"
    m = re.search(r"how many\s+([a-z]+)\s+(?:are\s+)?in\s+"
                  r"(-?\d+(?:\.\d+)?)\s+([a-z]+)", t)
    if m:
        hit = lookup(_canon_unit(m.group(3)), _canon_unit(m.group(1)), m.group(2))
        if hit:
            return hit
    return None


@dataclass
class Plan:
    actions: list[tuple[str, str]] = field(default_factory=list)  # (tool, arg)
    direct_answer: str | None = None
    fallback: bool = False          # True for the generic last-resort reply
    conv: tuple[str, str, str] | None = None  # (value, from_unit, to_unit)
    thoughts: list[str] = field(default_factory=list)
    engine: str = "rules"


def rule_plan(msg: str) -> Plan:
    t = msg.lower().strip()
    p = Plan(engine="rules")

    # --- no-tool intents -------------------------------------------------- #
    if re.match(r"^(hi|hello|hey|yo|good (morning|afternoon|evening))\b", t):
        p.direct_answer = "hello ! i am lumo , an ai agent running on your machine . what can i help with ?"
        return p
    if re.search(r"\b(who are you|what are you|your name|what is your name)\b", t):
        p.direct_answer = ("i am lumo — a small neural agent built from scratch in numpy . "
                           "no provider api , everything runs locally .")
        return p
    if re.search(r"\b(what can you do|help|commands|capabilities|your tools|what tools)\b", t):
        p.direct_answer = ("i can 🧮 calculate expressions , 🕒 tell you the time , "
                           "💾 remember facts , 🔎 recall them , and 📚 search my knowledge base .")
        return p
    if re.match(r"^(thanks|thank you|thx|cheers)\b", t):
        p.direct_answer = "you are welcome !"
        return p
    if re.match(r"^(bye|goodbye|see you|good night)\b", t):
        p.direct_answer = "goodbye ! come back any time ."
        return p
    if re.search(r"\b(how are you|how's it going|how do you do)\b", t):
        p.direct_answer = "running well and ready to help . what do you need ?"
        return p

    # --- memory: remember -------------------------------------------------- #
    m = re.search(r"\b(?:remember|save|note down|store|write down)\b\s*(?:that\s+)?(.+)", t)
    if m and re.search(r"\bmy\b|\b(?:is|=)\b", m.group(1)):
        rest = m.group(1).strip().rstrip("!.?")
        # "my favorite color is blue" → key "favorite color"
        m2 = re.match(r"(?:my\s+)?(.+?)\s+is\s+(.+)", rest)
        if m2 and "=" not in rest:
            key, val = m2.group(1).strip(), m2.group(2).strip()
        elif "=" in rest:
            key, val = [x.strip() for x in rest.split("=", 1)]
        else:
            key, val = None, None
        if key and val:
            key = re.sub(r"^(favorite|favourite)\s+", r"favorite ", key) if key.startswith("fav") else key
            p.thoughts.append("i will store that in memory .")
            p.actions.append(("remember", f"{key} = {val}"))
            return p

    # --- memory: recall ---------------------------------------------------- #
    m = re.search(r"\b(?:what|whats|what's)\s+(?:is|was)?\s*my\s+(.+?)[?!.]*$", t)
    if not m:
        m = re.search(r"\b(?:do you|did you)\s+(?:remember|know)\s+(?:my\s+|about\s+)?(.+?)[?!.]*$", t)
    if not m:
        m = re.search(r"\b(?:what did i|recall)\b\s*(?:tell you\s+|about\s+)?(.+?)[?!.]*$", t)
    if m:
        key = m.group(1).strip()
        key = re.sub(r"^(the|a|an)\s+", "", key)
        p.thoughts.append("i will check my memory .")
        p.actions.append(("recall", key))
        return p

    # --- time / date -------------------------------------------------------- #
    if re.search(r"\b(what time|current time|the time now|time is it|clock)\b", t):
        p.thoughts.append("i will check the clock .")
        p.actions.append(("time", "now"))
        return p
    if re.search(r"\b(what (?:'s|is)? ?(?:the )?date|what day|today's date|date today)\b", t):
        p.thoughts.append("i will check today's date .")
        p.actions.append(("time", "now"))
        return p

    # --- unit conversions ---------------------------------------------------- #
    conv = _normalize_conversion(t)
    if conv:
        expr, v, u1d, u2d = conv
        p.thoughts.append("i will use the calculator to convert that .")
        p.actions.append(("calculator", expr))
        p.conv = (v, u1d, u2d)
        return p

    # --- calculator ---------------------------------------------------------- #
    expr = _normalize_math(t)
    if expr:
        p.thoughts.append("i will use the calculator tool .")
        p.actions.append(("calculator", expr))
        return p

    # --- knowledge search ---------------------------------------------------- #
    if re.search(r"\b(what|who|where|why|how|tell me|explain|define|information about|"
                 r"facts about|know about|learn about|describe)\b", t):
        p.thoughts.append("i will search my knowledge base .")
        p.actions.append(("search", re.sub(r"[^a-z0-9\s]", " ", t)))
        return p

    # --- last resort --------------------------------------------------------- #
    p.thoughts.append("i am not sure about that one .")
    p.direct_answer = ("i am a small offline agent , so my knowledge is limited to my tools : "
                       "try math like '17 * 23' , ask for the date , store a fact with "
                       "'remember my favorite color is blue' , or ask about a topic i know .")
    p.fallback = True
    return p


# --------------------------------------------------------------------------- #
# Neural planner — drives the from-scratch transformer
# --------------------------------------------------------------------------- #

STOP_TOOLS = {"finish", "action", "observation", "lumo", "user", "sys", "<sys>"}


@dataclass
class NeuralResult:
    ok: bool
    answer: str | None = None
    trace: list[tuple[str, str, str]] = field(default_factory=list)  # (kind, name, text)
    error: str | None = None


class NeuralAgent:
    def __init__(self, model: TinyGPT | None, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Only engage generation once the model has seen enough training steps.
        self.trained = False
        self.min_steps = 250

    @property
    def ready(self) -> bool:
        return self.model is not None and self.trained

    def _gen(self, prompt: str, max_new: int = 140) -> str:
        """Sample until a boundary line (action/finish/…) completes or budget ends."""
        assert self.model is not None
        ids = self.tokenizer.encode(prompt)
        arr = np_array(ids)
        newline_id = self.tokenizer.stoi.get("\n")
        out = self.model.generate(
            arr, max_new_tokens=max_new, temperature=0.15, top_k=10,
            newline_id=newline_id,
            stop_line_starts=("action", "finish", "observation",
                              "user", "<", "sys"),
            itos=self.tokenizer.itos,
        )
        new_ids = out[0, len(ids):]
        return self.tokenizer.decode(new_ids)

    @staticmethod
    def _lines(text: str) -> list[str]:
        """Split generated text into lines even if newlines were flattened."""
        norm = re.sub(r"\s+(?=(?:action|finish|observation|lumo|user)\s*:)",
                      "\n", text)
        return [ln.strip() for ln in norm.split("\n") if ln.strip()]

    def respond(self, history: list[dict], user_msg: str, max_rounds: int = 4) -> NeuralResult:
        """ReAct loop: generate → parse actions → run tools → feed observations."""
        if not self.ready:
            return NeuralResult(ok=False, error="model not loaded")

        # Single-turn prompt: tools carry cross-turn state (memory, KB), so the
        # model always sees the exact format it was trained on.
        prompt = f"{SYS_LINE}\nuser : {user_msg}\nlumo : "
        trace: list[tuple[str, str, str]] = []
        answer: str | None = None

        for _round in range(max_rounds):
            gen = self._gen(prompt)
            lines = self._lines(gen)
            prompt = prompt + gen
            executed_any = False
            for line in lines:
                low = line.lower()
                if low.startswith("finish"):
                    answer = _after_colon(line)
                    break
                if low.startswith("user") or low.startswith("<"):
                    # model broke the format — abort to caller
                    return NeuralResult(ok=False, trace=trace,
                                        error="model broke transcript format")
                if low.startswith("action"):
                    action = _after_colon(line)
                    tool, arg = _split_action(action)
                    if tool is None:
                        trace.append(("note", "", f"dropped invalid action: {action[:60]}"))
                        continue
                    obs = run_tool(tool, arg)
                    trace.append(("action", tool, arg))
                    trace.append(("observation", tool, obs))
                    prompt += f"\nobservation : {obs}\nlumo : "
                    executed_any = True
                elif low.startswith("lumo") or low.startswith("thought"):
                    trace.append(("thought", "", _after_colon(line)))
                elif low.startswith("observation"):
                    # hallucinated observation with no preceding action — drop it
                    trace.append(("note", "", "ignored model-hallucinated observation"))
            if answer is not None:
                break
            if not executed_any and not lines:
                break

        if answer is None:
            return NeuralResult(ok=False, trace=trace,
                                error="model never produced a final answer")
        answer = answer.strip()
        words = [w for w in answer.split() if any(c.isalnum() for c in w)]
        # quality gate: a real answer is a full sentence; fragments like
        # "i will" or degenerate repetition mean the model isn't confident.
        good_obs = [(n, t) for k, n, t in trace
                    if k == "observation" and not t.startswith("error")]
        anchored = bool(good_obs)
        # if the model tried tools and every single call failed, any answer
        # it makes up is unverifiable → fall back to rules
        if any(k == "action" for k, _, _ in trace) and not good_obs:
            return NeuralResult(ok=False, trace=trace,
                                error="all tool calls failed")
        ok_len = 2 <= len(answer) <= 600
        ok_shape = len(words) >= 5 or (anchored and len(words) >= 3)
        if not ok_len or not ok_shape:
            return NeuralResult(ok=False, trace=trace,
                                error=f"answer failed quality gate: {answer[:60]!r}")
        # reject degenerate repetition ("the answer is the answer is …")
        _stop = {"that", "with", "from", "have", "will", "your", "what",
                 "when", "then", "than", "them", "they", "this", "these",
                 "those", "into", "over", "after", "most", "more", "been",
                 "were", "which", "while", "under", "between", "about"}
        for w in {x.lower() for x in words if len(x) >= 4} - _stop:
            if sum(1 for x in words if x.lower() == w) >= 2:
                return NeuralResult(ok=False, trace=trace,
                                    error=f"answer is repetitive: {answer[:60]!r}")
        # reject transcript leakage ("… sys > you are lumo …")
        low_words = {x.lower() for x in words}
        if low_words & {"sys", "<", ">", "</"}:
            return NeuralResult(ok=False, trace=trace,
                                error="answer leaked transcript tokens")
        # the answer must reflect real tool results it was given
        ans_norm = " ".join(re.findall(r"[a-z0-9]+", answer.lower()))
        for name, text in good_obs:
            if name == "calculator" and re.fullmatch(r"-?\d+(?:\.\d+)?", text.strip()):
                if text.strip() not in answer:
                    return NeuralResult(ok=False, trace=trace,
                                        error=f"answer ignores tool result {text}")
            if name == "remember" and text.startswith("stored "):
                val = text.split("=", 1)[-1].strip() if "=" in text else ""
                if val and val not in answer:
                    return NeuralResult(ok=False, trace=trace,
                                        error=f"answer ignores stored value {val}")
                # the stored value must come from what the user actually said
                if val and val.lower() not in user_msg.lower():
                    return NeuralResult(ok=False, trace=trace,
                                        error=f"stored value {val} not in the question")
            if name == "recall" and not text.startswith(("no memory", "error")):
                val = text.strip()
                if val and val.lower() not in ans_norm:
                    return NeuralResult(ok=False, trace=trace,
                                        error=f"answer ignores recalled value {val}")
            if name == "time":
                m_t = re.search(r"(\d{2}):(\d{2})", text)
                if m_t:
                    hh_mm = f"{m_t.group(1)}:{m_t.group(2)}"
                    if hh_mm not in answer.replace(" ", ""):
                        return NeuralResult(ok=False, trace=trace,
                                            error=f"answer time != clock {hh_mm}")
            if name == "search" and not text.startswith("no relevant"):
                prefix = " ".join(re.findall(r"[a-z0-9]+", text.lower())[:5])
                if prefix and prefix not in ans_norm:
                    return NeuralResult(ok=False, trace=trace,
                                        error="answer doesn't reflect the passage")
        # For clear intents the deterministic planner knows the right tool —
        # if the model reached for a different one (or none), reject it.
        expected = rule_plan(user_msg)
        if expected.actions and not expected.fallback:
            exp_tool = expected.actions[0][0]
            first_tool = next((n for k, n, _ in trace if k == "action"), None)
            if first_tool != exp_tool:
                return NeuralResult(ok=False, trace=trace,
                                    error=f"tool mismatch: expected "
                                          f"'{exp_tool}', model used "
                                          f"'{first_tool}'")
        return NeuralResult(ok=True, answer=answer, trace=trace)


def _after_colon(line: str) -> str:
    if ":" in line:
        return line.split(":", 1)[1].strip()
    # tokenizer may render ':' as its own token — handle 'action calculator 1 + 2'
    parts = line.split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _split_action(action: str) -> tuple[str | None, str]:
    action = action.strip()
    if not action:
        return None, ""
    # 'calculator 17 * 23' — first token is the tool
    m = re.match(r"^([a-z_]+)\s*(.*)$", action, flags=re.IGNORECASE)
    if not m:
        return None, ""
    tool, arg = m.group(1).lower(), m.group(2).strip()
    from .tools import TOOLS
    if tool not in TOOLS:
        return None, ""
    return tool, arg


def np_array(ids: list[int]):
    import numpy as np

    return np.asarray([ids], dtype=np.int32)


# --------------------------------------------------------------------------- #
# Unified agent
# --------------------------------------------------------------------------- #

class Agent:
    def __init__(self, neural: NeuralAgent):
        self.neural = neural
        self.history: list[dict] = []
        self.record = True  # set False to disable experience recording

    def reload_model(self, model: TinyGPT) -> None:
        self.neural.model = model

    def respond(self, msg: str, engine_pref: str = "auto") -> Iterator[Event]:
        """Yield SSE-able events for one user turn."""
        yield {"type": "trace", "kind": "thought", "text": "thinking …"}

        plan: Plan | None = None
        neural: NeuralResult | None = None
        use_neural = engine_pref in ("auto", "neural") and self.neural.ready

        # small talk / identity / capability replies are handled
        # deterministically in auto mode — always correct, never hallucinated.
        if use_neural and engine_pref == "auto":
            rp = rule_plan(msg)
            if rp.direct_answer is not None and not rp.fallback:
                plan = rp
                use_neural = False

        if use_neural:
            try:
                neural = self.neural.respond(self.history, msg)
            except Exception as e:  # noqa: BLE001
                neural = NeuralResult(ok=False, error=f"neural error: {e}")
            if neural is not None and neural.ok:
                # replay the neural trace for the UI
                for kind, name, text in neural.trace:
                    yield {"type": "trace", "kind": kind, "name": name, "text": text}
                engine_used = "neural"
                answer = neural.answer
            elif engine_pref == "neural":
                # strict neural mode: report the failure honestly
                for kind, name, text in (neural.trace if neural else []):
                    yield {"type": "trace", "kind": kind, "name": name, "text": text}
                yield {"type": "trace", "kind": "note", "text":
                       f"neural planner failed ({neural.error if neural else 'unknown'}) — "
                       "try training more steps or switch to auto/rules"}
                yield {"type": "done", "answer": None, "engine": "neural-failed"}
                return
            else:
                # auto → fall back to rules
                if neural is not None:
                    for kind, name, text in neural.trace:
                        yield {"type": "trace", "kind": kind, "name": name, "text": text}
                    yield {"type": "trace", "kind": "note",
                           "text": f"neural fallback ({neural.error}) → rules"}
                plan = rule_plan(msg)
        else:
            if engine_pref == "neural":
                yield {"type": "trace", "kind": "note",
                       "text": "no trained model yet — train first or use rules"}
                yield {"type": "done", "answer": None, "engine": "neural-unavailable"}
                return
            if plan is None:
                plan = rule_plan(msg)

        # ---- execute the rules plan --------------------------------------
        a_lines: list[str] = []          # assistant transcript for next turn
        a_actions: list[tuple[str, str, str]] = []  # (tool, arg, obs)
        a_thoughts: list[str] = []

        if plan is not None:
            for thought in plan.thoughts:
                yield {"type": "trace", "kind": "thought", "text": thought}
            a_thoughts = list(plan.thoughts)
            if plan.direct_answer is not None:
                answer = plan.direct_answer
                yield {"type": "token", "text": answer}
                a_lines = ["i know how to answer this .", f"finish : {answer}"]
            else:
                answer = None
                for tool, arg in plan.actions:
                    obs = run_tool(tool, arg)
                    yield {"type": "trace", "kind": "action", "name": tool, "text": arg}
                    yield {"type": "trace", "kind": "observation", "name": tool, "text": obs}
                    a_actions.append((tool, arg, obs))
                    if tool == "remember" and not obs.startswith("error"):
                        answer = f"got it — {obs} ."
                    elif tool == "recall":
                        if obs.startswith("no memory") or obs.startswith("error"):
                            answer = (f"i do not have that in memory yet — try "
                                      f"'remember {arg} is …' first .")
                        else:
                            answer = (f"your {arg} is {obs} ."
                                      if obs.lower() != arg.lower() else obs)
                    elif tool == "time":
                        answer = f"right now it is {obs} ."
                    elif tool == "calculator":
                        if plan.conv and not obs.startswith("error"):
                            v, u1d, u2d = plan.conv
                            answer = f"{v} {u1d} = {obs} {u2d} ."
                        else:
                            answer = f"{arg} = {obs}" if not obs.startswith("error") else obs
                    elif tool == "search":
                        answer = obs if not obs.startswith("no relevant") else \
                            ("i could not find that in my knowledge base . my offline library "
                             "covers science , nature , history , and computing basics — "
                             "try asking about one of those .")
                if answer is None:
                    answer = "done ."
                a_lines = _hist_transcript(a_thoughts, a_actions, answer)
            # stream the answer as one token event (rules answers are short)
            if plan.direct_answer is None:
                yield {"type": "token", "text": answer}
            engine_used = "rules"
        else:
            # neural path finished above
            engine_used = "neural"
            yield {"type": "token", "text": answer}
            a_lines = _hist_from_neural_trace(neural.trace, answer) if neural else []

        self.history.append({"role": "user", "content": msg})
        self.history.append({"role": "assistant",
                             "content": "\n".join(a_lines) if a_lines else str(answer)})
        self.history = self.history[-12:]

        # Write the verified exchange down so the autopilot trainer can
        # learn from it while you're away (experience replay).
        if answer and a_lines and self.record:
            try:
                from . import experience

                experience.record(msg, "\n".join(a_lines), engine_used)
            except Exception:  # noqa: BLE001
                pass

        yield {"type": "done", "answer": answer, "engine": engine_used}


def _hist_transcript(thoughts: list[str],
                     actions: list[tuple[str, str, str]],
                     answer: str) -> list[str]:
    """Rebuild the assistant turn in exact training format."""
    lines: list[str] = []
    if thoughts:
        lines.append(thoughts[0])
    for tool, arg, obs in actions:
        lines.append(f"action : {tool} {arg}")
        lines.append(f"observation : {obs}")
    lines.append("lumo : done .")
    lines.append(f"finish : {answer}")
    return lines


def _hist_from_neural_trace(trace, answer: str) -> list[str]:
    thoughts: list[str] = []
    actions: list[tuple[str, str, str]] = []
    seen_action = False
    for kind, name, text in trace:
        if kind == "thought" and text and text != "thinking …" and not seen_action:
            thoughts.append(text)
        elif kind == "action":
            seen_action = True
            actions.append((name, text, ""))
        elif kind == "observation" and actions and actions[-1][2] == "":
            actions[-1] = (actions[-1][0], actions[-1][1], text)
    return _hist_transcript(thoughts, actions, answer)


def build_agent(model: TinyGPT | None, tokenizer: Tokenizer) -> Agent:
    return Agent(NeuralAgent(model, tokenizer))
