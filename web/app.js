/* Lumo web UI — chat over SSE, live training telemetry, no external deps. */
"use strict";

const $ = (id) => document.getElementById(id);
const messages = $("messages");
const form = $("chat-form");
const input = $("chat-input");
const sendBtn = $("send-btn");

let enginePref = "auto";
let busy = false;
let trainState = { running: false, step: 0, max_steps: 0, loss: null };
let lossHistory = []; // [step, loss]

const SUGGESTIONS = [
  "what is 17 * 23 ?",
  "remember that my favorite color is blue",
  "what is my favorite color ?",
  "what time is it ?",
  "tell me about black holes",
  "what can you do ?",
];

/* ── suggestions ─────────────────────────────────────────────── */
const suggestBox = $("suggest");
SUGGESTIONS.forEach((s) => {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = s;
  b.onclick = () => { input.value = s; input.focus(); };
  suggestBox.appendChild(b);
});

/* ── status ──────────────────────────────────────────────────── */
async function refreshStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    renderStatus(s);
  } catch (e) {
    console.error("status failed", e);
  }
}

function renderStatus(s) {
  const m = s.model;
  $("pill-params").textContent = `${(m.params / 1e6).toFixed(2)}M params · vocab ${m.vocab}`;
  const neuralPill = $("pill-model");
  if (m.neural_ready) {
    neuralPill.textContent = "model: neural ready";
    neuralPill.classList.add("on");
  } else {
    neuralPill.textContent = "model: training…";
    neuralPill.classList.remove("on");
  }

  // tools
  const toolsUl = $("tools");
  toolsUl.innerHTML = "";
  s.tools.forEach((t) => {
    const li = document.createElement("li");
    li.innerHTML = `<code>${t.emoji} ${t.name}</code><span>${t.desc}</span>`;
    toolsUl.appendChild(li);
  });

  // memory
  renderMemory(s.memory || {});

  // training snapshot
  if (s.training) {
    trainState = { ...trainState, ...s.training };
    if (Array.isArray(s.training.history) && s.training.history.length) {
      lossHistory = s.training.history.slice();
      drawChart();
    }
    paintTraining();
  }

  $("boot-info").textContent =
    `boot ${s.boot_ms}ms · ${m.corpus_tokens.toLocaleString()} training tokens · ` +
    `${s.knowledge_passages} knowledge passages · ${s.provider_apis} provider APIs`;
}

function renderMemory(mem) {
  const box = $("memory");
  const keys = Object.keys(mem);
  if (!keys.length) {
    box.innerHTML = `<span class="dim">nothing stored yet</span>`;
    return;
  }
  box.innerHTML = "";
  keys.forEach((k) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<b>${escapeHtml(k)}</b><span>${escapeHtml(mem[k])}</span>`;
    box.appendChild(row);
  });
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

/* ── chat ────────────────────────────────────────────────────── */
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = input.value.trim();
  if (!msg || busy) return;
  input.value = "";
  sendUser(msg);
});

function appendUser(msg) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble"></div>`;
  el.querySelector(".bubble").textContent = msg;
  messages.appendChild(el);
  scrollDown();
}

function createAssistant() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `
    <div class="bubble">
      <div class="trace"></div>
      <p class="answer cursor"></p>
    </div>
    <span class="meta">…</span>`;
  messages.appendChild(el);
  scrollDown();
  return {
    root: el,
    trace: el.querySelector(".trace"),
    answer: el.querySelector(".answer"),
    meta: el.querySelector(".meta"),
  };
}

function addTrace(node, kind, name, text) {
  const line = document.createElement("div");
  line.className = `trace-line ${kind}`;
  if (kind === "action") line.innerHTML = `<b>⚙ ${escapeHtml(name)}</b> ${escapeHtml(text)}`;
  else if (kind === "observation") line.innerHTML = `<b>👁 ${escapeHtml(name)}</b> ${escapeHtml(text)}`;
  else line.textContent = text;
  node.trace.appendChild(line);
  scrollDown();
}

async function sendUser(msg) {
  busy = true;
  sendBtn.disabled = true;
  appendUser(msg);
  const node = createAssistant();
  let answer = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg, engine: enginePref }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        handleEvent(ev, node, (t) => { answer += t; });
      }
    }
  } catch (err) {
    addTrace(node, "note", "", `stream error: ${err.message}`);
  }

  node.answer.classList.remove("cursor");
  if (!answer) node.answer.textContent = node.answer.textContent || "…";
  busy = false;
  sendBtn.disabled = false;
  input.focus();
  scrollDown();
  refreshStatus(); // memory may have changed
}

function handleEvent(ev, node, push) {
  switch (ev.type) {
    case "trace": {
      // replace the initial "thinking …" placeholder on first real trace
      if (node.trace.childElementCount === 1 &&
          node.trace.firstElementChild.textContent === "thinking …") {
        node.trace.innerHTML = "";
      }
      addTrace(node, ev.kind, ev.name || "", ev.text);
      break;
    }
    case "token": {
      node.answer.classList.remove("cursor");
      push(ev.text);
      node.answer.textContent = node.answer.textContent + ev.text;
      scrollDown();
      break;
    }
    case "done": {
      const eng = ev.engine || "?";
      node.meta.innerHTML = `engine: <span class="eng-${eng}">${eng}</span>` +
        (ev.answer == null ? " · no answer" : "");
      if (ev.answer != null && !node.answer.textContent) {
        node.answer.textContent = ev.answer;
      }
      break;
    }
    case "error":
      addTrace(node, "note", "", ev.message || "unknown error");
      break;
  }
}

function scrollDown() {
  messages.scrollTop = messages.scrollHeight;
}

/* ── engine toggle ───────────────────────────────────────────── */
document.querySelectorAll("#engine-toggle button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#engine-toggle button")
      .forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    enginePref = b.dataset.engine;
  });
});

/* ── training controls ───────────────────────────────────────── */
async function startTrain(mode) {
  try {
    const r = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const j = await r.json();
    if (!j.ok) alert(j.message);
    paintTraining();
  } catch (e) { alert(`training failed: ${e.message}`); }
}
$("btn-quick").onclick = () => startTrain("quick");
$("btn-full").onclick = () => startTrain("full");
$("btn-stop").onclick = async () => {
  await fetch("/api/train/stop", { method: "POST" });
};

$("btn-clear-mem").onclick = async () => {
  await fetch("/api/memory/forget", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  renderMemory({});
};

function paintTraining() {
  const st = trainState;
  const pill = $("pill-train");
  if (st.running) {
    pill.textContent = `training: step ${st.step}${st.max_steps ? "/" + st.max_steps : ""}`;
    pill.classList.add("train-on");
  } else if (st.done || st.step > 0) {
    pill.textContent = `trained: step ${st.step} · loss ${st.loss ?? "—"}`;
    pill.classList.add("on");
    pill.classList.remove("train-on");
  } else {
    pill.textContent = "training: idle";
    pill.classList.remove("train-on", "on");
  }
  $("btn-stop").disabled = !st.running;
  $("btn-quick").disabled = st.running;
  $("btn-full").disabled = st.running;
  $("st-step").textContent = st.step ?? 0;
  $("st-loss").textContent = st.loss ?? "—";
  $("st-tps").textContent = st.tokens_per_s ?? "—";
  $("st-elapsed").textContent = st.elapsed != null ? `${st.elapsed}s` : "—";
  const pct = st.max_steps ? Math.min(100, (100 * st.step) / st.max_steps) : 0;
  $("progress-bar").style.width = `${pct}%`;
}

/* ── loss chart ──────────────────────────────────────────────── */
const canvas = $("loss-chart");
const ctx = canvas.getContext("2d");

function drawChart() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  // grid
  ctx.strokeStyle = "rgba(139,151,184,.15)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h * i) / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  if (lossHistory.length < 2) {
    ctx.fillStyle = "rgba(139,151,184,.7)";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("waiting for training data…", 14, h / 2);
    return;
  }
  const xs = lossHistory.map((p) => p[0]);
  const ys = lossHistory.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  if (maxY - minY < 1e-4) { maxY = minY + 1; }
  const pad = 14;
  const X = (x) => pad + ((x - minX) / Math.max(1, maxX - minX)) * (w - 2 * pad);
  const Y = (y) => pad + (1 - (y - minY) / (maxY - minY)) * (h - 2 * pad);

  // area
  ctx.beginPath();
  lossHistory.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
  ctx.lineTo(X(maxX), h - pad); ctx.lineTo(X(minX), h - pad); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(94,234,212,.30)");
  grad.addColorStop(1, "rgba(94,234,212,0)");
  ctx.fillStyle = grad; ctx.fill();

  // line
  ctx.beginPath();
  lossHistory.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
  ctx.strokeStyle = "#5eead4";
  ctx.lineWidth = 2;
  ctx.shadowColor = "rgba(94,234,212,.9)";
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // last point
  const last = lossHistory[lossHistory.length - 1];
  ctx.beginPath();
  ctx.arc(X(last[0]), Y(last[1]), 3.5, 0, Math.PI * 2);
  ctx.fillStyle = "#a78bfa"; ctx.fill();

  // labels
  ctx.fillStyle = "rgba(223,231,255,.85)";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(maxY.toFixed(2), 6, 14);
  ctx.fillText(minY.toFixed(2), 6, h - 6);
  ctx.fillStyle = "rgba(167,139,250,.95)";
  ctx.textAlign = "right";
  ctx.fillText(`step ${last[0]}`, w - 8, 14);
  ctx.textAlign = "left";
}
drawChart();

/* ── training SSE ────────────────────────────────────────────── */
function connectTraining() {
  const es = new EventSource("/api/train/events");
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "snapshot") {
      trainState = { ...trainState, ...ev.state };
      if (Array.isArray(ev.history) && ev.history.length) {
        lossHistory = ev.history.slice();
        drawChart();
      }
      paintTraining();
    } else if (ev.type === "progress") {
      trainState = {
        ...trainState, running: true, step: ev.step, loss: ev.loss,
        elapsed: ev.elapsed, tokens_per_s: ev.tokens_per_s,
        max_steps: trainState.max_steps || 0,
      };
      lossHistory.push([ev.step, ev.loss]);
      if (lossHistory.length > 2500) lossHistory = lossHistory.slice(-2500);
      paintTraining();
      drawChart();
    } else if (ev.type === "done") {
      trainState = {
        ...trainState, running: false, done: true, step: ev.step,
        loss: ev.loss ?? trainState.loss,
      };
      paintTraining();
      refreshStatus();
    } else if (ev.type === "error") {
      trainState = { ...trainState, running: false, message: ev.message };
      paintTraining();
      refreshStatus();
    }
  };
  es.onerror = () => { /* EventSource auto-reconnects */ };
}
connectTraining();

refreshStatus();
setInterval(refreshStatus, 15000);
