"""A tiny GPT-style decoder written from scratch in NumPy.

No PyTorch, no TensorFlow, no provider APIs — just matrices:
  * learned token + position embeddings
  * pre-LayerNorm transformer blocks with causal multi-head self-attention
  * GELU feed-forward, tied embedding output head
  * hand-derived backprop and the Adam optimizer
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Activations / norms
# --------------------------------------------------------------------------- #

SQRT_2_PI = np.sqrt(2.0 / np.pi)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(SQRT_2_PI * (x + 0.044715 * x**3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(SQRT_2_PI * (x + 0.044715 * x**3))
    dt = SQRT_2_PI * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dt


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * inv
    return xhat * g + b, (x, mean, var, inv, xhat, g)


def layer_norm_backward(dy: np.ndarray, cache):
    x, mean, var, inv, xhat, g = cache
    N = x.shape[-1]
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * g
    dx = (inv / N) * (
        N * dxhat
        - dxhat.sum(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
    )
    return dx, dg, db


def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Adam
# --------------------------------------------------------------------------- #

class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 3e-4,
                 betas=(0.9, 0.95), eps: float = 1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray],
             lr: float | None = None):
        self.t += 1
        lr = self.lr if lr is None else lr
        b1t = 1 - self.b1**self.t
        b2t = 1 - self.b2**self.t
        for k, p in params.items():
            g = grads[k]
            m = self.m[k]
            v = self.v[k]
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * (g * g)
            p -= lr * (m / b1t) / (np.sqrt(v / b2t) + self.eps)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class TinyGPT:
    def __init__(self, vocab_size: int, seq_len: int = 96, d_model: int = 96,
                 n_head: int = 3, n_layer: int = 2, d_ff: int = 384, seed: int = 0):
        self.cfg = dict(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model,
                        n_head=n_head, n_layer=n_layer, d_ff=d_ff, seed=seed)
        assert d_model % n_head == 0
        rng = np.random.default_rng(seed)
        d, dh = d_model, d_model // n_head
        res_scale = 1.0 / np.sqrt(2.0 * n_layer)

        def w(shape, scale=0.02):
            return (rng.standard_normal(shape) * scale).astype(np.float32)

        P: dict[str, np.ndarray] = {}
        P["emb"] = w((vocab_size, d))
        P["pos"] = w((seq_len, d))
        for i in range(n_layer):
            p = f"l{i}."
            P[p + "ln1.g"] = np.ones(d, dtype=np.float32)
            P[p + "ln1.b"] = np.zeros(d, dtype=np.float32)
            P[p + "wq"] = w((d, d)); P[p + "bq"] = np.zeros(d, np.float32)
            P[p + "wk"] = w((d, d)); P[p + "bk"] = np.zeros(d, np.float32)
            P[p + "wv"] = w((d, d)); P[p + "bv"] = np.zeros(d, np.float32)
            P[p + "wo"] = w((d, d), 0.02 * res_scale); P[p + "bo"] = np.zeros(d, np.float32)
            P[p + "ln2.g"] = np.ones(d, dtype=np.float32)
            P[p + "ln2.b"] = np.zeros(d, dtype=np.float32)
            P[p + "w1"] = w((d, d_ff))
            P[p + "b1"] = np.zeros(d_ff, np.float32)
            P[p + "w2"] = w((d_ff, d), 0.02 * res_scale)
            P[p + "b2"] = np.zeros(d, np.float32)
        P["lnf.g"] = np.ones(d, dtype=np.float32)
        P["lnf.b"] = np.zeros(d, dtype=np.float32)

        self.P = P
        self.adam = Adam(P)

    # ----------------------------- forward -------------------------------- #

    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """idx, targets: (B, T) int arrays. Returns (logits, loss, cache)."""
        B, T = idx.shape
        cfg = self.cfg
        nh, dh = cfg["n_head"], cfg["d_model"] // cfg["n_head"]
        scale = 1.0 / np.sqrt(dh)
        P = self.P

        x = P["emb"][idx] + P["pos"][None, :T]           # (B,T,d)
        cache: dict = {"idx": idx, "layers": [], "x_in": x}

        # causal mask (shared across heads/layers)
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

        for i in range(cfg["n_layer"]):
            p = f"l{i}."
            lc: dict = {}
            xn1, lc["ln1"] = layer_norm(x, P[p + "ln1.g"], P[p + "ln1.b"])
            q = xn1 @ P[p + "wq"] + P[p + "bq"]
            k = xn1 @ P[p + "wk"] + P[p + "bk"]
            v = xn1 @ P[p + "wv"] + P[p + "bv"]
            lc["xn1"], lc["q"], lc["k"], lc["v"] = xn1, q, k, v

            qh = q.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)   # (B,nh,T,dh)
            kh = k.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
            vh = v.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
            att = (qh @ kh.transpose(0, 1, 3, 2)) * scale + mask
            probs = softmax_rows(att)
            oh = probs @ vh
            o = oh.transpose(0, 2, 1, 3).reshape(B, T, cfg["d_model"])
            lc["qh"], lc["kh"], lc["vh"], lc["probs"], lc["oh"] = qh, kh, vh, probs, oh
            attn_out = o @ P[p + "wo"] + P[p + "bo"]
            lc["o_pre"] = o
            x1 = x + attn_out

            xn2, lc["ln2"] = layer_norm(x1, P[p + "ln2.g"], P[p + "ln2.b"])
            z = xn2 @ P[p + "w1"] + P[p + "b1"]
            f = gelu(z)
            ff_out = f @ P[p + "w2"] + P[p + "b2"]
            lc["xn2"], lc["z"], lc["f"] = xn2, z, f
            x = x1 + ff_out
            cache["layers"].append(lc)

        xnf, cache["lnf"] = layer_norm(x, P["lnf.g"], P["lnf.b"])
        logits = xnf @ P["emb"].T                          # tied head (V,d)
        cache["xnf"] = xnf
        cache["logits"] = logits

        loss = None
        if targets is not None:
            # cross-entropy, computed stably
            flat = logits.reshape(-1, logits.shape[-1])
            tgt = targets.reshape(-1)
            mx = flat.max(axis=1, keepdims=True)
            e = np.exp(flat - mx)
            sum_e = e.sum(axis=1, keepdims=True)
            logp = (flat - mx) - np.log(sum_e)
            nll = -logp[np.arange(tgt.size), tgt]
            loss = float(nll.mean())
            cache["probs_out"] = e / sum_e                # softmax for backward
        return logits, loss, cache

    # ---------------------------- backward -------------------------------- #

    def backward(self, cache: dict, targets: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.cfg
        B, T = cache["idx"].shape
        nh, dh = cfg["n_head"], cfg["d_model"] // cfg["n_head"]
        d_model = cfg["d_model"]
        scale = 1.0 / np.sqrt(dh)
        P = self.P
        G: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in P.items()}

        # dloss/dlogits from softmax + mean over B*T
        probs = cache["probs_out"]                          # (B*T, V)
        V = probs.shape[-1]
        dlogits = probs.copy()
        dlogits[np.arange(B * T), targets.reshape(-1)] -= 1.0
        dlogits /= B * T
        dlogits = dlogits.astype(np.float32).reshape(B, T, V)

        xnf = cache["xnf"]
        # tied head
        d_xnf = dlogits @ P["emb"]                          # (B,T,d)
        G["emb"] += dlogits.reshape(-1, cfg["vocab_size"]).T @ xnf.reshape(-1, d_model)
        dx, dg, db = layer_norm_backward(d_xnf, cache["lnf"])
        G["lnf.g"] += dg; G["lnf.b"] += db

        for i in reversed(range(cfg["n_layer"])):
            p = f"l{i}."
            lc = cache["layers"][i]

            # x = x1 + ff_out
            d_ff_out = dx
            d_x1 = dx.copy()

            # ff_out = f @ w2 + b2
            f = lc["f"]
            G[p + "w2"] += f.reshape(-1, f.shape[-1]).T @ d_ff_out.reshape(-1, d_model)
            G[p + "b2"] += d_ff_out.sum(axis=(0, 1))
            d_f = d_ff_out @ P[p + "w2"].T

            # f = gelu(z)
            d_z = d_f * gelu_grad(lc["z"])
            xn2 = lc["xn2"]
            G[p + "w1"] += xn2.reshape(-1, xn2.shape[-1]).T @ d_z.reshape(-1, z_dim(d_model, cfg))
            G[p + "b1"] += d_z.sum(axis=(0, 1))
            d_xn2 = d_z @ P[p + "w1"].T
            dln, dg, db = layer_norm_backward(d_xn2, lc["ln2"])
            G[p + "ln2.g"] += dg; G[p + "ln2.b"] += db
            d_x1 += dln

            # x1 = x + attn_out
            d_attn_out = d_x1
            o_pre = lc["o_pre"]
            G[p + "wo"] += o_pre.reshape(-1, d_model).T @ d_attn_out.reshape(-1, d_model)
            G[p + "bo"] += d_attn_out.sum(axis=(0, 1))
            d_o = d_attn_out @ P[p + "wo"].T

            # attention reverse
            oh = lc["oh"]                                    # (B,nh,T,dh)
            d_oh = d_o.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
            probs_a = lc["probs"]
            d_probs = d_oh @ lc["vh"].transpose(0, 1, 3, 2)  # (B,nh,T,T)
            d_vh = probs_a.transpose(0, 1, 3, 2) @ d_oh
            # softmax backward row-wise
            s = (d_probs * probs_a).sum(axis=-1, keepdims=True)
            d_att = (d_probs - s) * probs_a
            d_att = (d_att * scale).astype(np.float32)
            qh, kh, vh = lc["qh"], lc["kh"], lc["vh"]
            d_qh = d_att @ kh
            d_kh = d_att.transpose(0, 1, 3, 2) @ qh
            d_q = d_qh.transpose(0, 2, 1, 3).reshape(B, T, d_model)
            d_k = d_kh.transpose(0, 2, 1, 3).reshape(B, T, d_model)
            d_v = d_vh.transpose(0, 2, 1, 3).reshape(B, T, d_model)

            xn1 = lc["xn1"]
            for name, din in (("wq", d_q), ("wk", d_k), ("wv", d_v)):
                G[p + name] += xn1.reshape(-1, d_model).T @ din.reshape(-1, d_model)
                G[p + "b" + name[1]] += din.sum(axis=(0, 1))
            d_xn1 = (d_q @ P[p + "wq"].T + d_k @ P[p + "wk"].T + d_v @ P[p + "wv"].T)
            dln, dg, db = layer_norm_backward(d_xn1, lc["ln1"])
            G[p + "ln1.g"] += dg; G[p + "ln1.b"] += db
            dx = d_x1 + dln                                  # residual into block input

        # embeddings: x = emb[idx] + pos  → scatter-add dx over idx
        idx = cache["idx"]
        flat_g = G["emb"]
        np.add.at(flat_g, idx, dx)
        T_pos = min(T, P["pos"].shape[0])
        G["pos"][:T_pos] += dx[:, :T_pos].sum(axis=0)
        return G

    # --------------------------- train step ------------------------------- #

    def train_step(self, idx: np.ndarray, targets: np.ndarray,
                   lr: float, clip: float = 1.0) -> float:
        _, loss, cache = self.forward(idx, targets)
        G = self.backward(cache, targets)
        # global grad clip
        total = 0.0
        for g in G.values():
            total += float(np.sum(g * g))
        norm = np.sqrt(total)
        if norm > clip and norm > 0:
            for k in G:
                G[k] *= np.float32(clip / norm)
        self.adam.step(self.P, G, lr=lr)
        return loss

    # ---------------------------- generate -------------------------------- #

    def generate(self, idx: np.ndarray, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 40, newline_id: int | None = None,
                 stop_line_starts: tuple[str, ...] | None = None,
                 itos: list[str] | None = None) -> np.ndarray:
        """Autoregressive sampling with optional per-line boundary stopping.

        When `newline_id` and `stop_line_starts` are given, sampling stops as
        soon as a completed line (terminated by the newline token) begins with
        one of the stop prefixes — e.g. 'action' or 'finish'. This lets the
        agent interleave real tool observations instead of hallucinating them.
        """
        out = idx
        line_start = idx.shape[1]
        stop_set = set(stop_line_starts or ())
        for _ in range(max_new_tokens):
            window = out[:, -self.cfg["seq_len"]:]
            logits, _, _ = self.forward(window, None)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k and top_k < logits.shape[-1]:
                kth = np.partition(logits, -top_k, axis=-1)[..., -top_k]
                logits = np.where(logits < kth[..., None], -1e9, logits)
            probs = softmax_rows(logits)
            nxt = int(np.random.choice(probs.shape[-1], p=probs[0]))
            out = np.concatenate(
                [out, np.array([[nxt]], dtype=np.int32)], axis=1)
            if newline_id is not None and nxt == newline_id and itos is not None:
                line = " ".join(itos[int(i)] for i in out[0, line_start:]).strip()
                first = line.split(" ", 1)[0] if line else ""
                if first in stop_set:
                    break
                line_start = out.shape[1]
        return out

    # ------------------------- save / load -------------------------------- #

    def num_params(self) -> int:
        return int(sum(v.size for v in self.P.values()))

    def save(self, path: str, meta: dict | None = None):
        payload = {f"p_{k}": v for k, v in self.P.items()}
        for k, v in self.cfg.items():
            payload[f"cfg_{k}"] = np.asarray(v)
        if meta:
            for k, v in meta.items():
                payload[f"meta_{k}"] = np.asarray(v)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "TinyGPT":
        z = np.load(path, allow_pickle=False)
        cfg = {}
        for k in z.files:
            if k.startswith("cfg_"):
                cfg[k[4:]] = int(z[k])
        model = cls(**cfg)
        for k in z.files:
            if k.startswith("p_"):
                model.P[k[2:]] = z[k].astype(np.float32)
        model.cfg.update(cfg)
        return model


def z_dim(d_model: int, cfg: dict) -> int:
    return cfg["d_ff"]
