"""Numerical gradient check for the from-scratch NumPy backprop.

Verifies the hand-derived gradients of every major parameter group against
central finite differences on a tiny model. If any derivative were wrong,
this fails.
"""
import unittest

import numpy as np

from lumo.model import TinyGPT


def _loss(model, idx, tgt):
    _, loss, _ = model.forward(idx, tgt)
    return loss


class TestBackprop(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.model = TinyGPT(vocab_size=23, seq_len=7, d_model=16,
                             n_head=2, n_layer=2, d_ff=24, seed=1)
        idx = rng.integers(0, 23, size=(2, 7)).astype(np.int32)
        tgt = rng.integers(0, 23, size=(2, 7)).astype(np.int32)
        _, _, cache = self.model.forward(idx, tgt)
        self.grads = self.model.backward(cache, tgt)
        self.idx, self.tgt = idx, tgt

    def test_gradients_match_finite_differences(self):
        rng = np.random.default_rng(42)
        params_to_check = ["emb", "pos", "l0.wq", "l0.wo", "l0.w1", "l0.w2",
                           "l0.ln1.g", "l0.ln1.b", "l1.bk", "l1.ln2.g",
                           "lnf.g", "lnf.b"]
        eps = 5e-4
        for name in params_to_check:
            p = self.model.P[name]
            analytic = self.grads[name]
            flat_idx = rng.choice(p.size, size=4, replace=False)
            for f in list(flat_idx) + [np.argmax(np.abs(analytic))]:
                f = int(f)
                orig = float(p.flat[f])
                p.flat[f] = orig + eps
                lp = _loss(self.model, self.idx, self.tgt)
                p.flat[f] = orig - eps
                lm = _loss(self.model, self.idx, self.tgt)
                p.flat[f] = orig
                numeric = (lp - lm) / (2 * eps)
                a = float(analytic.flat[f])
                denom = max(1e-4, abs(a) + abs(numeric))
                rel = abs(a - numeric) / denom
                self.assertLess(
                    rel, 5e-2,
                    f"{name}[{f}]: analytic={a:.6g} numeric={numeric:.6g} "
                    f"rel={rel:.3g}")

    def test_loss_decreases_with_training(self):
        rng = np.random.default_rng(7)
        model = TinyGPT(vocab_size=23, seq_len=7, d_model=16,
                        n_head=2, n_layer=2, d_ff=24, seed=2)
        idx = rng.integers(0, 23, size=(4, 7)).astype(np.int32)
        tgt = rng.integers(0, 23, size=(4, 7)).astype(np.int32)
        first = _loss(model, idx, tgt)
        for _ in range(40):
            model.train_step(idx, tgt, lr=3e-3)
        last = _loss(model, idx, tgt)
        self.assertLess(last, first * 0.8,
                        f"loss did not decrease: {first} → {last}")

    def test_generate_produces_valid_ids(self):
        out = self.model.generate(np.asarray([[1, 2, 3]], dtype=np.int32),
                                  max_new_tokens=10, temperature=0.8,
                                  top_k=10, newline_id=4,
                                  stop_line_starts=("finish",),
                                  itos=[str(i) for i in range(23)])
        self.assertEqual(out.shape, (1, 13))
        self.assertTrue(((out >= 0) & (out < 23)).all())

    def test_save_load_roundtrip(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.npz")
            self.model.save(path, meta={"step": 5})
            loaded = TinyGPT.load(path)
        for k in self.model.P:
            np.testing.assert_array_equal(self.model.P[k], loaded.P[k])
        self.assertEqual(loaded.cfg, self.model.cfg)


if __name__ == "__main__":
    unittest.main()
