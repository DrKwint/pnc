"""Finite-scale multi-layer P&C engine (amendment Round H).

Pure float64 reproduction of the sequential two-stage P&C on the 4×200 ReLU MLP:
  stage a = perturb l1 (W0+dWa), correct at l2;  stage b = perturb l3 (W2+dWb),
  correct at l4. Corrections are ridge/LS fits on the calibration set following the
  exact sequential path. Supports the interventional network variants needed for
  re-repair and interaction analysis:

    F_0, F_a(va), F_b(vb), F_ab(va,0), F_ab(0,vb), F_ab(va,vb)

and returns, per eval input, the final mean prediction plus the l4-preactivation
deviation from base (for re-repair survival / rotation). F_ab(0,0)==F_0 exactly
(zero perturbation ⇒ LS correction reproduces the original affine).
"""
from __future__ import annotations
import numpy as np
from experiments.scripts.pnc_theory.linalg import ridge_solve


def _relu(x):
    return np.maximum(x, 0.0)


class GeneralTwoStage:
    """Two perturb-correct stages at arbitrary layers (la<lb≤2): perturb la /
    correct la+1, then perturb lb / correct lb+1. Handles the OVERLAPPING case
    lb==la+1 (stage b perturbs the *corrected* weights of layer la+1). Correction
    targets are the original base preactivations (base-target). Returns per-member
    (mean, var). float64."""

    def __init__(self, base, X_sub, la, lb, lam=0.0, toward_orig=True):
        self.W = [np.asarray(base.layers[i].kernel.get_value(), np.float64) for i in range(4)]
        self.b = [np.asarray(base.layers[i].bias.get_value(), np.float64) for i in range(4)]
        self.Wm = np.asarray(base.mean_layer.kernel.get_value(), np.float64)
        self.bm = np.asarray(base.mean_layer.bias.get_value(), np.float64)
        self.Wv = np.asarray(base.var_layer.kernel.get_value(), np.float64)
        self.bv = np.asarray(base.var_layer.bias.get_value(), np.float64)
        assert la < lb <= 2, "need la<lb<=2 (correct hidden layers only)"
        self.la, self.lb = int(la), int(lb)
        self.overlap = (lb == la + 1)
        self.lam = float(lam); self.toward_orig = bool(toward_orig)
        self.X_sub = np.asarray(X_sub, np.float64)
        self._acts = self._orig(self.X_sub)

    def _orig(self, X):
        a = [np.asarray(X, np.float64)]; z = [None]
        h = a[0]
        for k in range(4):
            zk = h @ self.W[k] + self.b[k]; z.append(zk); h = _relu(zk); a.append(h)
        return {"a": a, "z": z}   # a[k]=input to layer k (a[0]=X); z[k+1]=preact of layer k

    def _prior(self, li):
        return (np.concatenate([self.W[li], self.b[li][None, :]], axis=0)
                if (self.toward_orig and self.lam > 0) else None)

    def fit(self, dWa, dWb):
        la, lb = self.la, self.lb
        a = self._acts["a"]; z = self._acts["z"]
        # stage a on calibration
        hp_a = _relu(a[la] @ (self.W[la] + dWa) + self.b[la])
        Wca = ridge_solve(_aug(hp_a), z[la + 2], self.lam, self._prior(la + 1))  # z[la+2]=preact of layer la+1
        if self.overlap:
            # stage b perturbs the CORRECTED layer la+1 (weights Wca)
            hp_b = _relu(hp_a @ (Wca[:-1] + dWb) + Wca[-1])
            Wcb = ridge_solve(_aug(hp_b), z[lb + 2], self.lam, self._prior(lb + 1))
        else:
            h = _relu(hp_a @ Wca[:-1] + Wca[-1])
            for k in range(la + 2, lb):
                h = _relu(h @ self.W[k] + self.b[k])
            hp_b = _relu(h @ (self.W[lb] + dWb) + self.b[lb])
            Wcb = ridge_solve(_aug(hp_b), z[lb + 2], self.lam, self._prior(lb + 1))
        return Wca, Wcb

    def predict_member(self, X, dWa, Wca, dWb, Wcb):
        la, lb = self.la, self.lb
        o = self._orig(X); a = o["a"]
        hp_a = _relu(a[la] @ (self.W[la] + dWa) + self.b[la])
        if self.overlap:
            hp_b = _relu(hp_a @ (Wca[:-1] + dWb) + Wca[-1])
            h = _relu(hp_b @ Wcb[:-1] + Wcb[-1])
        else:
            h = _relu(hp_a @ Wca[:-1] + Wca[-1])
            for k in range(la + 2, lb):
                h = _relu(h @ self.W[k] + self.b[k])
            hp_b = _relu(h @ (self.W[lb] + dWb) + self.b[lb])
            h = _relu(hp_b @ Wcb[:-1] + Wcb[-1])
        for k in range(lb + 2, 4):
            h = _relu(h @ self.W[k] + self.b[k])
        mean = h @ self.Wm + self.bm
        var = np.logaddexp(0.0, h @ self.Wv + self.bv) + 1e-6
        return mean, var


def _aug(h):
    return np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)


class MultiStage:
    """Two-stage P&C on a 4×200 ReLU MLP, float64."""

    def __init__(self, base, X_sub, lam=0.0, toward_orig=True):
        self.W = [np.asarray(base.layers[i].kernel.get_value(), np.float64) for i in range(4)]
        self.b = [np.asarray(base.layers[i].bias.get_value(), np.float64) for i in range(4)]
        self.Wm = np.asarray(base.mean_layer.kernel.get_value(), np.float64)
        self.bm = np.asarray(base.mean_layer.bias.get_value(), np.float64)
        self.Wv = np.asarray(base.var_layer.kernel.get_value(), np.float64)
        self.bv = np.asarray(base.var_layer.bias.get_value(), np.float64)
        self.lam = float(lam)
        self.toward_orig = bool(toward_orig)
        self.X_sub = np.asarray(X_sub, np.float64)
        self._cal = self._orig_acts(self.X_sub)

    # original forward, returns intermediate acts
    def _orig_acts(self, X):
        W, b = self.W, self.b
        a0 = np.asarray(X, np.float64)
        z1 = a0 @ W[0] + b[0]; a1 = _relu(z1)
        z2 = a1 @ W[1] + b[1]; a2 = _relu(z2)
        z3 = a2 @ W[2] + b[2]; a3 = _relu(z3)
        z4 = a3 @ W[3] + b[3]; a4 = _relu(z4)
        mean = a4 @ self.Wm + self.bm
        return dict(a0=a0, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, a3=a3, z4=z4, a4=a4, mean=mean)

    def _prior(self, layer_idx):
        if self.toward_orig and self.lam > 0:
            return np.concatenate([self.W[layer_idx], self.b[layer_idx][None, :]], axis=0)
        return None

    # ---- fit the two corrections on the calibration set for a given (dWa,dWb) ----
    def fit(self, dWa, dWb, stage_b=True, incremental=False):
        """incremental=False (base-target): stage-b target = ORIGINAL base l4
        preactivation (repairs all upstream change — the shipped behavior).
        incremental=True: stage-b target = l4 preactivation of the stage-a-
        corrected path with stage b UNperturbed (repairs only the new stage-b
        change, preserving stage-a's residual)."""
        c = self._cal
        # stage a: perturb l1, target = orig l2 preact (z2)
        hp_a = _relu(c["a0"] @ (self.W[0] + dWa) + self.b[0])
        Wca = ridge_solve(_aug(hp_a), c["z2"], self.lam, self._prior(1))
        a2_corr = _relu(hp_a @ Wca[:-1] + Wca[-1])          # stage-a-corrected post-l2 (cal)
        Wcb = None
        if stage_b:
            hp_b = _relu(a2_corr @ (self.W[2] + dWb) + self.b[2])
            if incremental:
                # target = l4 preact along stage-a-corrected path, stage-b unperturbed
                tgt = _relu(a2_corr @ self.W[2] + self.b[2]) @ self.W[3] + self.b[3]
                prior = None  # no clean base prior for the incremental target
            else:
                tgt = c["z4"]
                prior = self._prior(3)
            Wcb = ridge_solve(_aug(hp_b), tgt, self.lam, prior)
        return Wca, Wcb

    # ---- eval forward with fitted corrections ----
    def forward(self, X, dWa, Wca, dWb=None, Wcb=None, stage_b=True):
        o = self._orig_acts(X)
        W, b = self.W, self.b
        hp_a = _relu(o["a0"] @ (W[0] + dWa) + b[0])
        a2c = _relu(hp_a @ Wca[:-1] + Wca[-1])              # after stage a correction (post-l2)
        if stage_b:
            hp_b = _relu(a2c @ (W[2] + dWb) + b[2])
            z4c = hp_b @ Wcb[:-1] + Wcb[-1]                 # corrected l4 preact
            a4 = _relu(z4c)
            mean = a4 @ self.Wm + self.bm
            z4_dev = z4c - o["z4"]                          # l4-preact deviation from base
        else:
            # original l3, l4 downstream of stage-a correction
            a3 = _relu(a2c @ W[2] + b[2])
            z4 = a3 @ W[3] + b[3]
            a4 = _relu(z4)
            mean = a4 @ self.Wm + self.bm
            z4_dev = z4 - o["z4"]
        var = np.logaddexp(0.0, a4 @ self.Wv + self.bv) + 1e-6   # softplus aleatoric head
        return dict(mean=mean, z4_dev=z4_dev, var=var)

    # ---- convenience variant builders (fit + forward) ----
    def F0(self, X):
        return self._orig_acts(X)["mean"]

    def F_a(self, X, dWa):
        Wca, _ = self.fit(dWa, None, stage_b=False)
        return self.forward(X, dWa, Wca, stage_b=False)

    def F_ab(self, X, dWa, dWb):
        Wca, Wcb = self.fit(dWa, dWb, stage_b=True)
        return self.forward(X, dWa, Wca, dWb, Wcb, stage_b=True)


class SingleLayerPnC:
    """Single perturb-correct layer at arbitrary position li (correct li+1),
    running the original network before and after. Supports li in {0,1,2}
    (hidden→hidden correction). Returns per-member (mean, var) so ensemble
    disagreement / AUROC / NLL can be computed. float64."""

    def __init__(self, base, X_sub, li, lam=0.0, toward_orig=True):
        self.W = [np.asarray(base.layers[i].kernel.get_value(), np.float64) for i in range(4)]
        self.b = [np.asarray(base.layers[i].bias.get_value(), np.float64) for i in range(4)]
        self.Wm = np.asarray(base.mean_layer.kernel.get_value(), np.float64)
        self.bm = np.asarray(base.mean_layer.bias.get_value(), np.float64)
        self.Wv = np.asarray(base.var_layer.kernel.get_value(), np.float64)
        self.bv = np.asarray(base.var_layer.bias.get_value(), np.float64)
        self.li = int(li)
        assert self.li in (0, 1, 2), "single-layer engine supports li in {0,1,2}"
        self.lam = float(lam); self.toward_orig = bool(toward_orig)
        self.X_sub = np.asarray(X_sub, np.float64)

    def _to_input(self, X):
        """Run original layers 0..li-1, return input to layer li."""
        h = np.asarray(X, np.float64)
        for k in range(self.li):
            h = _relu(h @ self.W[k] + self.b[k])
        return h

    def _orig_next_preact(self, h_in):
        """Original preactivation at layer li+1 given input to layer li."""
        a = _relu(h_in @ self.W[self.li] + self.b[self.li])          # orig post-li
        return a @ self.W[self.li + 1] + self.b[self.li + 1]

    def fit(self, dW):
        h_in = self._to_input(self.X_sub)
        hp = _relu(h_in @ (self.W[self.li] + dW) + self.b[self.li])   # perturbed post-li
        tgt = self._orig_next_preact(h_in)                            # orig li+1 preact
        prior = (np.concatenate([self.W[self.li + 1], self.b[self.li + 1][None, :]], axis=0)
                 if (self.toward_orig and self.lam > 0) else None)
        return ridge_solve(_aug(hp), tgt, self.lam, prior)

    def predict_member(self, X, dW, Wc):
        h_in = self._to_input(X)
        hp = _relu(h_in @ (self.W[self.li] + dW) + self.b[self.li])
        h = _relu(hp @ Wc[:-1] + Wc[-1])                             # after correction (post li+1)
        for k in range(self.li + 2, 4):                              # remaining hidden layers
            h = _relu(h @ self.W[k] + self.b[k])
        mean = h @ self.Wm + self.bm
        var = np.logaddexp(0.0, h @ self.Wv + self.bv) + 1e-6        # softplus
        return mean, var

    def displacement_and_residual(self, X, dW, Wc):
        """E‖Δh‖ (perturbed vs orig post-li) and local residual r at li+1."""
        h_in = self._to_input(X)
        a_orig = _relu(h_in @ self.W[self.li] + self.b[self.li])
        hp = _relu(h_in @ (self.W[self.li] + dW) + self.b[self.li])
        z_corr = hp @ Wc[:-1] + Wc[-1]
        z_orig = a_orig @ self.W[self.li + 1] + self.b[self.li + 1]
        return np.linalg.norm(hp - a_orig, axis=1), np.linalg.norm(z_corr - z_orig, axis=1)
