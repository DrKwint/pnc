"""Linearized SLL prediction: probit rule (primary) + Monte-Carlo posterior predictive (secondary).

Given base logits z(x), selected Jacobians J_S(x) [C,S], and posterior perturbations dW [S,M]:
  sampled linearized logit changes  dz_m(x) = J_S(x) dW_m
  PROBIT (primary):  v_c = Var_m[dz_{m,c}];  ztilde_c = z_c / sqrt(1 + (pi/8) v_c);  softmax(ztilde/T)
  MC (secondary):    p = mean_m softmax((z + dz_m)/T)  -> entropy, expected entropy, MI, max prob
Predictive entropy of the probit probabilities is the primary OOD score. J_S is lambda-independent,
so the prior grid is swept cheaply from one Jacobian pass.
"""
from __future__ import annotations

import numpy as np
import jax, jax.numpy as jnp

PI_8 = np.pi / 8.0


def _softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def selected_jacobians(fns, X, batch: int = 250):
    """J_S [N,C,S] and base logits [N,C]. Per-example jacrev (vmapping jacrev over a batch OOMs);
    base logits are still batched. `batch` only controls the base-logits forward batching."""
    sel_jac = fns["sel_jac"]; base_logits = fns["base_logits"]; w0 = fns["w_sub0"]
    Xd = jnp.asarray(X)
    Js = [np.asarray(sel_jac(w0, Xd[i])) for i in range(len(X))]           # each (C,S)
    Zs = [np.asarray(base_logits(Xd[s:s + batch])) for s in range(0, len(X), batch)]
    return np.stack(Js, 0), np.concatenate(Zs, 0)


def probit_logits_from(J, Z, dW):
    """z_c / sqrt(1 + pi/8 * Var_m[J dW]) ; returns (probit_logits[N,C], v[N,C])."""
    dz = np.einsum("ncs,sm->ncm", J.astype(np.float32), dW.astype(np.float32))  # (N,C,M)
    v = dz.var(axis=2)                                                          # (N,C)
    return Z / np.sqrt(1.0 + PI_8 * v), v, dz


def mc_stats(Z, dz, T):
    """Monte-Carlo posterior predictive from sampled linearized logits (Z + dz_m)/T."""
    M = dz.shape[2]
    logits = (Z[:, :, None] + dz) / T                     # (N,C,M)
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits); p_m = e / e.sum(1, keepdims=True)  # per-sample softmax (N,C,M)
    p = p_m.mean(2)                                        # (N,C)
    ent = -(p * np.log(p + 1e-12)).sum(1)                  # predictive entropy
    exp_ent = -(p_m * np.log(p_m + 1e-12)).sum(1).mean(1)  # expected entropy
    mi = ent - exp_ent
    return dict(mc_probs=p, mc_entropy=ent, expected_entropy=exp_ent, mutual_information=mi,
                mc_max_prob=p.max(1))


def predict_from_jac(J, Z, dW, T, with_mc=True):
    """Full prediction bundle from precomputed J,Z. Returns per-example arrays."""
    pl, v, dz = probit_logits_from(J, Z, dW)
    probs = _softmax(pl / T)
    ent = -(probs * np.log(probs + 1e-12)).sum(1)
    out = dict(base_logits=Z, probit_logits=pl, probit_probs=probs, predictive_entropy=ent, v=v)
    if with_mc:
        out.update(mc_stats(Z, dz, T))
    return out


def predict_streaming(fns, X, dW, T, batch: int = 32, with_mc=True):
    """Batch-streamed prediction for large OOD sets (never stores full J). Returns per-example arrays."""
    keys = ["base_logits", "probit_logits", "probit_probs", "predictive_entropy"]
    if with_mc:
        keys += ["mc_entropy", "expected_entropy", "mutual_information", "mc_max_prob"]
    acc = {k: [] for k in keys}
    for s in range(0, len(X), batch):
        J, Z = selected_jacobians(fns, X[s:s + batch], batch=batch)
        o = predict_from_jac(J, Z, dW, T, with_mc=with_mc)
        for k in keys:
            acc[k].append(o[k])
    return {k: np.concatenate(v, 0) for k, v in acc.items()}


def fit_temperature(probit_logits, y):
    """Scalar T on ID-val via the shared golden-section NLL protocol (matches SCOD/benchmark)."""
    from util import _fit_posthoc_temperature
    return float(_fit_posthoc_temperature(jnp.asarray(probit_logits[None]),
                                          jnp.asarray(np.asarray(y).astype(int))))


def id_metrics(probs, y):
    y = np.asarray(y).astype(int); eps = 1e-12
    acc = float((probs.argmax(1) == y).mean()) * 100
    nll = float(-np.log(probs[np.arange(len(y)), y] + eps).mean())
    conf = probs.max(1); pred = probs.argmax(1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, 16); ece = 0.0
    for i in range(15):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    onehot = np.zeros_like(probs); onehot[np.arange(len(y)), y] = 1.0
    brier = float(((probs - onehot) ** 2).sum(1).mean())
    return dict(accuracy=acc, nll=nll, ece=float(ece), brier=brier)
