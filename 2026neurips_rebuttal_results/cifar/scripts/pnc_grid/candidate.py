"""Val-ONLY P&C candidate build + ID-validation evaluation (no OOD loaded).

Builds a single-block P&C ensemble via the submitted `_build_single_block_pnc_ensemble` and scores
it on the ID-validation split, fitting the scalar temperature with the submitted ID-only procedure.
Common random numbers are inherent: the calib subset, K=20 basis, and MxK member coefficients are
all deterministic from the checkpoint `seed`, so all scale x bootstrap candidates within a block
share them; only perturbation_scale and bootstrap_frac vary.
"""
from __future__ import annotations
import time
import numpy as np
import jax, jax.numpy as jnp

from experiments.scod_cifar.parameter_layout import load_base_model
from experiments.scod_cifar.protocol import get_splits
from cifar_tasks import _build_single_block_pnc_ensemble
from util import _fit_posthoc_temperature

FIXED = dict(n_directions=20, n_members=50, lambda_reg=1e-3, subset_size=1024,
             random_directions=True)
BATCH = 250


def _softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def _member_logits(ens, X):
    return np.concatenate([np.asarray(ens.predict(jnp.asarray(X[i:i+BATCH])))
                           for i in range(0, len(X), BATCH)], axis=1)   # (S, N, C)


def _base_logits(model, X):
    return np.concatenate([np.asarray(model(jnp.asarray(X[i:i+BATCH]), use_running_average=True))
                           for i in range(0, len(X), BATCH)], axis=0)   # (N, C)


def _metrics(mixture, y):
    y = np.asarray(y).astype(int); eps = 1e-12
    acc = float((mixture.argmax(1) == y).mean()) * 100
    nll = float(-np.log(mixture[np.arange(len(y)), y] + eps).mean())
    conf = mixture.max(1); pred = mixture.argmax(1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, 16); ece = 0.0
    for i in range(15):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.sum() > 0: ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    oh = np.zeros_like(mixture); oh[np.arange(len(y)), y] = 1.0
    brier = float(((mixture - oh) ** 2).sum(1).mean())
    return dict(accuracy=acc, nll=nll, ece=float(ece), brier=brier)


def load_seed(seed):
    """Load base model + splits once per checkpoint seed (cached by the grid runner)."""
    model = load_base_model(seed)
    x_tr, y_tr, x_va, y_va = get_splits()
    return dict(model=model, x_tr=np.asarray(x_tr), x_va=np.asarray(x_va),
                y_va=np.asarray(y_va).astype(int))


def build_and_eval(seed, stage_idx, block_idx, scale, bootstrap_frac, chunk_size, cache=None):
    """Returns (metrics dict, mixtures dict) with only ID-val touched. `cache` = load_seed(seed)."""
    t0 = time.time()
    c = cache or load_seed(seed)
    model = c["model"]; x_tr = c["x_tr"]; x_va = c["x_va"]; y_va = c["y_va"]
    build_t0 = time.time()
    ens, extras = _build_single_block_pnc_ensemble(
        model, x_tr, target_stage_idx=stage_idx, target_block_idx=block_idx,
        n_directions=FIXED["n_directions"], n_perturbations=FIXED["n_members"],
        perturbation_scale=float(scale), subset_size=FIXED["subset_size"], chunk_size=int(chunk_size),
        lambda_reg=FIXED["lambda_reg"], random_directions=True, seed=int(seed),
        bootstrap_frac=float(bootstrap_frac), bootstrap_seed=int(seed))
    build_secs = time.time() - build_t0

    L = _member_logits(ens, x_va)                              # (S, N, C)
    T = float(_fit_posthoc_temperature(jnp.asarray(L), jnp.asarray(y_va)))
    mix_cal = _softmax(L / T).mean(0)
    mix_unc = _softmax(L).mean(0)
    m_cal = _metrics(mix_cal, y_va); m_unc = _metrics(mix_unc, y_va)
    base_pred = _base_logits(model, x_va).argmax(1)
    base_agree = float((mix_cal.argmax(1) == base_pred).mean())

    metrics = dict(
        seed=int(seed), stage_idx=int(stage_idx), block_idx=int(block_idx),
        scale=float(scale), bootstrap_frac=float(bootstrap_frac), chunk_size=int(chunk_size),
        temperature=T,
        val_nll_calibrated=m_cal["nll"], val_nll_uncalibrated=m_unc["nll"],
        val_accuracy=m_cal["accuracy"], val_ece=m_cal["ece"], val_brier=m_cal["brier"],
        base_agreement=base_agree,
        n_val=int(len(y_va)), build_secs=round(build_secs, 1),
        total_secs=round(time.time() - t0, 1), **{f"fixed_{k}": v for k, v in FIXED.items()})
    mixtures = dict(mix_cal=mix_cal.astype(np.float32), mix_unc=mix_unc.astype(np.float32),
                    y_va=y_va.astype(np.int16), temperature=T)
    return metrics, mixtures
