"""Baseline uncertainty methods on the frozen Banking77 DistilBERT (no weight updates).

  * deterministic checkpoint: MSP, predictive entropy, Energy
  * MC Dropout: 20 stochastic passes through the model's existing dropout
  * uncorrected internal perturbation: perturb final lin1, leave lin2 unchanged
  * head P&C: perturb pre_classifier, correct classifier (shallow analogue)

All produce per-example uncertainty scores (higher = more OOD) + logits, so the same
metric code as internal-FFN P&C applies.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from construct import perturbation_basis, member_coefficients, base_scale, member_dW1
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "scripts"))
from pnc_theory.linalg import ridge_solve                    # noqa: E402


def softmax_np(z, T=1.0):
    z = z / T; z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e / e.sum(-1, keepdims=True)


def deterministic_scores(logits, T=1.0):
    """MSP / entropy / energy from a single set of logits (B,77)."""
    p = softmax_np(logits, T)
    ent = -np.sum(p * np.log(p + 1e-12), -1)
    energy = -T * jax.scipy.special.logsumexp(jnp.asarray(logits) / T, axis=-1)
    return {"msp_uncertainty": 1.0 - p.max(-1), "predictive_entropy": ent,
            "energy": np.asarray(energy), "probs": p}


def mc_dropout_logits(model, params, input_ids, attention_mask, n_passes=20, seed=0, bs=32):
    """n stochastic forward passes with the model's existing dropout (train=True).
    Returns (n, B, 77). Batched over examples so the full-model attention (B x heads x
    T x T) never blows up GPU memory on large splits. LayerNorm stays deterministic."""
    N = len(input_ids)
    outs = []
    for i in range(n_passes):
        chunks = []
        for j in range(0, N, bs):
            rng = jax.random.PRNGKey(seed * 100000 + i * 1000 + j)   # distinct per pass+chunk
            o = model(jnp.asarray(input_ids[j:j + bs]),
                      attention_mask=jnp.asarray(attention_mask[j:j + bs]),
                      params=params, dropout_rng=rng, train=True)
            chunks.append(np.asarray(o.logits))
        outs.append(np.concatenate(chunks, 0))
    return np.stack(outs, 0)


def build_uncorrected(ad, seed, scale_mult, M=20, K=20, target_rel=0.5):
    """Perturb lin1 only (no lin2 correction) — the correction ablation."""
    U = perturbation_basis(seed, K); coeffs = member_coefficients(seed, M, K)
    W1n = np.linalg.norm(np.asarray(ad.W1))
    scale = base_scale(U, coeffs, W1n) * scale_mult
    return [{"coeff": coeffs[m]} for m in range(M)], U, scale


def uncorrected_logits(ad, h0, members, U, scale):
    outs = []
    for mem in members:
        W1v = ad.W1 + member_dW1(U, mem["coeff"], scale)
        outs.append(np.asarray(ad.tail(h0, W1=W1v)))          # perturbed lin1, ORIGINAL lin2
    return np.stack(outs, 0)


def build_head_pnc(ad, OUT, seed, scale_mult, M=20, K=20, ridge=1e-3, target_rel=0.5):
    """Head P&C: perturb pre_classifier.kernel, correct classifier on ID block outputs.
    OUT: (N,768) layer-5 block outputs (pre_classifier inputs) on calibration data."""
    Din = 768; Dpc = Din * Din
    rng = np.random.RandomState(seed)
    U = rng.normal(size=(K, Dpc)).astype(np.float32); U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    coeffs = member_coefficients(seed, M, K)
    PCw = np.asarray(ad.pc_w); PCn = np.linalg.norm(PCw)
    rels = [np.linalg.norm(coeffs[m] @ U) / PCn for m in range(M)]
    scale = float(target_rel / (np.median(rels) + 1e-12)) * scale_mult
    # target pre-softmax head output for correction: relu(pre_classifier(OUT)) -> classifier
    OUT = jnp.asarray(OUT)
    pooled0 = jax.nn.relu(OUT @ ad.pc_w + ad.pc_b)
    Zhead = np.asarray(pooled0 @ ad.cl_w + ad.cl_b)          # (N,77) target logits
    Theta = np.concatenate([np.asarray(ad.cl_b)[None, :], np.asarray(ad.cl_w)], 0)  # [b; W]
    N = len(OUT)
    members = []
    for m in range(M):
        dPC = (scale * (coeffs[m] @ U)).reshape(Din, Din)
        pooled = np.asarray(jax.nn.relu(OUT @ (ad.pc_w + jnp.asarray(dPC)) + ad.pc_b))
        idx = np.random.RandomState(seed * 1000 + m).randint(0, N, N)
        Pa = np.concatenate([np.ones((N, 1), np.float32), pooled[idx]], 1)
        Th = ridge_solve(Pa, Zhead[idx], ridge, w_prior=Theta)
        members.append({"dPC": dPC.astype(np.float32), "cl_b": np.asarray(Th[0], np.float32),
                        "cl_w": np.asarray(Th[1:], np.float32)})
    return members, scale


def head_pnc_logits(ad, out_block, members):
    """out_block: (B,768) layer-5 block outputs. Apply perturbed head per member."""
    outs = []
    for mem in members:
        pc_w = ad.pc_w + jnp.asarray(mem["dPC"])
        outs.append(np.asarray(ad.head_forward(out_block, pc_w=pc_w,
                                               cl_w=jnp.asarray(mem["cl_w"]), cl_b=jnp.asarray(mem["cl_b"]))))
    return np.stack(outs, 0)
