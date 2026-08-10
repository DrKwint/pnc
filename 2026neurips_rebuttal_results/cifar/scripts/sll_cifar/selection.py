"""Diagonal-GGN subnetwork selection for SLL (Section 'Subnetwork selection').

Step 1: diagonal GGN over the candidate pool,  G_diag,j = sum_{n,c} (d ztilde_{n,c} / d w_j)^2,
        accumulated in a P-shaped vector without a dense all-parameter GGN. Reuses the SCOD
        Fisher-weighted Jacobian (jac of ztilde, [C,P]) -- diag = sum_c jac[c,:]^2.
Step 2: select the top-S candidate coordinates by LARGEST approximate marginal variance
        1/(G_diag+lambda), i.e. SMALLEST G_diag (deterministic tie-break by flat index).
        Backbone candidates exclude fc.{kernel,bias}. Temperature = 1 for curvature.
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import jax, jax.numpy as jnp

from flax import nnx
from experiments.scod_cifar.parameter_layout import split_params
from experiments.scod_cifar.protocol import load_base_model
from experiments.scod_cifar.categorical_fisher import fisher_weighted_logits
from experiments.sll_cifar.subnetwork import candidate_mask, build_spec_from_flat_indices


def compute_diag_ggn(seed: int, x_calib: np.ndarray, log_every: int = 256):
    """Full-P diagonal GGN (fisher-weighted) AND raw-logit Jacobian energy over the calibration set.

    From ONE raw-logit Jacobian per example J=[C,P] (p=softmax(logits)):
      R_j        = sum_{n,c} J[c,j]^2                        (raw influence / output sensitivity)
      G_diag,j   = sum_{n,c} (sqrt(p_c)(J[c,j]-sum_k p_k J[k,j]))^2   (fisher-weighted GGN diagonal)
    Deriving the fisher factor from the raw J avoids a second backward pass. Returns
    (G_diag[P], R[P], layout, flat_w, seconds)."""
    model = load_base_model(seed)
    graphdef, params, rest, flat_w, unravel, layout = split_params(model)
    P = int(flat_w.shape[0]); N = int(len(x_calib)); Xd = jnp.asarray(x_calib)

    def logits(fw, x1):
        m = nnx.merge(graphdef, unravel(fw), rest)
        return m(x1[None], use_running_average=True)[0]

    @jax.jit
    def contrib(fw, x1):
        J = jax.jacrev(logits)(fw, x1)                          # (C,P) raw-logit Jacobian
        z = logits(fw, x1); p = jax.nn.softmax(z)
        Jt = jnp.sqrt(jnp.clip(p, 1e-12))[:, None] * (J - (p[None, :] @ J))  # (C,P) fisher-weighted
        return jnp.sum(J * J, axis=0), jnp.sum(Jt * Jt, axis=0)  # (R_contrib, Gdiag_contrib)

    R = jnp.zeros((P,), jnp.float32); G = jnp.zeros((P,), jnp.float32)
    fw = jnp.asarray(flat_w); t0 = time.time()
    for i in range(N):
        rc, gc = contrib(fw, Xd[i]); R = R + rc; G = G + gc
        if log_every and i % log_every == 0:
            print(f"  [diag-ggn] {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    return np.asarray(G), np.asarray(R), layout, np.asarray(flat_w), round(time.time() - t0, 1)


def select_subnetwork(diag: np.ndarray, R: np.ndarray, layout, S: int, include_fc: bool = False):
    """Top-S backbone coords by linearized PREDICTIVE-VARIANCE CONTRIBUTION

        score_j = R_j / (G_diag,j + lambda0),   lambda0 = median positive G_diag (candidates),

    where R_j is the raw-logit Jacobian energy (predictive influence) and 1/(G_diag+lambda0) is the
    marginal posterior variance. This is the quantity SLL's Wasserstein-predictive objective targets;
    the pure marginal-variance ranking (smallest G_diag) is a diagonal approximation that DEGENERATES
    here -- it selects predictively-inert near-zero-Jacobian weights (see report). Coordinates with
    zero influence (R_j = 0) are non-identifiable and excluded. Deterministic tie-break by flat index.
    """
    mask = candidate_mask(layout, include_fc)
    cand_idx = np.where(mask)[0]
    cand_R = R[cand_idx]; cand_G = diag[cand_idx]
    n_zero_curv = int(np.sum(cand_G == 0.0)); n_zero_infl = int(np.sum(cand_R == 0.0))
    keep = cand_R > 0.0                                  # need nonzero predictive influence
    kidx = cand_idx[keep]; kR = cand_R[keep]; kG = cand_G[keep]
    lam0 = float(np.median(kG[kG > 0])) if np.any(kG > 0) else 1.0
    score = kR / (kG + lam0)
    order = np.lexsort((kidx, -score))                   # primary: score desc; tie: flat_index asc
    idx_S = np.sort(kidx[order[:S]])
    info = dict(S=int(S), include_fc=bool(include_fc), selection="predictive_variance_contribution",
                lambda0=lam0, n_candidates=int(len(cand_idx)),
                n_zero_curvature_candidates=n_zero_curv,
                zero_curvature_candidate_fraction=float(n_zero_curv / len(cand_idx)),
                n_zero_influence_candidates=n_zero_infl, zero_influence_excluded=True,
                selected_zero_curvature_fraction=float(np.mean(diag[idx_S] == 0.0)),
                selected_diag_min=float(diag[idx_S].min()), selected_diag_max=float(diag[idx_S].max()),
                selected_diag_median=float(np.median(diag[idx_S])),
                selected_R_median=float(np.median(R[idx_S])),
                selected_score_min=float(score[order[:S]].min()),
                selected_score_max=float(score[order[:S]].max()))
    return idx_S, info


def layer_selection_summary(idx_S: np.ndarray, layout) -> dict:
    """Per-leaf count and percentage selected; assert all backbone (no fc)."""
    spec = build_spec_from_flat_indices(idx_S, layout)
    per_leaf = {}
    for e in spec.per_leaf:
        numel = int(np.prod(e["shape"]))
        per_leaf[e["name"]] = dict(selected=len(e["within_leaf_indices"]), leaf_numel=numel,
                                   pct=round(100.0 * len(e["within_leaf_indices"]) / numel, 4))
    has_fc = any(n.startswith("fc.") for n in per_leaf)
    return dict(per_leaf=per_leaf, num_leaves_touched=len(per_leaf), contains_fc=has_fc)


def run_selection(seed: int, x_calib: np.ndarray, S: int, out_dir: Path, include_fc: bool = False,
                  reuse_diag: bool = True):
    """Compute diag GGN (cached per seed), select S, save artifacts. Returns (idx_S, layout)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    diag_path = out_dir / f"diag_ggn_seed{seed}_N{len(x_calib)}.npz"
    if reuse_diag and diag_path.exists() and "R" in np.load(diag_path):
        d = np.load(diag_path); diag = d["diag"]; R = d["R"]; flat_w = d["flat_w"]
        layout = pickle.loads(d["layout"].tobytes()); secs = float(d["seconds"])
        print(f"[select] seed{seed} reuse cached diag GGN", flush=True)
    else:
        diag, R, layout, flat_w, secs = compute_diag_ggn(seed, x_calib)
        np.savez(diag_path, diag=diag, R=R, flat_w=flat_w,
                 layout=np.frombuffer(pickle.dumps(layout), dtype=np.uint8), seconds=secs)
    idx_S, info = select_subnetwork(diag, R, layout, S, include_fc)
    summ = layer_selection_summary(idx_S, layout)
    assert not summ["contains_fc"], "selection must be backbone-only (no fc)"
    tag = "all" if include_fc else "backbone"
    np.savez(out_dir / f"selected_indices_seed{seed}_S{S}_{tag}.npz",
             idx_S=idx_S, selected_values=flat_w[idx_S], selected_diag=diag[idx_S])
    json.dump(dict(seed=seed, tag=tag, diag_seconds=secs, **info, **summ),
              open(out_dir / f"layer_selection_summary_seed{seed}_S{S}_{tag}.json", "w"), indent=2)
    print(f"[select] seed{seed} S={S} {tag}: zero-curv frac={info['selected_zero_curvature_fraction']:.3f} "
          f"leaves touched={summ['num_leaves_touched']} diag[min,med,max]="
          f"[{info['selected_diag_min']:.2e},{info['selected_diag_median']:.2e},{info['selected_diag_max']:.2e}]",
          flush=True)
    return idx_S, layout
