"""Selected-subnetwork representation and Jacobians for SLL (JAX/Flax).

A subnetwork is S coordinates of the frozen flat parameter vector. `logits_from_subnetwork`
scatters the S selected values into a copy of the frozen flat vector, rebuilds the params (cheap
reshape views), and forwards with frozen BN. Differentiating w.r.t. the S-dim `w_sub` (argnums=0)
yields a [C, S] Jacobian directly -- we never materialise a [C, P] full-parameter Jacobian during
full-covariance construction.

Reuses the SCOD-validated model plumbing (parameter_layout.split_params / make_ztilde_fn).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from experiments.scod_cifar.parameter_layout import (
    load_base_model, split_params, N_CLASSES)

# fc occupies the first flat coordinates (fc.bias[0:10], fc.kernel[10:5130]); everything at or
# after this offset is internal backbone (conv kernels + BN affine incl. final_bn).
FC_FLAT_END = 5130


@dataclass
class SubnetworkSpec:
    idx_S: np.ndarray                 # (S,) global flat indices, sorted ascending (deterministic)
    S: int
    per_leaf: list = field(default_factory=list)  # [{name,shape,dtype,within_leaf_indices,sub_offsets}]

    def summary_by_leaf(self) -> dict:
        out = {}
        for e in self.per_leaf:
            out[e["name"]] = dict(selected=len(e["within_leaf_indices"]), leaf_numel=int(np.prod(e["shape"])))
        return out


def candidate_mask(layout, include_fc: bool) -> np.ndarray:
    """Boolean (P,) mask of candidate coordinates. Backbone = exclude fc.{kernel,bias}."""
    P = sum(l["numel"] for l in layout)
    mask = np.ones(P, dtype=bool) if include_fc else np.arange(P) >= FC_FLAT_END
    return mask


def build_spec_from_flat_indices(idx_S: np.ndarray, layout) -> SubnetworkSpec:
    """Map global flat indices -> per-leaf (path, shape, dtype, within-leaf idx, sub-vector offsets)."""
    idx_S = np.sort(np.asarray(idx_S).astype(np.int64))
    # layout entries carry offset/numel/shape/name in flat order
    starts = np.array([l["offset"] for l in layout])
    per_leaf = []
    for l in layout:
        lo, hi = l["offset"], l["offset"] + l["numel"]
        sel_mask = (idx_S >= lo) & (idx_S < hi)
        if not sel_mask.any():
            continue
        sub_offsets = np.where(sel_mask)[0]            # positions within idx_S / w_sub
        within = idx_S[sel_mask] - lo                  # flat index inside this leaf
        per_leaf.append(dict(name=l["name"], shape=list(l["shape"]), dtype="float32",
                             within_leaf_indices=within.astype(np.int64),
                             sub_offsets=sub_offsets.astype(np.int64)))
    return SubnetworkSpec(idx_S=idx_S, S=int(len(idx_S)), per_leaf=per_leaf)


def make_subnetwork_fns(seed: int, idx_S: np.ndarray, temperature: float = 1.0):
    """Return a bundle of jitted fns for one checkpoint + subnetwork.

    base_logits(x_batch) -> (B,C)           frozen base logits / T
    sel_jac(w_sub, x1)   -> (C,S)           d logits/T / d w_sub at x1
    w_sub0                                     the frozen selected values (S,)
    """
    model = load_base_model(seed)
    graphdef, params, rest, flat_w, unravel, layout = split_params(model)
    idx = jnp.asarray(np.sort(np.asarray(idx_S).astype(np.int64)))
    frozen_flat = jnp.asarray(flat_w)
    w_sub0 = np.asarray(flat_w)[np.sort(np.asarray(idx_S).astype(np.int64))].astype(np.float32)
    T = float(temperature)

    def _logits(w_sub, x1):
        flat = frozen_flat.at[idx].set(w_sub)
        m = nnx.merge(graphdef, unravel(flat), rest)
        return m(x1[None], use_running_average=True)[0] / T

    from experiments.scod_cifar.categorical_fisher import fisher_weighted_logits

    def _ztilde(w_sub, x1):
        return fisher_weighted_logits(_logits(w_sub, x1))

    @jax.jit
    def sel_jac(w_sub, x1):
        return jax.jacrev(_logits)(w_sub, x1)            # (C, S)

    @jax.jit
    def sel_jac_ztilde(w_sub, x1):
        # Fisher factor: (dztilde/dw)^T (dztilde/dw) = J_S^T (diag(p)-pp^T) J_S = per-example GGN
        return jax.jacrev(_ztilde)(w_sub, x1)            # (C, S)

    @jax.jit
    def base_logits(x_batch):
        m = nnx.merge(graphdef, unravel(frozen_flat), rest)
        return m(x_batch, use_running_average=True) / T

    @jax.jit
    def logits_from_subnetwork(w_sub, x1):
        return _logits(w_sub, x1)

    return dict(base_logits=base_logits, sel_jac=sel_jac, sel_jac_ztilde=sel_jac_ztilde,
                logits_from_subnetwork=logits_from_subnetwork,
                w_sub0=jnp.asarray(w_sub0), graphdef=graphdef, rest=rest, unravel=unravel,
                flat_w=flat_w, layout=layout, idx=idx, temperature=T)
