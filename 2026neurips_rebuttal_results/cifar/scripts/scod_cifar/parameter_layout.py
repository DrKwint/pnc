"""Model loading, parameter split, flat-vector layout, and the shared Fisher-weighted
Jacobian factory used by both the sketch builder and the test-time scorer.

SCOD (JAX/Flax adaptation) differentiates the temperature-scaled, Fisher-weighted logits
w.r.t. ALL classifier parameters that influence the logits: conv kernels, BatchNorm affine
(scale gamma / bias beta), and the final linear weight+bias. BatchNorm running mean/var are
frozen buffers (nnx.BatchStat) and are NOT differentiated. The model is always evaluated with
use_running_average=True so no BN state is mutated (default is False = training mode, which
would corrupt the frozen classifier -- see Section 4 / gate 8).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.flatten_util import ravel_pytree

from models import PreActResNet18
from experiments.scod_cifar.categorical_fisher import fisher_weighted_logits

REPO = Path(__file__).resolve().parents[2]
CKPT_TMPL = ("results/cifar10/preact_resnet18_train_e300_optsgd_lr1e-01_wd5e-04_bs128_"
             "wu5_mom0p9_n1_augfcco8_ls0_seed{seed}.pkl")
N_CLASSES = 10


def checkpoint_path(seed: int) -> Path:
    return REPO / CKPT_TMPL.format(seed=seed)


def load_base_model(seed: int) -> PreActResNet18:
    """Load a submitted base PreActResNet18 checkpoint (frozen classifier)."""
    import pickle
    model = PreActResNet18(n_classes=N_CLASSES, rngs=nnx.Rngs(seed))
    with open(checkpoint_path(seed), "rb") as f:
        ckpt = pickle.load(f)
    nnx.update(model, ckpt["state"])
    return model


def split_params(model: PreActResNet18):
    """Split into (graphdef, params, rest). `params` = all nnx.Param (differentiated);
    `rest` = frozen BatchStat (running mean/var). Returns also a stable flat layout."""
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    flat_w, unravel = ravel_pytree(params)
    layout = _build_layout(params)
    return graphdef, params, rest, flat_w, unravel, layout


def _build_layout(params) -> list[dict]:
    """Ordered [{name, shape, offset, numel, included, reason}] matching ravel_pytree order.

    ravel_pytree flattens by JAX's canonical (sorted) tree-leaf order. We reproduce that order
    with jax.tree_util.tree_flatten_with_path so offsets line up with the flat vector exactly.
    """
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    layout, offset = [], 0
    for path, leaf in leaves_with_path:
        name = ".".join(_key_str(k) for k in path)
        if name.endswith(".value"):
            name = name[: -len(".value")]
        numel = int(np.prod(leaf.shape))
        layout.append(dict(name=name, shape=list(leaf.shape), offset=offset,
                           numel=numel, included=True,
                           reason="influences logits (conv kernel / BN affine / fc weight-bias)"))
        offset += numel
    return layout


def _key_str(k) -> str:
    for attr in ("key", "name", "idx"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def make_ztilde_fn(graphdef, rest, unravel, temperature: float) -> Callable:
    """Return ztilde(flat_w, x1) -> (C,): Fisher-weighted, temperature-scaled logits for one
    image (non-jitted; compose into jvp/jacrev/fused kernels). BN frozen (no state mutation)."""
    T = float(temperature)

    def ztilde(flat_w, x1):
        model = nnx.merge(graphdef, unravel(flat_w), rest)
        z = model(x1[None], use_running_average=True)[0] / T
        return fisher_weighted_logits(z)

    return ztilde


def make_ztilde_jac_fn(graphdef, rest, unravel, temperature: float) -> Callable:
    """Return jitted jac(flat_w, x1) -> (C, P): Jacobian of the Fisher-weighted, temperature-
    scaled logits at a single image x1 (H,W,C), w.r.t. the flat parameter vector.

    G = d ztilde / d w  satisfies  G^T G = J_z^T (diag(p)-p p^T) J_z  (categorical Fisher).
    """
    T = float(temperature)

    def ztilde(flat_w, x1):
        model = nnx.merge(graphdef, unravel(flat_w), rest)
        z = model(x1[None], use_running_average=True)[0] / T
        return fisher_weighted_logits(z)

    @jax.jit
    def jac(flat_w, x1):
        return jax.jacrev(ztilde)(flat_w, x1)  # (C, P)

    return jac


def make_logits_fn(graphdef, rest, unravel, temperature: float = 1.0) -> Callable:
    """Return jitted f(flat_w, x_batch) -> (B, C) temperature-scaled logits (frozen BN)."""
    T = float(temperature)

    @jax.jit
    def f(flat_w, x):
        model = nnx.merge(graphdef, unravel(flat_w), rest)
        return model(x, use_running_average=True) / T

    return f


def write_parameter_layout(seed: int, out_dir: Path) -> dict:
    """Compute and save the flat-parameter layout JSON for a checkpoint seed."""
    model = load_base_model(seed)
    _, _, _, flat_w, _, layout = split_params(model)
    P = int(flat_w.shape[0])
    doc = dict(seed=seed, total_params=P, num_leaves=len(layout),
               parameter_set="all_logit_influencing (conv kernels + BN affine + fc weight/bias)",
               excluded=dict(reason="BatchNorm running mean/var are frozen buffers (nnx.BatchStat), "
                                    "not differentiated; no integer counters / disconnected params",
                             names=["*.bn*.mean", "*.bn*.var", "final_bn.mean", "final_bn.var"]),
               layout=layout)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seed}_parameter_layout.json"
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return dict(path=str(path), total_params=P, num_leaves=len(layout))


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    info = write_parameter_layout(seed, REPO / "results" / "scod_cifar" / "configs")
    print(f"[parameter_layout] seed{seed}: P={info['total_params']} leaves={info['num_leaves']}")
    print(f"  -> {info['path']}")
