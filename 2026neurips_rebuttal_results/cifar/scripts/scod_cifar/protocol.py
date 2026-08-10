"""Frozen submitted-protocol helpers shared by the SCOD runners.

Reproduces the EXACT submitted P&C data/split/subset/temperature protocol (audited against
cifar_tasks.py / util.py / data.py) so SCOD is directly comparable to the submitted CIFAR-10 table:

  * checkpoints: results/cifar10/preact_resnet18_train_e300_..._seed{0,1,2}.pkl (key 'state')
  * split:  RandomState(99).permutation(50000) -> val = first 5000, train = remaining 45000
  * 1024 calib subset (per checkpoint seed s): RandomState(s).choice(45000, 1024, replace=False)
  * normalization: (x/255 - mean)/std, mean/std the standard CIFAR values (applied at load)
  * temperature: fit on ID-val via util._fit_posthoc_temperature (golden-section NLL)
  * OpenOOD v1.5: Near = {cifar100, tiny_imagenet}, Far = {mnist, svhn, textures, places365}

NOTE (framework): the SCOD spec is written for PyTorch; this repo is JAX/Flax. The temperature is
fit for the BASE classifier (single model) under the identical _fit_posthoc_temperature protocol,
since SCOD reports base-classifier Acc/NLL and scores base logits (the submitted P&C temperature is
for the ensemble mixture and is not the right scale for the single base net).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from data import load_openood_cifar_benchmark, load_cifar10
from util import _split_data, _fit_posthoc_temperature
from experiments.scod_cifar.parameter_layout import (
    load_base_model, split_params, make_logits_fn, checkpoint_path, REPO)

NEAR = ["cifar100", "tiny_imagenet"]
FAR = ["mnist", "svhn", "textures", "places365"]
SPLIT_SEED = 99
VAL_FRACTION = 0.1
SUBSET_SIZE = 1024
BATCH = 250

_BENCH = None


def load_benchmark():
    """Load the OpenOOD v1.5 CIFAR-10 benchmark (cached in-process)."""
    global _BENCH
    if _BENCH is None:
        _BENCH = load_openood_cifar_benchmark("cifar10")
    return _BENCH


_CIFAR_TEST = None


def get_splits(benchmark=None):
    """(x_tr, y_tr, x_va, y_va) via the submitted _split_data (seed 99, 10% val).

    Uses load_cifar10() directly (identical to benchmark['id_train'] per audit) so the BUILD stage
    never has to load the multi-GB OpenOOD sets into RAM. If a benchmark is passed we use its
    id_train to stay bit-identical to the evaluate stage.
    """
    if benchmark is not None:
        xtr_full = benchmark["id_train"]["inputs"]; ytr_full = benchmark["id_train"]["targets"]
    else:
        xtr_full, ytr_full, _, _ = load_cifar10()
    return _split_data(xtr_full, ytr_full, val_split=VAL_FRACTION, seed=SPLIT_SEED)


def get_id_test():
    """CIFAR-10 test (inputs, targets) via load_cifar10 -- avoids loading OOD in the build stage."""
    global _CIFAR_TEST
    if _CIFAR_TEST is None:
        _, _, xte, yte = load_cifar10()
        _CIFAR_TEST = (np.asarray(xte), np.asarray(yte).astype(int))
    return _CIFAR_TEST


def get_calibration_subset(seed: int, benchmark=None):
    """The exact per-seed 1024 calibration images (RandomState(seed).choice on x_tr). No bootstrap."""
    x_tr, y_tr, _, _ = get_splits(benchmark)
    n = len(x_tr)
    idx = np.random.RandomState(int(seed)).choice(n, min(SUBSET_SIZE, n), replace=False)
    return np.asarray(x_tr[idx]), np.asarray(y_tr[idx]), idx


def _predict_logits(logits_fn, flat_w, X):
    outs = []
    for i in range(0, len(X), BATCH):
        outs.append(np.asarray(logits_fn(flat_w, jnp.asarray(X[i:i + BATCH]))))
    return np.concatenate(outs, 0)


def fit_base_temperature(seed: int, flat_w, graphdef, rest, unravel, benchmark=None):
    """Fit the base-classifier temperature on the ID-val split (identical protocol)."""
    _, _, x_va, y_va = get_splits(benchmark)
    logits_fn = make_logits_fn(graphdef, rest, unravel, temperature=1.0)
    val_logits = _predict_logits(logits_fn, flat_w, x_va)          # (Nval, C)
    T = float(_fit_posthoc_temperature(jnp.asarray(val_logits[None]),  # (1, Nval, C)
                                       jnp.asarray(np.asarray(y_va).astype(int))))
    val_nll = _nll(val_logits / T, np.asarray(y_va).astype(int))
    return T, dict(val_nll=val_nll, n_val=int(len(y_va)))


def _softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def _nll(logits, y):
    p = _softmax(np.asarray(logits)); eps = 1e-12
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + eps)))


def _ece(logits, y, nb=15):
    p = _softmax(np.asarray(logits)); conf = p.max(1); pred = p.argmax(1)
    acc = (pred == y).astype(float); bins = np.linspace(0, 1, nb + 1); e = 0.0
    for i in range(nb):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def base_id_metrics(seed: int, flat_w, graphdef, rest, unravel, T: float, benchmark=None):
    """Base-classifier ID test accuracy / NLL / ECE at temperature T (bookkeeping, Section 14)."""
    logits_fn = make_logits_fn(graphdef, rest, unravel, temperature=1.0)
    if benchmark is not None:
        xte = benchmark["id_test"]["inputs"]; y = np.asarray(benchmark["id_test"]["targets"]).astype(int)
    else:
        xte, y = get_id_test()
    Lid = _predict_logits(logits_fn, flat_w, xte)
    Lt = Lid / T
    acc = float((Lt.argmax(-1) == y).mean()) * 100
    return dict(id_test_acc=acc, id_test_nll=_nll(Lt, y), id_test_ece=_ece(Lt, y),
                id_test_nll_untempered=_nll(Lid, y), n_test=int(len(y)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()


def load_model_only(seed: int):
    """Load model + param split WITHOUT loading any dataset (for the eval stage, to avoid a
    duplicate CIFAR load). Temperature is read separately from the build metadata."""
    model = load_base_model(seed)
    graphdef, params, rest, flat_w, unravel, layout = split_params(model)
    return dict(model=model, graphdef=graphdef, rest=rest, flat_w=flat_w, unravel=unravel,
                P=int(flat_w.shape[0]))


def prepare_seed(seed: int):
    """Load model, split params, fit temperature. Returns a dict of everything downstream needs."""
    model = load_base_model(seed)
    graphdef, params, rest, flat_w, unravel, layout = split_params(model)
    T, tinfo = fit_base_temperature(seed, flat_w, graphdef, rest, unravel)
    return dict(model=model, graphdef=graphdef, params=params, rest=rest, flat_w=flat_w,
                unravel=unravel, layout=layout, temperature=T, temp_info=tinfo, P=int(flat_w.shape[0]))
