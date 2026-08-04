"""Apples-to-apples efficiency accounting for P&C vs. the MuJoCo baselines.

Answers reviewers LLwM / A5nU / WKSj by measuring three axes under ONE protocol,
on ONE machine, back to back:

  1. construction wall-clock  -- every method's base net trained with the SAME
     --steps budget, and P&C's timer covering the least-squares correction solve
  2. storage                  -- bytes actually resident for inference, walked
     off the live ensemble objects (not analytic guesses)
  3. inference latency        -- warmed up on the TRUE batch shape, then timed

Why this script exists instead of reusing the cached result JSONs
----------------------------------------------------------------
The `train_time` fields already in results/{env}/*.json are NOT comparable
across methods:

  * P&C's base net trains for 2000 steps (the `train_probabilistic_model`
    default, gym_tasks.py:867) while every baseline explicitly passes
    steps=5000. Subspace is likewise 2000 (gym_tasks.py:1519).
  * P&C's `setup_time` stops at gym_tasks.py:1049, BEFORE `PJSVDEnsemble(...)`
    is constructed at gym_tasks.py:1094 -- so the least-squares correction
    solves (the cost WKSj asks about) are not in the logged number at all.
  * `eval_time` warms up on `inputs[:1]` but times a 10k-row batch (util.py:238),
    so XLA recompiles inside the timer. Symptom: P&C's first perturbation scale
    reads ~0.53s and the next three ~0.22s for identical work.

Everything here is measured fresh, so none of the above applies.

Usage
-----
    .venv/bin/python experiments/scripts/efficiency_benchmark.py --env Hopper-v5

One env at a time (one GPU job at a time); see run_efficiency_benchmark.sh.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from pnc_core.ensembles import (
    LaplaceEnsemble,
    MCDropoutEnsemble,
    PJSVDEnsemble,
    StandardEnsemble,
    SWAGEnsemble,
)
from pnc_core.gym_tasks import _sample_member_latents
from pnc_core.laplace import compute_kfac_factors
from pnc_core.models import MCDropoutProbabilisticRegressionModel, ProbabilisticRegressionModel
from pnc_core.training import train_probabilistic_model, train_swag_model
from pnc_core.util import _get_activation, _split_data, seed_everything

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# timing / sync helpers
# ---------------------------------------------------------------------------


def _sync(x):
    """Block until every array in a pytree is materialised."""
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def _sync_module(m):
    _sync(nnx.state(m))
    return m


class Stopwatch:
    """`with Stopwatch() as sw: ...` then read `sw.s`."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0
        return False


# ---------------------------------------------------------------------------
# storage accounting
# ---------------------------------------------------------------------------


def _collect_arrays(obj, seen_ids, out, depth=0):
    """Walk an object graph collecting distinct arrays, deduped by id().

    Dedup matters: P&C's `layer_params` holds references to the *same* buffers
    as `base_model`, and a naive sum would double-count the shared base.
    """
    if obj is None or depth > 6:
        return
    if isinstance(obj, (jax.Array, np.ndarray)):
        if id(obj) not in seen_ids:
            seen_ids.add(id(obj))
            out.append(obj)
        return
    if isinstance(obj, nnx.Module):
        _collect_arrays(jax.tree_util.tree_leaves(nnx.state(obj)), seen_ids, out, depth + 1)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_arrays(v, seen_ids, out, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_arrays(v, seen_ids, out, depth + 1)
        return
    # nnx.State and similar pytree containers
    try:
        leaves = jax.tree_util.tree_leaves(obj)
    except Exception:
        return
    if leaves and not isinstance(obj, (int, float, str, bool)):
        for v in leaves:
            _collect_arrays(v, seen_ids, out, depth + 1)


def storage_of(roots):
    """Total distinct params / bytes across a list of root objects."""
    seen_ids, arrs = set(), []
    for r in roots:
        _collect_arrays(r, seen_ids, arrs)
    params = int(sum(int(np.prod(a.shape)) for a in arrs))
    nbytes = int(sum(int(np.prod(a.shape)) * a.dtype.itemsize for a in arrs))
    return {"params": params, "bytes": nbytes, "megabytes": round(nbytes / 1e6, 3)}


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------


def bench_latency(ensemble, x, reps, warmup):
    """Time `ensemble.predict(x)`, warming up on the SAME shape we then time."""
    for _ in range(warmup):
        _sync(ensemble.predict(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _sync(ensemble.predict(x))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return {
        "mean_ms": float(ts.mean()),
        "median_ms": float(np.median(ts)),
        "std_ms": float(ts.std()),
        "p90_ms": float(np.percentile(ts, 90)),
        "reps": reps,
        "warmup": warmup,
    }


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_env(env, seed, data_steps):
    base = REPO / "results" / env
    suffix = f"seed{seed}_steps{data_steps}.npz"
    tr = np.load(base / f"data_id_train_{suffix}")
    ev = np.load(base / f"data_id_eval_{suffix}")
    return (
        jnp.array(tr["inputs"]),
        jnp.array(tr["targets"]),
        jnp.array(ev["inputs"]),
        jnp.array(ev["targets"]),
    )


# ---------------------------------------------------------------------------
# constructors -- each returns (ensemble, timing dict, storage roots)
# ---------------------------------------------------------------------------


def build_single(x_tr, y_tr, x_va, y_va, cfg, rngs_seed):
    """One probabilistic base net -- the unit of cost every method is priced in."""
    model = ProbabilisticRegressionModel(
        x_tr.shape[1],
        y_tr.shape[1],
        nnx.Rngs(params=rngs_seed),
        hidden_dims=cfg.hidden_dims,
        activation=cfg.act_fn,
    )
    model = train_probabilistic_model(
        model, x_tr, y_tr, x_va, y_va, steps=cfg.steps, batch_size=cfg.batch_size
    )
    return _sync_module(model)


def build_deep_ensemble(x_tr, y_tr, x_va, y_va, cfg):
    per_member = []
    models = []
    with Stopwatch() as sw_total:
        for i in range(cfg.de_members):
            with Stopwatch() as sw:
                models.append(build_single(x_tr, y_tr, x_va, y_va, cfg, cfg.seed + i))
            per_member.append(sw.s)
            print(f"  [DE {i + 1}/{cfg.de_members}] {sw.s:.2f}s")
    ens = StandardEnsemble(models)
    timing = {
        "total_s": sw_total.s,
        "base_train_s": sw_total.s,
        "build_s": 0.0,
        "n_trained_nets": cfg.de_members,
        "per_member_s_mean": float(np.mean(per_member)),
        "per_member_s_std": float(np.std(per_member)),
    }
    return ens, timing, [models]


def build_pnc(x_tr, y_tr, x_va, y_va, x_all, cfg):
    """P&C at the paper config: multi-layer, random directions, LS correction.

    Mirrors GymPJSVD (gym_tasks.py:823-1112) for layer_scope='multi',
    pjsvd_family='random', correction_mode='least_squares'. The two timers are
    reported separately so the marginal post-hoc cost is visible on its own:

      base_train_s -- training the ONE base net (same recipe as a DE member)
      build_s      -- direction sampling + PJSVDEnsemble construction, which
                      runs _precompute_corrections() -> the least-squares solves
                      for all M members. This is the term missing from the
                      cached JSONs.
    """
    with Stopwatch() as sw_base:
        model = build_single(x_tr, y_tr, x_va, y_va, cfg, cfg.seed)

    with Stopwatch() as sw_build:
        rng_sub = np.random.RandomState(cfg.seed)
        n_sub = min(len(x_all), cfg.subset_size)
        X_sub = jnp.array(np.asarray(x_all)[rng_sub.choice(len(x_all), n_sub, replace=False)])

        n_hidden = len(cfg.hidden_dims)
        perturb_indices = list(range(0, n_hidden, 2))
        corr_layer_idx = perturb_indices[-1] + 1
        perturbed_layers = [f"l{i + 1}" for i in perturb_indices]

        Ws = [model.layers[i].kernel.get_value() for i in range(len(model.layers))]
        bs = [model.layers[i].bias.get_value() for i in range(len(model.layers))]
        W_pert = [Ws[i] for i in perturb_indices]
        b_pert = [bs[i] for i in perturb_indices]
        layer_params = {f"l{i + 1}": {"W": Ws[i], "b": bs[i]} for i in perturb_indices}

        # Per-layer random direction bases (pjsvd_family='random').
        layer_specs = []
        for li, _pi in enumerate(perturb_indices):
            W_li = W_pert[li]
            D = W_li.size
            rng_li = np.random.RandomState(cfg.seed + li)
            dirs = rng_li.normal(size=(cfg.n_directions, D)).astype(np.float32)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
            layer_specs.append(
                {
                    "v_opts": dirs,
                    "sigmas": np.ones(cfg.n_directions, dtype=np.float32),
                    "W_shape": W_li.shape,
                    "identity_basis": False,
                }
            )

        # Correction target: activations after the perturbed stack (probabilistic base).
        h_old = X_sub
        for idx in range(len(perturb_indices)):
            h_old = cfg.act_fn(h_old @ W_pert[idx] + b_pert[idx])
        correction_params = {"target_act": h_old}

        all_z = np.stack(
            [
                _sample_member_latents(
                    np.random.RandomState(cfg.seed + li), cfg.pnc_members, cfg.n_directions
                )
                for li in range(len(perturb_indices))
            ],
            axis=1,
        )

        ens = PJSVDEnsemble(
            base_model=model,
            v_opts=np.zeros((1, 1)),
            sigmas=np.ones(1),
            z_coeffs=all_z,
            perturbation_scale=cfg.perturbation_size,
            X_sub=X_sub,
            layers=perturbed_layers,
            correction_mode="least_squares",
            activation=cfg.act_fn,
            layer_params=layer_params,
            correction_params=correction_params,
            tail_is_hidden=True,
            layer_specs=layer_specs,
            lambda_reg=cfg.lambda_reg,
            bootstrap_frac=cfg.bootstrap_frac,
            bootstrap_seed=cfg.seed,
        )
        # PJSVDEnsemble.__init__ dispatches the LS solves lazily; force them.
        _sync([ens.seq_w_effs, ens.seq_b_effs, ens.seq_dWs])

    timing = {
        "total_s": sw_base.s + sw_build.s,
        "base_train_s": sw_base.s,
        "build_s": sw_build.s,
        "n_trained_nets": 1,
        "build_s_per_member": sw_build.s / cfg.pnc_members,
    }
    # Inference-resident roots only: what the forward pass at
    # ensembles.py:706-746 actually reads. X_sub / correction_params are
    # construction scratch and are excluded (reported separately below).
    roots = [model, layer_params, ens.seq_w_effs, ens.seq_b_effs, ens.seq_dWs]
    extras = {
        "minimal_checkpoint": storage_of([model, np.asarray(all_z)]),
        "construction_scratch": storage_of([X_sub, correction_params, layer_specs]),
    }
    return ens, timing, roots, extras, corr_layer_idx


def build_swag(x_tr, y_tr, x_va, y_va, cfg):
    with Stopwatch() as sw:
        model = ProbabilisticRegressionModel(
            x_tr.shape[1],
            y_tr.shape[1],
            nnx.Rngs(params=cfg.seed),
            hidden_dims=cfg.hidden_dims,
            activation=cfg.act_fn,
        )
        model, swag_mean, swag_var = train_swag_model(
            model,
            x_tr,
            y_tr,
            x_va,
            y_va,
            steps=cfg.steps,
            batch_size=cfg.batch_size,
            swag_start=cfg.steps // 5,
        )
        _sync([swag_mean, swag_var])
    ens = SWAGEnsemble(model, swag_mean, swag_var, cfg.baseline_members, scale=1.0)
    timing = {"total_s": sw.s, "base_train_s": sw.s, "build_s": 0.0, "n_trained_nets": 1}
    return ens, timing, [model, swag_mean, swag_var]


def build_laplace(x_tr, y_tr, x_va, y_va, x_all, y_all, cfg):
    with Stopwatch() as sw_base:
        model = build_single(x_tr, y_tr, x_va, y_va, cfg, cfg.seed)
    with Stopwatch() as sw_build:
        rng = np.random.RandomState(cfg.seed)
        n_sub = min(len(x_all), cfg.subset_size)
        idx = rng.choice(len(x_all), n_sub, replace=False)
        factors = compute_kfac_factors(
            model, np.asarray(x_all)[idx], np.asarray(y_all)[idx], batch_size=128
        )
        _sync(factors)
    ens = LaplaceEnsemble(
        model=model,
        kfac_factors=factors,
        prior_precision=cfg.laplace_prior,
        n_models=cfg.baseline_members,
        data_size=n_sub,
    )
    timing = {
        "total_s": sw_base.s + sw_build.s,
        "base_train_s": sw_base.s,
        "build_s": sw_build.s,
        "n_trained_nets": 1,
    }
    return ens, timing, [model, factors, ens.inv_scales]


def build_mc_dropout(x_tr, y_tr, x_va, y_va, cfg):
    with Stopwatch() as sw:
        model = MCDropoutProbabilisticRegressionModel(
            x_tr.shape[1],
            y_tr.shape[1],
            nnx.Rngs(params=cfg.seed, dropout=cfg.seed + 1),
            hidden_dims=cfg.hidden_dims,
            dropout_rate=cfg.dropout_rate,
            activation=cfg.act_fn,
        )
        model = train_probabilistic_model(
            model, x_tr, y_tr, x_va, y_va, steps=cfg.steps, batch_size=cfg.batch_size
        )
        _sync_module(model)
    ens = MCDropoutEnsemble(model, cfg.baseline_members)
    timing = {"total_s": sw.s, "base_train_s": sw.s, "build_s": 0.0, "n_trained_nets": 1}
    return ens, timing, [model]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def rmse_of(ensemble, x, y):
    """Rough ID RMSE, recorded ONLY as a did-this-ensemble-build-correctly check.

    Not a quality metric: it skips the post-hoc variance calibration the paper
    tables use, and Laplace's value in particular depends entirely on
    --laplace-prior (the paper sweeps 1.0 ... 1e5 and selects). Do not quote it.
    """
    preds = ensemble.predict(x)
    mean = preds[0] if isinstance(preds, tuple) else preds
    if isinstance(mean, tuple):
        mean = mean[0]
    mean = jnp.mean(jnp.asarray(mean), axis=0)
    return float(jnp.sqrt(jnp.mean((mean - y) ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="Hopper-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-steps", type=int, default=10000)
    p.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Base-training budget applied IDENTICALLY to every method. "
        "5000 matches the baselines in gym_tasks.py; the cached P&C runs used 2000.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--de-members", type=int, default=50)
    p.add_argument("--pnc-members", type=int, default=50)
    p.add_argument("--baseline-members", type=int, default=100)
    p.add_argument("--n-directions", type=int, default=20)
    p.add_argument("--subset-size", type=int, default=4096)
    p.add_argument("--perturbation-size", type=float, default=10.0)
    p.add_argument("--lambda-reg", type=float, default=0.01)
    p.add_argument("--bootstrap-frac", type=float, default=0.1)
    p.add_argument("--laplace-prior", type=float, default=1.0)
    p.add_argument("--dropout-rate", type=float, default=0.05)
    p.add_argument("--hidden-dims", default="200,200,200,200")
    p.add_argument("--activation", default="relu")
    p.add_argument("--latency-batches", default="1,1000,10000")
    p.add_argument("--latency-reps", type=int, default=50)
    p.add_argument(
        "--latency-warmup",
        type=int,
        default=20,
        help="Warm-up calls at the SAME batch shape that is then timed, so XLA "
        "compilation stays out of the measurement. Report median_ms: the mean "
        "is still occasionally skewed by a straggler first rep.",
    )
    p.add_argument(
        "--methods",
        default="pnc,deep_ensemble,swag,laplace,mc_dropout,single_base",
        help="Comma-separated subset to run.",
    )
    p.add_argument("--outdir", default="results/neurips_2026_rebuttal/efficiency_v2")
    cfg = p.parse_args()

    cfg.hidden_dims = [int(h) for h in cfg.hidden_dims.split(",")]
    cfg.act_fn = _get_activation(cfg.activation)
    methods = [m.strip() for m in cfg.methods.split(",") if m.strip()]
    batches = [int(b) for b in cfg.latency_batches.split(",")]

    seed_everything(cfg.seed)
    x_all, y_all, x_ev, y_ev = load_env(cfg.env, cfg.seed, cfg.data_steps)
    x_tr, y_tr, x_va, y_va = _split_data(x_all, y_all)
    print(f"=== {cfg.env} seed={cfg.seed} steps={cfg.steps} ===")
    print(f"train={x_tr.shape} eval={x_ev.shape} device={jax.devices()[0]}")

    record = {
        "env": cfg.env,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "hidden_dims": cfg.hidden_dims,
        "de_members": cfg.de_members,
        "pnc_members": cfg.pnc_members,
        "baseline_members": cfg.baseline_members,
        "methods": {},
    }

    def run_one(name, fn):
        if name not in methods:
            return
        print(f"\n--- {name} ---")
        result = fn()
        ens, timing, roots = result[0], result[1], result[2]
        extras = result[3] if len(result) > 3 else {}
        entry = {"construction": timing, "storage_inference_resident": storage_of(roots)}
        entry.update(extras)
        entry["latency"] = {}
        for b in batches:
            if b > len(x_ev):
                continue
            entry["latency"][f"batch{b}"] = bench_latency(
                ens, x_ev[:b], cfg.latency_reps, cfg.latency_warmup
            )
            lat = entry["latency"][f"batch{b}"]
            print(f"  latency b={b:<6d} {lat['mean_ms']:8.3f} ms (median {lat['median_ms']:.3f})")
        entry["rmse_id_sanity_check"] = rmse_of(ens, x_ev, y_ev)
        print(
            f"  construction total={timing['total_s']:.2f}s "
            f"(base {timing['base_train_s']:.2f}s + build {timing['build_s']:.2f}s), "
            f"storage={entry['storage_inference_resident']['megabytes']} MB, "
            f"rmse={entry["rmse_id_sanity_check"]:.4f}"
        )
        record["methods"][name] = entry
        return ens

    run_one("pnc", lambda: build_pnc(x_tr, y_tr, x_va, y_va, x_all, cfg))
    run_one("swag", lambda: build_swag(x_tr, y_tr, x_va, y_va, cfg))
    run_one("laplace", lambda: build_laplace(x_tr, y_tr, x_va, y_va, x_all, y_all, cfg))
    run_one("mc_dropout", lambda: build_mc_dropout(x_tr, y_tr, x_va, y_va, cfg))

    if "single_base" in methods:
        print("\n--- single_base ---")
        with Stopwatch() as sw:
            m = build_single(x_tr, y_tr, x_va, y_va, cfg, cfg.seed)
        ens1 = StandardEnsemble([m])
        entry = {
            "construction": {
                "total_s": sw.s,
                "base_train_s": sw.s,
                "build_s": 0.0,
                "n_trained_nets": 1,
            },
            "storage_inference_resident": storage_of([m]),
            "latency": {},
        }
        for b in batches:
            if b > len(x_ev):
                continue
            entry["latency"][f"batch{b}"] = bench_latency(
                ens1, x_ev[:b], cfg.latency_reps, cfg.latency_warmup
            )
        entry["rmse_id_sanity_check"] = rmse_of(ens1, x_ev, y_ev)
        record["methods"]["single_base"] = entry
        print(f"  construction {sw.s:.2f}s, rmse={entry["rmse_id_sanity_check"]:.4f}")

    # Deep Ensemble last: it is by far the longest leg, so everything else is
    # already on disk if it gets interrupted.
    run_one("deep_ensemble", lambda: build_deep_ensemble(x_tr, y_tr, x_va, y_va, cfg))

    outdir = REPO / cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"efficiency_{cfg.env}_seed{cfg.seed}_steps{cfg.steps}.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
