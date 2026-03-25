"""Luigi tasks for Gym environment experiments."""
import json
import pickle
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import luigi
import numpy as np
from flax import nnx

from data import load_uci
from ensembles import (
    PJSVDEnsemble,
    LaplaceEnsemble,
    MCDropoutEnsemble,
    StandardEnsemble,
    SubspaceInferenceEnsemble,
    SWAGEnsemble,
)
from laplace import compute_kfac_factors
from metrics import compute_calibration, compute_nll, print_metrics
from models import (
    MCDropoutRegressionModel,
    ProbabilisticRegressionModel,
    RegressionModel,
)
from pjsvd import (
    find_optimal_perturbation_multi_layer,
)
from training import (
    train_model,
    train_probabilistic_model,
    train_subspace_model,
    train_swag_model,
)
from util import _find_pjsvd_directions, _get_activation, _ps_str, seed_everything, _split_data
import jax

# ===========================================================================
# UCI REGRESSION TASKS
# ===========================================================================

def _evaluate_uci(ensemble_name: str, ensemble: Any,
                  x_test: jax.Array, y_test: jax.Array,
                  sidecar_path: str | None = None) -> dict[str, float]:
    """Evaluate a regression ensemble on UCI data."""
    print(f"\n--- Results: {ensemble_name} ---")

    # Warm-up
    _ = ensemble.predict(x_test[:1])

    t0 = time.time()
    preds    = ensemble.predict(x_test)          # (S, N, D)
    preds.block_until_ready()
    eval_time = time.time() - t0
    mean     = jnp.mean(preds, axis=0)           # (N, D)
    var      = jnp.var(preds, axis=0)            # (N, D)
    avg_var  = float(jnp.mean(var))
    nll      = float(compute_nll(mean, var, y_test))
    ece      = float(compute_calibration(mean, var, y_test))
    rmse     = float(jnp.sqrt(jnp.mean((mean - y_test) ** 2)))
    print_metrics("Test", rmse, avg_var, nll, ece)

    sq_err_per_pt = np.array(jnp.mean((mean - y_test) ** 2, axis=-1))
    pred_var_per_pt = np.array(jnp.mean(var, axis=-1))

    if sidecar_path is not None:
        Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(sidecar_path, sq_error=sq_err_per_pt, pred_var=pred_var_per_pt)

    return {"rmse": rmse, "nll": nll, "ece": ece, "var": avg_var, "eval_time": eval_time}

class UCIBaseTask(luigi.Task):
    dataset    = luigi.Parameter(default="boston")
    steps      = luigi.IntParameter(default=5000)
    seed       = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")

    def _setup_task(self) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int, int, list[int]]:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train_full, y_train_full, x_test, y_test = load_uci(self.dataset, seed=self.seed)
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)
        in_dim, out_dim = x_train_full.shape[1], y_train_full.shape[1]
        # Use length of full training set for consistency in hidden_dims decision
        hidden_dims = [50] if len(x_train_full) < 5000 else [1000, 1000, 500, 50]
        return act_fn, x_tr, y_tr, x_va, y_va, x_test, y_test, in_dim, out_dim, hidden_dims

class UCITrainBaseModel(UCIBaseTask):
    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"base_model_steps{self.steps}_act-{self.activation}_seed{self.seed}.pkl"))

    def run(self) -> None:
        act_fn, x_tr, y_tr, x_va, y_va, _, _, in_dim, out_dim, hidden_dims = self._setup_task()
        Path("results/uci").mkdir(parents=True, exist_ok=True)

        print(f"\n=== UCI Base Model Training ({self.dataset}, steps={self.steps}, act={self.activation}) ===")
        t0 = time.time()
        model = RegressionModel(in_dim, out_dim, nnx.Rngs(self.seed), hidden_dims=hidden_dims, activation=act_fn)
        model = train_model(model, x_tr, y_tr, x_va, y_va, steps=self.steps)
        print(f"Training time: {time.time()-t0:.2f}s")

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "wb") as f:
            pickle.dump(state, f)
        print(f"Model checkpoint saved to {self.output().path}")


class UCIStandardEnsemble(UCIBaseTask):
    n_baseline = luigi.IntParameter(default=5)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"standard_ensemble_n{self.n_baseline}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        act_fn, x_tr, y_tr, x_va, y_va, x_test, y_test, in_dim, out_dim, hidden_dims = self._setup_task()

        print(f"\n=== UCI EXP 1: Standard Ensemble ({self.dataset}, n={self.n_baseline}, act={self.activation}) ===")
        t0 = time.time()
        models = []
        for i in range(self.n_baseline):
            m = ProbabilisticRegressionModel(in_dim, out_dim, nnx.Rngs(self.seed + i), hidden_dims=hidden_dims, activation=act_fn)
            m = train_probabilistic_model(m, x_tr, y_tr, x_va, y_va, steps=self.steps)
            models.append(m)
        
        # Ensure all models are ready
        for m in models:
            for p in jax.tree_util.tree_leaves(nnx.state(m)):
                if hasattr(p, 'block_until_ready'):
                    p.block_until_ready()

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")
        ensemble = StandardEnsemble(models)

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        metrics = _evaluate_uci("Standard Ensemble", ensemble, x_test, y_test,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        metrics["train_time"] = train_time
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class UCIMCDropout(UCIBaseTask):
    n_perturbations = luigi.IntParameter(default=100)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"mc_dropout_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        act_fn, x_tr, y_tr, x_va, y_va, x_test, y_test, in_dim, out_dim, hidden_dims = self._setup_task()

        print(f"\n=== UCI EXP 2: MC Dropout ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = MCDropoutRegressionModel(in_dim, out_dim, nnx.Rngs(self.seed),
                                         hidden_dims=hidden_dims, activation=act_fn)
        model = train_model(model, x_tr, y_tr, x_va, y_va, steps=self.steps)
        
        # Ensure model is ready
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")
        ensemble = MCDropoutEnsemble(model, self.n_perturbations)

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        metrics = _evaluate_uci("MC Dropout", ensemble, x_test, y_test,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        metrics["train_time"] = train_time
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class UCISWAG(UCIBaseTask):
    n_perturbations = luigi.IntParameter(default=100)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"swag_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        act_fn, x_tr, y_tr, x_va, y_va, x_test, y_test, in_dim, out_dim, hidden_dims = self._setup_task()

        print(f"\n=== UCI EXP 3: SWAG ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = ProbabilisticRegressionModel(in_dim, out_dim, nnx.Rngs(self.seed), hidden_dims=hidden_dims, activation=act_fn)
        model, swag_mean, swag_var = train_swag_model(
            model, x_tr, y_tr, x_va, y_va, steps=self.steps, swag_start=self.steps // 2)
        
        # Ensure SWAG stats are ready
        jax.block_until_ready(swag_mean)
        jax.block_until_ready(swag_var)

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")
        ensemble = SWAGEnsemble(model, swag_mean, swag_var, self.n_perturbations)

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        metrics = _evaluate_uci("SWAG", ensemble, x_test, y_test,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        metrics["train_time"] = train_time
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class UCILaplace(UCIBaseTask):
    n_perturbations = luigi.IntParameter(default=100)
    laplace_priors  = luigi.ListParameter(default=[1.0, 10.0, 100.0])

    def output(self) -> luigi.LocalTarget:
        priors_str = _ps_str(self.laplace_priors)
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"laplace_priors{priors_str}_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        act_fn, x_tr, y_tr, x_va, y_va, x_test, y_test, in_dim, out_dim, hidden_dims = self._setup_task()

        print(f"\n=== UCI EXP 4: Laplace ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t_start = time.time()
        model = RegressionModel(in_dim, out_dim, nnx.Rngs(self.seed), hidden_dims=hidden_dims, activation=act_fn)
        model = train_model(model, x_tr, y_tr, x_va, y_va, steps=self.steps)
        
        # Ensure base training is done
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        print(f"Base training time: {time.time()-t_start:.2f}s")

        subset_idx = np.random.choice(len(x_tr), min(4096, len(x_tr)), replace=False)
        print("Computing KFAC Factors...")
        t0 = time.time()
        factors = compute_kfac_factors(
            model, x_tr[subset_idx], y_tr[subset_idx],
            batch_size=128, is_classification=False)
        
        # Ensure factors are ready
        for f in jax.tree_util.tree_leaves(factors):
            if hasattr(f, 'block_until_ready'):
                f.block_until_ready()

        setup_time = time.time() - t_start
        print(f"KFAC time: {time.time()-t0:.2f}s")

        all_metrics = {}
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        base_npz = self.output().path.replace(".json", "")
        for prior in self.laplace_priors:
            lap_ens = LaplaceEnsemble(
                model=model, kfac_factors=factors,
                prior_precision=prior, n_models=self.n_perturbations,
                data_size=len(subset_idx))
            m = _evaluate_uci(f"Laplace (prior={prior})", lap_ens, x_test, y_test,
                              sidecar_path=f"{base_npz}_prior{prior}.npz")
            m["train_time"] = setup_time
            all_metrics[str(prior)] = m

        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class UCIPJSVD(UCIBaseTask):
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 60.0, 80.0])
    subset_size        = luigi.IntParameter(default=4096)

    def requires(self):
        return UCITrainBaseModel(dataset=self.dataset, steps=self.steps, seed=self.seed, activation=self.activation)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"pjsvd_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        act_fn, x_tr, y_tr, x_va, y_va, x_test, y_test, in_dim, out_dim, hidden_dims = self._setup_task()

        model = RegressionModel(in_dim, out_dim, nnx.Rngs(self.seed), hidden_dims=hidden_dims, activation=act_fn)
        with open(self.input().path, "rb") as f:
            state = pickle.load(f)
        nnx.update(model, state)

        print(f"\n=== UCI PJSVD ({self.dataset}, n={self.n_perturbations}, K={self.n_directions}, act={self.activation}) ===")
        actual_subset = min(len(x_tr), self.subset_size)
        idx   = np.random.choice(len(x_tr), actual_subset, replace=False)
        X_sub = x_tr[idx]

        W1 = model.layers[0].kernel.get_value()
        b1 = model.layers[0].bias.get_value()
        W2 = model.layers[1].kernel.get_value()
        b2 = model.layers[1].bias.get_value()

        def model_fn_l1(w):
            return act_fn(X_sub @ w + b1)

        t_start = time.time() # overall setup
        t_dir = time.time()
        v_opts, sigmas = _find_pjsvd_directions(model_fn_l1, W1, self.n_directions, seed=self.seed)
        h_old   = model_fn_l1(W1)
        mu_old  = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old,  axis=0)
        
        # Use a seeded RNG for the z-coefficients
        rng = np.random.RandomState(self.seed)
        all_z = rng.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        
        v_opts.block_until_ready()
        setup_time = time.time() - t_start
        print(f"Direction search time: {time.time()-t_dir:.2f}s")

        all_metrics = {}
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        base_npz = self.output().path.replace(".json", "")
        for p_size in self.perturbation_sizes:
            ens = PJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=["l1"], correction_mode="affine",
                activation=act_fn, sigma_sq_weights=self.sigma_sq_weights,
                layer_params={"l1": {"W": W1, "b": b1}},
                correction_params={"mu_old": mu_old, "std_old": std_old, "next_w": W2, "next_b": b2}
            )
            m = _evaluate_uci(f"PJSVD-{self.activation} (size={p_size})", ens, x_test, y_test,
                              sidecar_path=f"{base_npz}_ps{p_size}.npz")
            m["train_time"] = setup_time
            all_metrics[str(p_size)] = m

        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)

class UCIMultiLayerPJSVD(UCIBaseTask):
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[5.0, 10.0, 20.0, 40.0])
    subset_size        = luigi.IntParameter(default=4096)

    def requires(self):
        return UCITrainBaseModel(dataset=self.dataset, steps=self.steps, seed=self.seed, activation=self.activation)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"ml_pjsvd_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        act_fn, x_tr, y_tr, x_va, y_va, x_test, y_test, in_dim, out_dim, hidden_dims = self._setup_task()

        # MultiLayer needs at least 3 layers
        if len(hidden_dims) < 2:
            print("Skipping MultiLayerPJSVD: model too shallow.")
            Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output().path, "w") as f:
                json.dump({"skipped": True}, f)
            return

        model = RegressionModel(in_dim, out_dim, nnx.Rngs(self.seed), hidden_dims=hidden_dims, activation=act_fn)
        with open(self.input().path, "rb") as f:
            state = pickle.load(f)
        nnx.update(model, state)

        print(f"\n=== UCI ML-PJSVD ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t_start = time.time()
        actual_subset = min(len(x_tr), self.subset_size)
        idx = np.random.choice(len(x_tr), actual_subset, replace=False)
        X_sub = x_tr[idx]

        W1 = model.layers[0].kernel.get_value()
        b1 = model.layers[0].bias.get_value()
        W2 = model.layers[1].kernel.get_value()
        b2 = model.layers[1].bias.get_value()
        W3 = model.layers[2].kernel.get_value()
        b3 = model.layers[2].bias.get_value()

        def model_fn_layers(ws):
            w1, w2 = ws
            h1 = act_fn(X_sub @ w1 + b1)
            h2 = act_fn(h1 @ w2 + b2)
            return h2

        t_dir = time.time()
        v_opts_buf = np.zeros((self.n_directions, W1.size + W2.size), dtype=np.float32)
        direction_mask = np.zeros(self.n_directions, dtype=bool)
        sigmas = []

        for k in range(self.n_directions):
            v_opts_jax = jnp.array(v_opts_buf)
            mask_jax = jnp.array(direction_mask)
            v_opts_list, sigma = find_optimal_perturbation_multi_layer(
                model_fn_layers, [W1, W2], max_iter=500,
                orthogonal_directions=v_opts_jax, direction_mask=mask_jax,
                seed=self.seed + k)
            v_opts_buf[k] = np.array(jnp.concatenate([v.flatten() for v in v_opts_list]))
            direction_mask[k] = True
            sigmas.append(float(sigma))

        v_opts = jnp.array(v_opts_buf)
        v_opts.block_until_ready()
        setup_time = time.time() - t_start
        h_old = model_fn_layers([W1, W2])
        mu_old = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old, axis=0)
        
        rng = np.random.RandomState(self.seed)
        all_z = rng.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        print(f"ML Direction search time: {time.time()-t_dir:.2f}s")

        all_metrics = {}
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        base_npz = self.output().path.replace(".json", "")
        for p_size in self.perturbation_sizes:
            ens = PJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=["l1", "l2"], correction_mode="affine",
                activation=act_fn,
                layer_params={"l1": {"W": W1, "b": b1}, "l2": {"W": W2, "b": b2}},
                correction_params={"mu_old": mu_old, "std_old": std_old, "next_w": W3, "next_b": b3}
            )
            m = _evaluate_uci(f"ML-PJSVD-{self.activation} (size={p_size})", ens, x_test, y_test,
                              sidecar_path=f"{base_npz}_ps{p_size}.npz")
            m["train_time"] = setup_time
            all_metrics[str(p_size)] = m

        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)

class UCISubspaceInference(UCIBaseTask):
    n_perturbations = luigi.IntParameter(default=100)
    temperature     = luigi.FloatParameter(default=0.0)

    def requires(self):
        return UCITrainBaseModel(dataset=self.dataset, steps=self.steps, seed=self.seed, activation=self.activation)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"subspace_inference_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}_T{self.temperature}.json"))

    def run(self) -> None:
        act_fn, x_tr, y_tr, x_va, y_va, x_test, y_test, in_dim, out_dim, hidden_dims = self._setup_task()

        print(f"\n=== UCI EXP: Subspace Inference ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = ProbabilisticRegressionModel(in_dim, out_dim, nnx.Rngs(self.seed), hidden_dims=hidden_dims, activation=act_fn)
        model, swag_mean, pca_components = train_subspace_model(
            model, x_tr, y_tr, x_va, y_va, steps=self.steps, batch_size=64, swag_start=self.steps // 2, max_rank=20)

        T = self.temperature if self.temperature > 0 else 2.0 * float(np.sqrt(len(x_tr)))
        ensemble = SubspaceInferenceEnsemble(model, swag_mean, pca_components, self.n_perturbations,
                                             temperature=T, X_train=x_tr, Y_train=y_tr, use_ess=True, is_classification=False)
        # Ensure everything is ready
        jax.block_until_ready(swag_mean)
        for c in jax.tree_util.tree_leaves(pca_components):
            if hasattr(c, 'block_until_ready'):
                c.block_until_ready()

        setup_time = time.time() - t0
        print(f"Training time: {setup_time:.2f}s")

        m = _evaluate_uci("Subspace Inference", ensemble, x_test, y_test,
                          sidecar_path=self.output().path.replace(".json", ".npz"))
        m["train_time"] = setup_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(m, f, indent=2)


class AllUCIExperiments(luigi.WrapperTask):
    """Umbrella task that requires all UCI sub-experiments."""
    dataset            = luigi.Parameter(default="boston")
    steps              = luigi.IntParameter(default=5000)
    n_perturbations    = luigi.IntParameter(default=100)
    n_baseline         = luigi.IntParameter(default=5)
    n_directions       = luigi.IntParameter(default=40)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 60.0, 80.0])
    laplace_priors     = luigi.ListParameter(default=[1.0, 10.0, 100.0])
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")

    def requires(self) -> list[luigi.Task]:
        shared = dict(dataset=self.dataset, steps=self.steps, seed=self.seed, activation=self.activation)
        return [
            UCITrainBaseModel(**shared),
            UCIStandardEnsemble(n_baseline=self.n_baseline, **shared),
            UCIMCDropout(n_perturbations=self.n_perturbations, **shared),
            UCISWAG(n_perturbations=self.n_perturbations, **shared),
            UCILaplace(n_perturbations=self.n_perturbations,
                         laplace_priors=self.laplace_priors, **shared),
            UCIPJSVD(n_directions=self.n_directions,
                     n_perturbations=self.n_perturbations,
                     perturbation_sizes=self.perturbation_sizes, **shared),
            UCIMultiLayerPJSVD(n_directions=self.n_directions,
                               n_perturbations=self.n_perturbations,
                               perturbation_sizes=self.perturbation_sizes, **shared),
            UCISubspaceInference(n_perturbations=self.n_perturbations, **shared),
        ]
