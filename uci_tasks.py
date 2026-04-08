"""Luigi tasks for Gym environment experiments."""
import time
import json
import pickle
from pathlib import Path

import luigi
import jax.numpy as jnp
from flax import nnx
import numpy as np

from util import (
    _get_activation, seed_everything, _ps_str, _find_pjsvd_directions
)
from pjsvd import (
    find_optimal_perturbation_multi_layer,
    find_optimal_perturbation_multi_layer_full,
)
from models import ProbabilisticRegressionModel
from data import load_uci
from training import (
    train_probabilistic_model, train_swag_model,
    train_subspace_model
)
from ensembles import (
    CompactPJSVDEnsemble, CompactMultiLayerPJSVDEnsemble,
    LeastSquaresCompactPJSVDEnsemble, LeastSquaresCompactMultiLayerPJSVDEnsemble,
    StandardEnsemble, MCDropoutEnsemble, SWAGEnsemble, LaplaceEnsemble,
    SubspaceInferenceEnsemble,
)
from laplace import compute_kfac_factors

# ===========================================================================
# UCI REGRESSION TASKS
# ===========================================================================

def _evaluate_uci(ensemble_name: str, ensemble,
                  x_test, y_test,
                  sidecar_path: str | None = None) -> dict:
    """Evaluate a regression ensemble on UCI data."""
    print(f"\n--- Results: {ensemble_name} ---")

    preds    = ensemble.predict(x_test)          # (S, N, D)
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

    return {"rmse": rmse, "nll": nll, "ece": ece, "var": avg_var}

class UCITrainBaseModel(luigi.Task):
    dataset    = luigi.Parameter(default="boston")
    steps      = luigi.IntParameter(default=5000)
    seed       = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")

    def output(self):
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"base_model_steps{self.steps}_act-{self.activation}_seed{self.seed}.pkl"))

    def run(self):
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        Path("results/uci").mkdir(parents=True, exist_ok=True)

        x_train, y_train, _, _ = load_uci(self.dataset, seed=self.seed)
        in_dim, out_dim = x_train.shape[1], y_train.shape[1]

        hidden_dims = [50] if len(x_train) < 5000 else [1000, 1000, 500, 50]

        print(f"\n=== UCI Base Model Training ({self.dataset}, steps={self.steps}, act={self.activation}) ===")
        t0 = time.time()
        model = RegressionModel(in_dim, out_dim, nnx.Rngs(0), hidden_dims=hidden_dims, activation=act_fn)
        model = train_model(model, x_train, y_train, steps=self.steps)
        print(f"Training time: {time.time()-t0:.2f}s")

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "wb") as f:
            pickle.dump(state, f)
        print(f"Model checkpoint saved to {self.output().path}")


class UCIStandardEnsemble(luigi.Task):
    dataset    = luigi.Parameter(default="boston")
    steps      = luigi.IntParameter(default=5000)
    n_baseline = luigi.IntParameter(default=5)
    seed       = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")

    def output(self):
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"standard_ensemble_n{self.n_baseline}_act-{self.activation}_seed{self.seed}.json"))

    def run(self):
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_uci(self.dataset, seed=self.seed)
        in_dim, out_dim = x_train.shape[1], y_train.shape[1]
        hidden_dims = [50] if len(x_train) < 5000 else [1000, 1000, 500, 50]

        print(f"\n=== UCI EXP 1: Standard Ensemble ({self.dataset}, n={self.n_baseline}, act={self.activation}) ===")
        t0 = time.time()
        models = []
        for i in range(self.n_baseline):
            m = ProbabilisticRegressionModel(in_dim, out_dim, nnx.Rngs(i), hidden_dims=hidden_dims, activation=act_fn)
            m = train_probabilistic_model(m, x_train, y_train, steps=self.steps)
            models.append(m)
        ensemble = StandardEnsemble(models)
        print(f"Training time: {time.time()-t0:.2f}s")

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        metrics = _evaluate_uci("Standard Ensemble", ensemble, x_test, y_test,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class UCIMCDropout(luigi.Task):
    dataset         = luigi.Parameter(default="boston")
    steps           = luigi.IntParameter(default=5000)
    n_perturbations = luigi.IntParameter(default=100)
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def output(self):
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"mc_dropout_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self):
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_uci(self.dataset, seed=self.seed)
        in_dim, out_dim = x_train.shape[1], y_train.shape[1]
        hidden_dims = [50] if len(x_train) < 5000 else [1000, 1000, 500, 50]

        print(f"\n=== UCI EXP 2: MC Dropout ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = MCDropoutRegressionModel(in_dim, out_dim, nnx.Rngs(42),
                                         hidden_dims=hidden_dims, activation=act_fn)
        model = train_model(model, x_train, y_train, steps=self.steps)
        ensemble = MCDropoutEnsemble(model, self.n_perturbations)
        print(f"Training time: {time.time()-t0:.2f}s")

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        metrics = _evaluate_uci("MC Dropout", ensemble, x_test, y_test,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class UCISWAG(luigi.Task):
    dataset         = luigi.Parameter(default="boston")
    steps           = luigi.IntParameter(default=5000)
    n_perturbations = luigi.IntParameter(default=100)
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def output(self):
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"swag_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self):
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_uci(self.dataset, seed=self.seed)
        in_dim, out_dim = x_train.shape[1], y_train.shape[1]
        hidden_dims = [50] if len(x_train) < 5000 else [1000, 1000, 500, 50]

        print(f"\n=== UCI EXP 3: SWAG ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = ProbabilisticRegressionModel(in_dim, out_dim, nnx.Rngs(99), hidden_dims=hidden_dims, activation=act_fn)
        model, swag_mean, swag_var = train_swag_model(
            model, x_train, y_train, steps=self.steps, swag_start=self.steps // 2)
        ensemble = SWAGEnsemble(model, swag_mean, swag_var, self.n_perturbations)
        print(f"Training time: {time.time()-t0:.2f}s")

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        metrics = _evaluate_uci("SWAG", ensemble, x_test, y_test,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class UCILaplace(luigi.Task):
    dataset         = luigi.Parameter(default="boston")
    steps           = luigi.IntParameter(default=5000)
    n_perturbations = luigi.IntParameter(default=100)
    laplace_priors  = luigi.ListParameter(default=[1.0, 10.0, 100.0])
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def output(self):
        priors_str = _ps_str(self.laplace_priors)
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"laplace_priors{priors_str}_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self):
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_uci(self.dataset, seed=self.seed)
        in_dim, out_dim = x_train.shape[1], y_train.shape[1]
        hidden_dims = [50] if len(x_train) < 5000 else [1000, 1000, 500, 50]

        print(f"\n=== UCI EXP 4: Laplace ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = RegressionModel(in_dim, out_dim, nnx.Rngs(7), hidden_dims=hidden_dims, activation=act_fn)
        model = train_model(model, x_train, y_train, steps=self.steps)
        print(f"Base training time: {time.time()-t0:.2f}s")

        subset_idx = np.random.choice(len(x_train), min(4096, len(x_train)), replace=False)
        print("Computing KFAC Factors...")
        t0 = time.time()
        factors = compute_kfac_factors(
            model, x_train[subset_idx], y_train[subset_idx],
            batch_size=128, is_classification=False)
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
            all_metrics[str(prior)] = m

        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class UCIPJSVD(luigi.Task):
    dataset            = luigi.Parameter(default="boston")
    steps              = luigi.IntParameter(default=5000)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 60.0, 80.0])
    subset_size        = luigi.IntParameter(default=4096)
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")

    def requires(self):
        return UCITrainBaseModel(dataset=self.dataset, steps=self.steps, seed=self.seed, activation=self.activation)

    def output(self):
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"pjsvd_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"))

    def run(self):
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_uci(self.dataset, seed=self.seed)
        in_dim, out_dim = x_train.shape[1], y_train.shape[1]
        hidden_dims = [50] if len(x_train) < 5000 else [1000, 1000, 500, 50]

        model = RegressionModel(in_dim, out_dim, nnx.Rngs(0), hidden_dims=hidden_dims, activation=act_fn)
        with open(self.input().path, "rb") as f:
            state = pickle.load(f)
        nnx.update(model, state)

        print(f"\n=== UCI PJSVD ({self.dataset}, n={self.n_perturbations}, K={self.n_directions}, act={self.activation}) ===")
        actual_subset = min(len(x_train), self.subset_size)
        idx   = np.random.choice(len(x_train), actual_subset, replace=False)
        X_sub = x_train[idx]

        W1 = model.layers[0].kernel.get_value()
        b1 = model.layers[0].bias.get_value()
        W2 = model.layers[1].kernel.get_value()
        b2 = model.layers[1].bias.get_value()

        def model_fn_l1(w):
            return act_fn(X_sub @ w + b1)

        t0 = time.time()
        v_opts, sigmas = _find_pjsvd_directions(model_fn_l1, W1, self.n_directions)
        h_old   = model_fn_l1(W1)
        mu_old  = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old,  axis=0)
        all_z   = np.random.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        print(f"Direction search time: {time.time()-t0:.2f}s")

        all_metrics = {}
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        base_npz = self.output().path.replace(".json", "")
        for p_size in self.perturbation_sizes:
            ens = CompactPJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, W1=W1, b1=b1, W2=W2, b2=b2,
                mu_old=mu_old, std_old=std_old, activation=act_fn, X_sub=X_sub)
            m = _evaluate_uci(f"PJSVD-{self.activation} (size={p_size})", ens, x_test, y_test,
                              sidecar_path=f"{base_npz}_ps{p_size}.npz")
            all_metrics[str(p_size)] = m

        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)

class UCIMultiLayerPJSVD(luigi.Task):
    dataset            = luigi.Parameter(default="boston")
    steps              = luigi.IntParameter(default=5000)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[5.0, 10.0, 20.0, 40.0])
    subset_size        = luigi.IntParameter(default=4096)
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")

    def requires(self):
        return UCITrainBaseModel(dataset=self.dataset, steps=self.steps, seed=self.seed, activation=self.activation)

    def output(self):
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"ml_pjsvd_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"))

    def run(self):
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_uci(self.dataset, seed=self.seed)
        in_dim, out_dim = x_train.shape[1], y_train.shape[1]
        hidden_dims = [50] if len(x_train) < 5000 else [1000, 1000, 500, 50]

        # MultiLayer needs at least 3 layers
        if len(hidden_dims) < 2:
            print("Skipping MultiLayerPJSVD: model too shallow.")
            Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output().path, "w") as f:
                json.dump({"skipped": True}, f)
            return

        model = RegressionModel(in_dim, out_dim, nnx.Rngs(0), hidden_dims=hidden_dims, activation=act_fn)
        with open(self.input().path, "rb") as f:
            state = pickle.load(f)
        nnx.update(model, state)

        print(f"\n=== UCI ML-PJSVD ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        actual_subset = min(len(x_train), self.subset_size)
        idx = np.random.choice(len(x_train), actual_subset, replace=False)
        X_sub = x_train[idx]

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

        t0 = time.time()
        v_opts_buf = np.zeros((self.n_directions, W1.size + W2.size), dtype=np.float32)
        direction_mask = np.zeros(self.n_directions, dtype=bool)
        sigmas = []

        for k in range(self.n_directions):
            v_opts_jax = jnp.array(v_opts_buf)
            mask_jax = jnp.array(direction_mask)
            v_opts_list, sigma = find_optimal_perturbation_multi_layer(
                model_fn_layers, [W1, W2], max_iter=500,
                orthogonal_directions=v_opts_jax, direction_mask=mask_jax)
            v_opts_buf[k] = np.array(jnp.concatenate([v.flatten() for v in v_opts_list]))
            direction_mask[k] = True
            sigmas.append(float(sigma))

        v_opts = jnp.array(v_opts_buf)
        h_old = model_fn_layers([W1, W2])
        mu_old = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old, axis=0)
        all_z = np.random.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        print(f"ML Direction search time: {time.time()-t0:.2f}s")

        all_metrics = {}
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        base_npz = self.output().path.replace(".json", "")
        for p_size in self.perturbation_sizes:
            ens = CompactMultiLayerPJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
                mu_old=mu_old, std_old=std_old, activation=act_fn, X_sub=X_sub)
            m = _evaluate_uci(f"ML-PJSVD-{self.activation} (size={p_size})", ens, x_test, y_test,
                              sidecar_path=f"{base_npz}_ps{p_size}.npz")
            all_metrics[str(p_size)] = m

        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)

class UCISubspaceInference(luigi.Task):
    dataset         = luigi.Parameter()
    steps           = luigi.IntParameter(default=5000)
    n_perturbations = luigi.IntParameter(default=100)
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")
    temperature     = luigi.FloatParameter(default=0.0)

    def requires(self):
        return UCITrainBaseModel(dataset=self.dataset, steps=self.steps, seed=self.seed, activation=self.activation)

    def output(self):
        return luigi.LocalTarget(
            str(Path("results") / "uci" / self.dataset /
                f"subspace_inference_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}_T{self.temperature}.json"))

    def run(self):
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_uci(self.dataset, seed=self.seed)

        print(f"\n=== UCI EXP: Subspace Inference ({self.dataset}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        in_dim, out_dim = x_train.shape[1], y_train.shape[1]
        hidden_dims = [50] if len(x_train) < 5000 else [1000, 1000, 500, 50]
        model = ProbabilisticRegressionModel(in_dim, out_dim, nnx.Rngs(params=888), hidden_dims=hidden_dims, activation=act_fn)
        model, swag_mean, pca_components = train_subspace_model(
            model, x_train, y_train, steps=self.steps, batch_size=64, swag_start=self.steps // 2, max_rank=20)

        T = self.temperature if self.temperature > 0 else 2.0 * float(np.sqrt(len(x_train)))
        ensemble = SubspaceInferenceEnsemble(model, swag_mean, pca_components, self.n_perturbations,
                                             temperature=T, X_train=x_train, Y_train=y_train, use_ess=True, is_classification=False)
        print(f"Training time: {time.time()-t0:.2f}s")

        m = _evaluate_uci("Subspace Inference", ensemble, x_test, y_test,
                          sidecar_path=self.output().path.replace(".json", ".npz"))
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

    def requires(self):
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
