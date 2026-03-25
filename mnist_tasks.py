import time
import json
import pickle
from pathlib import Path

import luigi
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from util import (
    _get_activation, seed_everything, _evaluate_mnist,
    _ps_str, _find_pjsvd_directions, _split_data
)
from pjsvd import (
    find_optimal_perturbation_multi_layer,
)
from models import ClassificationModel, MCDropoutClassificationModel
from data import load_mnist
from training import (
    train_classification_model, train_swag_classification_model,
)
from ensembles import (
    PJSVDEnsemble,
    LaplaceEnsemble,
    EnsemblePJSVDHybrid,
    StandardEnsemble,
    MCDropoutEnsemble,
    SWAGEnsemble,
)
from laplace import compute_kfac_factors


class MNISTTrainBaseModel(luigi.Task):
    """Train and checkpoint the shared MNIST base model."""
    steps      = luigi.IntParameter(default=5000)
    seed       = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "mnist" /
                f"base_model_steps{self.steps}_act-{self.activation}_seed{self.seed}.pkl"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        Path("results/mnist").mkdir(parents=True, exist_ok=True)

        x_train_full, y_train_full, _, _ = load_mnist()
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)
        in_dim, n_cls = x_train_full.shape[1], 10

        print(f"\n=== MNIST Base Model Training (steps={self.steps}, act={self.activation}) ===")
        t0 = time.time()
        model = ClassificationModel(in_dim, n_cls, nnx.Rngs(self.seed), activation=act_fn)
        model = train_classification_model(model, x_tr, y_tr, x_va, y_va, steps=self.steps)
        
        # Ensure model is ready
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        print(f"Training time: {time.time()-t0:.2f}s")

        state = nnx.state(model)
        with open(self.output().path, "wb") as f:
            pickle.dump(state, f)
        print(f"Model checkpoint saved to {self.output().path}")


class MNISTStandardEnsemble(luigi.Task):
    steps      = luigi.IntParameter(default=5000)
    n_baseline = luigi.IntParameter(default=5)
    seed       = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "mnist" /
                f"standard_ensemble_n{self.n_baseline}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train_full, y_train_full, x_test, y_test = load_mnist()
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)
        in_dim, n_cls = x_train_full.shape[1], 10

        print(f"\n=== MNIST EXP 1: Standard Ensemble (n={self.n_baseline}, act={self.activation}) ===")
        t0 = time.time()
        models = []
        for i in range(self.n_baseline):
            m = ClassificationModel(in_dim, n_cls, nnx.Rngs(self.seed + i), activation=act_fn)
            m = train_classification_model(m, x_tr, y_tr, x_va, y_va, steps=self.steps)
            models.append(m)
        
        # Ensure all models are ready
        for m in models:
            for p in jax.tree_util.tree_leaves(nnx.state(m)):
                if hasattr(p, 'block_until_ready'):
                    p.block_until_ready()

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")
        ensemble = StandardEnsemble(models)

        metrics = _evaluate_mnist("Standard Ensemble", ensemble, x_test, y_test, n_cls)
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class MNISTMCDropout(luigi.Task):
    steps           = luigi.IntParameter(default=5000)
    n_perturbations = luigi.IntParameter(default=100)
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "mnist" /
                f"mc_dropout_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train_full, y_train_full, x_test, y_test = load_mnist()
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)
        in_dim, n_cls = x_train_full.shape[1], 10

        print(f"\n=== MNIST EXP 2: MC Dropout (n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = MCDropoutClassificationModel(in_dim, n_cls, nnx.Rngs(self.seed), activation=act_fn)
        model = train_classification_model(model, x_tr, y_tr, x_va, y_va, steps=self.steps)
        
        # Ensure model is ready
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")
        ensemble = MCDropoutEnsemble(model, self.n_perturbations)

        metrics = _evaluate_mnist("MC Dropout", ensemble, x_test, y_test, n_cls)
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class MNISTSwag(luigi.Task):
    steps           = luigi.IntParameter(default=5000)
    n_perturbations = luigi.IntParameter(default=100)
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / "mnist" /
                f"swag_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train_full, y_train_full, x_test, y_test = load_mnist()
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)
        in_dim, n_cls = x_train_full.shape[1], 10

        print(f"\n=== MNIST EXP 3: SWAG (n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = ClassificationModel(in_dim, n_cls, nnx.Rngs(self.seed), activation=act_fn)
        model, swag_mean, swag_var = train_swag_classification_model(
            model, x_tr, y_tr, x_va, y_va, steps=self.steps, swag_start=self.steps // 2)
        
        # Ensure SWAG stats are ready
        jax.block_until_ready(swag_mean)
        jax.block_until_ready(swag_var)

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")
        ensemble = SWAGEnsemble(model, swag_mean, swag_var, self.n_perturbations)

        metrics = _evaluate_mnist("SWAG", ensemble, x_test, y_test, n_cls)
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class MNISTLaplace(luigi.Task):
    steps           = luigi.IntParameter(default=5000)
    n_perturbations = luigi.IntParameter(default=100)
    laplace_priors  = luigi.ListParameter(default=[1.0, 10.0, 100.0])
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def output(self) -> luigi.LocalTarget:
        priors_str = _ps_str(self.laplace_priors)
        return luigi.LocalTarget(
            str(Path("results") / "mnist" /
                f"laplace_priors{priors_str}_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train_full, y_train_full, x_test, y_test = load_mnist()
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)
        in_dim, n_cls = x_train_full.shape[1], 10

        print(f"\n=== MNIST EXP 4: Laplace (n={self.n_perturbations}, act={self.activation}) ===")
        t_start = time.time()
        model = ClassificationModel(in_dim, n_cls, nnx.Rngs(self.seed), activation=act_fn)
        model = train_classification_model(model, x_tr, y_tr, x_va, y_va, steps=self.steps)
        
        # Ensure base training is done
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        print(f"Base training time: {time.time()-t_start:.2f}s")

        rng = np.random.RandomState(self.seed)
        subset_idx = rng.choice(len(x_tr), min(4096, len(x_tr)), replace=False)
        print("Computing KFAC Factors...")
        t0 = time.time()
        factors = compute_kfac_factors(
            model, x_tr[subset_idx], y_tr[subset_idx],
            batch_size=128, is_classification=True)
        
        # Ensure factors are ready
        for f in jax.tree_util.tree_leaves(factors):
            if hasattr(f, 'block_until_ready'):
                f.block_until_ready()

        setup_time = time.time() - t_start
        print(f"KFAC time: {time.time()-t0:.2f}s")

        all_metrics = {}
        for prior in self.laplace_priors:
            lap_ens = LaplaceEnsemble(
                model=model, kfac_factors=factors,
                prior_precision=prior, n_models=self.n_perturbations,
                data_size=len(subset_idx))
            m = _evaluate_mnist(f"Laplace (prior={prior})", lap_ens, x_test, y_test, n_cls)
            m["train_time"] = setup_time
            all_metrics[str(prior)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class MNISTPJSVD(luigi.Task):
    """Single-layer PJSVD on MNIST — parameterised by activation."""
    steps              = luigi.IntParameter(default=5000)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 60.0, 80.0, 100.0])
    subset_size        = luigi.IntParameter(default=4096)
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")

    def requires(self) -> luigi.Task:
        return MNISTTrainBaseModel(steps=self.steps, seed=self.seed,
                                   activation=self.activation)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path("results") / "mnist" /
                f"pjsvd_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_mnist()
        in_dim, n_cls = x_train.shape[1], 10

        model = ClassificationModel(in_dim, n_cls, nnx.Rngs(self.seed), activation=act_fn)
        with open(self.input().path, "rb") as f:
            state = pickle.load(f)
        nnx.update(model, state)

        print(f"\n=== MNIST PJSVD (n={self.n_perturbations}, K={self.n_directions}, act={self.activation}) ===")
        actual_subset = min(len(x_train), self.subset_size)
        rng   = np.random.RandomState(self.seed)
        idx   = rng.choice(len(x_train), actual_subset, replace=False)
        X_sub = x_train[idx]

        W1 = model.l1.kernel.get_value()
        b1 = model.l1.bias.get_value()
        W2 = model.l2.kernel.get_value()
        b2 = model.l2.bias.get_value()

        def model_fn_l1(w):
            return act_fn(X_sub @ w + b1)

        print(f"Finding {self.n_directions} ORIGINAL null-space directions...")
        t_start = time.time() # renamed from t0
        t_dir = time.time()
        v_opts_orig, sigmas_orig = _find_pjsvd_directions(model_fn_l1, W1, self.n_directions, use_full_span=False, seed=self.seed)
        v_opts_orig.block_until_ready()
        print(f"Original direction time: {time.time()-t_dir:.2f}s")

        print(f"Finding {self.n_directions} FULL SPAN null-space directions...")
        t_dir = time.time()
        v_opts_full, sigmas_full = _find_pjsvd_directions(model_fn_l1, W1, self.n_directions, use_full_span=True, seed=self.seed)
        v_opts_full.block_until_ready()
        print(f"Full span direction time: {time.time()-t_dir:.2f}s")
        
        setup_time = time.time() - t_start

        h_old   = model_fn_l1(W1)
        mu_old  = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old,  axis=0)
        
        rng = np.random.RandomState(self.seed)
        all_z = rng.normal(0, 1, size=(self.n_perturbations, self.n_directions))

        all_metrics = {}
        base_npz = self.output().path.replace(".json", "")
        for p_size in self.perturbation_sizes:
            ens_orig = PJSVDEnsemble(
                base_model=model, v_opts=v_opts_orig, sigmas=sigmas_orig, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=["l1"], correction_mode="affine",
                activation=act_fn, sigma_sq_weights=False,
                layer_params={"l1": {"W": W1, "b": b1}},
                correction_params={"mu_old": mu_old, "std_old": std_old, "next_w": W2, "next_b": b2}
            )
            m_orig = _evaluate_mnist(f"PJSVD Orig (size={p_size})", ens_orig, x_test, y_test, n_cls,
                                     sidecar_path=f"{base_npz}_orig_ps{p_size}.npz")
            all_metrics[f"{p_size}_orig"] = m_orig

            ens_full = PJSVDEnsemble(
                base_model=model, v_opts=v_opts_full, sigmas=sigmas_full, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=["l1"], correction_mode="least_squares",
                activation=act_fn, sigma_sq_weights=False,
                layer_params={"l1": {"W": W1, "b": b1}},
                correction_params={"target_act": h_old @ W2 + b2}
            )
            m_full = _evaluate_mnist(f"PJSVD Full Span Least Squares (size={p_size})", ens_full, x_test, y_test, n_cls,
                                     sidecar_path=f"{base_npz}_full_ps{p_size}.npz")
            m_orig["train_time"] = setup_time
            m_full["train_time"] = setup_time
            all_metrics[f"{p_size}_orig"] = m_orig
            all_metrics[f"{p_size}_full"] = m_full

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class MNISTEnsemblePJSVD(luigi.Task):
    """EXP 6: Deep Ensemble + PJSVD hybrid."""
    steps              = luigi.IntParameter(default=5000)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 60.0, 80.0, 100.0])
    n_base_models      = luigi.IntParameter(default=5)
    subset_size        = luigi.IntParameter(default=4096)
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path("results") / "mnist" /
                f"ensemble_pjsvd_m{self.n_base_models}_k{self.n_directions}"
                f"_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train_full, y_train_full, x_test, y_test = load_mnist()
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)
        in_dim, n_cls = x_train_full.shape[1], 10
        actual_subset = min(len(x_train_full), self.subset_size)

        print(f"\n=== MNIST EXP 6: Ensemble+PJSVD Hybrid "
              f"({self.n_base_models} models × {self.n_perturbations} samples, act={self.activation}) ===")
        t_start = time.time()
        all_member_ensembles_orig = {p: [] for p in self.perturbation_sizes}
        all_member_ensembles_full = {p: [] for p in self.perturbation_sizes}

        for m_idx in range(self.n_base_models):
            print(f"\n  [Member {m_idx+1}/{self.n_base_models}] Training base model...")
            base_m = ClassificationModel(in_dim, n_cls, nnx.Rngs(self.seed + m_idx), activation=act_fn)
            base_m = train_classification_model(base_m, x_tr, y_tr, x_va, y_va, steps=self.steps)

            rng = np.random.RandomState(self.seed + m_idx)
            idx = rng.choice(len(x_tr), actual_subset, replace=False)
            X_sub = x_tr[idx]
            W1 = base_m.l1.kernel.get_value()
            b1 = base_m.l1.bias.get_value()
            W2 = base_m.l2.kernel.get_value()
            b2 = base_m.l2.bias.get_value()

            def model_fn_l1(w, _b1=b1, _X=X_sub):
                return act_fn(_X @ w + _b1)

            print(f"  [Member {m_idx+1}] Finding {self.n_directions} ORIGINAL null-space directions...")
            v_opts_orig, sigmas_orig = _find_pjsvd_directions(model_fn_l1, W1, self.n_directions, use_full_span=False, seed=self.seed + m_idx)
            print(f"  [Member {m_idx+1}] Finding {self.n_directions} FULL SPAN null-space directions...")
            v_opts_full, sigmas_full = _find_pjsvd_directions(model_fn_l1, W1, self.n_directions, use_full_span=True, seed=self.seed + m_idx)

            h_old   = model_fn_l1(W1)
            mu_old  = jnp.mean(h_old, axis=0)
            std_old = jnp.std(h_old,  axis=0)
            
            rng_z = np.random.RandomState(self.seed + m_idx)
            all_z = rng_z.normal(0, 1, size=(self.n_perturbations, self.n_directions))

            for p_size in self.perturbation_sizes:
                all_member_ensembles_orig[p_size].append(
                    PJSVDEnsemble(
                        base_model=base_m, v_opts=v_opts_orig, sigmas=sigmas_orig, z_coeffs=all_z,
                        perturbation_scale=p_size, X_sub=X_sub,
                        layers=["l1"], correction_mode="affine",
                        activation=act_fn, sigma_sq_weights=False,
                        layer_params={"l1": {"W": W1, "b": b1}},
                        correction_params={"mu_old": mu_old, "std_old": std_old, "next_w": W2, "next_b": b2}))
                all_member_ensembles_full[p_size].append(
                    PJSVDEnsemble(
                        base_model=base_m, v_opts=v_opts_full, sigmas=sigmas_full, z_coeffs=all_z,
                        perturbation_scale=p_size, X_sub=X_sub,
                        layers=["l1"], correction_mode="least_squares",
                        activation=act_fn, sigma_sq_weights=False,
                        layer_params={"l1": {"W": W1, "b": b1}},
                        correction_params={"target_act": h_old @ W2 + b2}))
        
        setup_time = time.time() - t_start
        all_metrics = {}
        for p_size in self.perturbation_sizes:
            hybrid_orig = EnsemblePJSVDHybrid(all_member_ensembles_orig[p_size])
            m_orig = _evaluate_mnist(f"Ensemble+PJSVD Orig (size={p_size})", hybrid_orig, x_test, y_test, n_cls)
            
            hybrid_full = EnsemblePJSVDHybrid(all_member_ensembles_full[p_size])
            m_full = _evaluate_mnist(f"Ensemble+PJSVD Full Span Least Squares (size={p_size})", hybrid_full, x_test, y_test, n_cls)
            
            m_orig["train_time"] = setup_time
            m_full["train_time"] = setup_time

            all_metrics[f"{p_size}_orig"] = m_orig
            all_metrics[f"{p_size}_full"] = m_full

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class MNISTMultiLayerPJSVD(luigi.Task):
    steps              = luigi.IntParameter(default=5000)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[5.0, 10.0, 20.0, 40.0, 80.0])
    subset_size        = luigi.IntParameter(default=4096)
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")

    def requires(self) -> luigi.Task:
        return MNISTTrainBaseModel(steps=self.steps, seed=self.seed, activation=self.activation)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path("results") / "mnist" /
                f"ml_pjsvd_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        x_train, y_train, x_test, y_test = load_mnist()
        in_dim, n_cls = x_train.shape[1], 10

        model = ClassificationModel(in_dim, n_cls, nnx.Rngs(self.seed), activation=act_fn)
        with open(self.input().path, "rb") as f:
            state = pickle.load(f)
        nnx.update(model, state)

        print(f"\n=== MNIST ML-PJSVD (n={self.n_perturbations}, act={self.activation}) ===")
        t_start = time.time()
        actual_subset = min(len(x_train), self.subset_size)
        idx = np.random.choice(len(x_train), actual_subset, replace=False)
        X_sub = x_train[idx]

        W1 = model.l1.kernel.get_value()
        b1 = model.l1.bias.get_value()
        W2 = model.l2.kernel.get_value()
        b2 = model.l2.bias.get_value()
        W3 = model.l3.kernel.get_value()
        b3 = model.l3.bias.get_value()

        def model_fn_layers(ws):
            w1, w2 = ws
            h1 = act_fn(X_sub @ w1 + b1)
            h2 = h1 @ w2 + b2
            return h2

        print(f"Finding {self.n_directions} ORIGINAL multi-layer null-space directions...")
        t_dir = time.time() # renamed from t0
        v_opts_buf_orig = np.zeros((self.n_directions, W1.size + W2.size), dtype=np.float32)
        direction_mask_orig = np.zeros(self.n_directions, dtype=bool)
        sigmas_orig = []

        for k in range(self.n_directions):
            v_opts_jax = jnp.array(v_opts_buf_orig)
            mask_jax = jnp.array(direction_mask_orig)
            v_opts_list, sigma = find_optimal_perturbation_multi_layer(
                model_fn_layers, [W1, W2], max_iter=500,
                orthogonal_directions=v_opts_jax, direction_mask=mask_jax,
                seed=self.seed + k)
            v_opts_buf_orig[k] = np.array(jnp.concatenate([v.flatten() for v in v_opts_list]))
            direction_mask_orig[k] = True
            sigmas_orig.append(float(sigma))

        v_opts_orig = jnp.array(v_opts_buf_orig)
        v_opts_orig.block_until_ready()
        print(f"Original direction time: {time.time()-t_dir:.2f}s")

        print(f"Finding {self.n_directions} FULL SPAN multi-layer null-space directions...")
        t_dir = time.time()
        v_opts_buf_full = np.zeros((self.n_directions, W1.size + W2.size), dtype=np.float32)
        direction_mask_full = np.zeros(self.n_directions, dtype=bool)
        sigmas_full = []

        for k in range(self.n_directions):
            v_opts_jax = jnp.array(v_opts_buf_full)
            mask_jax = jnp.array(direction_mask_full)
            v_opts_list, sigma = find_optimal_perturbation_multi_layer(
                model_fn_layers, [W1, W2], max_iter=500,
                orthogonal_directions=v_opts_jax, direction_mask=mask_jax,
                use_full_span=True,
                seed=self.seed + k)
            v_opts_buf_full[k] = np.array(jnp.concatenate([v.flatten() for v in v_opts_list]))
            direction_mask_full[k] = True
            sigmas_full.append(float(sigma))

        v_opts_full = jnp.array(v_opts_buf_full)
        v_opts_full.block_until_ready()
        print(f"Full span direction time: {time.time()-t_dir:.2f}s")
        
        setup_time = time.time() - t_start

        h_old = model_fn_layers([W1, W2])
        mu_old = jnp.mean(h_old, axis=0)
        std_old = jnp.std(h_old, axis=0)
        
        rng = np.random.RandomState(self.seed)
        all_z = rng.normal(0, 1, size=(self.n_perturbations, self.n_directions))

        all_metrics = {}
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        base_npz = self.output().path.replace(".json", "")
        for p_size in self.perturbation_sizes:
            ens_orig = PJSVDEnsemble(
                base_model=model, v_opts=v_opts_orig, sigmas=sigmas_orig, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=["l1", "l2"], correction_mode="affine",
                activation=act_fn,
                layer_params={"l1": {"W": W1, "b": b1}, "l2": {"W": W2, "b": b2}},
                correction_params={"mu_old": mu_old, "std_old": std_old, "next_w": W3, "next_b": b3}
            )
            m_orig = _evaluate_mnist(f"ML-PJSVD Orig (size={p_size})", ens_orig, x_test, y_test, n_cls,
                                     sidecar_path=f"{base_npz}_orig_ps{p_size}.npz")
            all_metrics[f"{p_size}_orig"] = m_orig

            ens_full = PJSVDEnsemble(
                base_model=model, v_opts=v_opts_full, sigmas=sigmas_full, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=["l1", "l2"], correction_mode="least_squares",
                activation=act_fn,
                layer_params={"l1": {"W": W1, "b": b1}, "l2": {"W": W2, "b": b2}},
                correction_params={"target_act": h_old @ W3 + b3}
            )
            m_full = _evaluate_mnist(f"ML-PJSVD Full Span Least Squares (size={p_size})", ens_full, x_test, y_test, n_cls,
                                     sidecar_path=f"{base_npz}_full_ps{p_size}.npz")
            m_orig["train_time"] = setup_time
            m_full["train_time"] = setup_time
            all_metrics[f"{p_size}_orig"] = m_orig
            all_metrics[f"{p_size}_full"] = m_full

        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class AllMNISTExperiments(luigi.WrapperTask):
    """Umbrella task that requires all MNIST sub-experiments."""
    steps              = luigi.IntParameter(default=5000)
    n_perturbations    = luigi.IntParameter(default=100)
    n_baseline         = luigi.IntParameter(default=5)
    n_directions       = luigi.IntParameter(default=40)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 60.0, 80.0, 160.0, 320.0])
    laplace_priors     = luigi.ListParameter(default=[1.0, 10.0, 100.0])
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")

    def requires(self) -> list[luigi.Task]:
        shared = dict(steps=self.steps, seed=self.seed, activation=self.activation)
        return [
            MNISTTrainBaseModel(**shared),
            MNISTStandardEnsemble(n_baseline=self.n_baseline, **shared),
            MNISTMCDropout(n_perturbations=self.n_perturbations, **shared),
            MNISTSwag(n_perturbations=self.n_perturbations, **shared),
            MNISTLaplace(n_perturbations=self.n_perturbations,
                         laplace_priors=self.laplace_priors, **shared),
            MNISTPJSVD(n_directions=self.n_directions,
                       n_perturbations=self.n_perturbations,
                       perturbation_sizes=self.perturbation_sizes, **shared),
            MNISTEnsemblePJSVD(n_directions=self.n_directions,
                               n_perturbations=self.n_perturbations,
                               perturbation_sizes=self.perturbation_sizes,
                               n_base_models=self.n_baseline, **shared),
            MNISTMultiLayerPJSVD(n_directions=self.n_directions,
                                 n_perturbations=self.n_perturbations,
                                 perturbation_sizes=[ps / 4 for ps in self.perturbation_sizes],
                                 **shared),
        ]
