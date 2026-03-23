"""Luigi tasks for Gym environment experiments."""
import time
import json
from pathlib import Path

import jax
import luigi
import jax.numpy as jnp
from flax import nnx
import numpy as np

from util import (
    _get_activation, seed_everything, _evaluate_gym,
    _load_gym_data, _ps_str, _find_pjsvd_directions,
    _split_data
)
from pjsvd import (
    find_optimal_perturbation_multi_layer,
    find_optimal_perturbation_multi_layer_full,
)
from models import TransitionModel, MCDropoutTransitionModel, ProbabilisticRegressionModel
from data import collect_data, id_policy_random, OODPolicyWrapper
from jaxtyping_bridge import Array, Float
from training import train_model, train_swag_model, train_probabilistic_model, train_subspace_model
from ensembles import (
    PJSVDEnsemble,
    StandardEnsemble, MCDropoutEnsemble, SWAGEnsemble, LaplaceEnsemble,
    SubspaceInferenceEnsemble,
)
from laplace import compute_kfac_factors


class CollectGymData(luigi.Task):
    """Collect ID (train + eval) and OOD transitions for a gym environment."""
    env   = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    seed  = luigi.IntParameter(default=0)

    def _base(self) -> Path:
        return Path("results") / self.env

    def output(self) -> dict[str, luigi.LocalTarget]:
        b = self._base()
        s = f"seed{self.seed}_steps{self.steps}"
        return {
            "id_train": luigi.LocalTarget(str(b / f"data_id_train_{s}.npz")),
            "id_eval":  luigi.LocalTarget(str(b / f"data_id_eval_{s}.npz")),
            "ood":      luigi.LocalTarget(str(b / f"data_ood_{s}.npz")),
        }

    def run(self) -> None:
        seed_everything(self.seed)
        self._base().mkdir(parents=True, exist_ok=True)

        print(f"\n=== Collecting gym data: {self.env} (steps={self.steps}) ===")
        inputs_id,      targets_id      = collect_data(self.env, self.steps, OODPolicyWrapper(),  seed=self.seed)
        inputs_id_eval, targets_id_eval = collect_data(self.env, self.steps, OODPolicyWrapper(),  seed=self.seed + 1)
        inputs_ood,     targets_ood     = collect_data(self.env, self.steps, id_policy_random,   seed=self.seed + 2)

        np.savez(self.output()["id_train"].path, inputs=np.array(inputs_id),      targets=np.array(targets_id))
        np.savez(self.output()["id_eval"].path,  inputs=np.array(inputs_id_eval), targets=np.array(targets_id_eval))
        np.savez(self.output()["ood"].path,      inputs=np.array(inputs_ood),     targets=np.array(targets_ood))
        print("Data saved.")


class GymStandardEnsemble(luigi.Task):
    env        = luigi.Parameter()
    steps      = luigi.IntParameter(default=10000)
    n_baseline = luigi.IntParameter(default=5)
    seed       = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / self.env /
                f"standard_ensemble_n{self.n_baseline}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        inputs_id, targets_id, inputs_id_eval, targets_id_eval, inputs_ood, targets_ood = \
            _load_gym_data(self.input())
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(f"\n=== EXP 2: Standard Ensemble ({self.env}, n={self.n_baseline}, act={self.activation}) ===")
        t0 = time.time()
        models = []
        for i in range(self.n_baseline):
            m = ProbabilisticRegressionModel(inputs_id.shape[1], targets_id.shape[1], nnx.Rngs(params=self.seed + i),
                                             hidden_dims=[64, 64], activation=act_fn)
            print(f"  Training model {i+1}/{self.n_baseline}...")
            m = train_probabilistic_model(m, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64)
            models.append(m)
        
        # Ensure models are fully trained/initialized
        for m in models:
            for p in jax.tree_util.tree_leaves(nnx.state(m)):
                if hasattr(p, 'block_until_ready'):
                    p.block_until_ready()
                    
        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")

        ensemble = StandardEnsemble(models)
        metrics = _evaluate_gym("Standard Ensemble", ensemble,
                                inputs_id_eval, targets_id_eval, inputs_ood, targets_ood,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class GymMCDropout(luigi.Task):
    env             = luigi.Parameter()
    steps           = luigi.IntParameter(default=10000)
    n_perturbations = luigi.IntParameter(default=1000)
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / self.env /
                f"mc_dropout_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        inputs_id, targets_id, inputs_id_eval, targets_id_eval, inputs_ood, targets_ood = \
            _load_gym_data(self.input())
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(f"\n=== EXP 3: MC Dropout ({self.env}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = MCDropoutTransitionModel(inputs_id.shape[1], targets_id.shape[1],
                                         nnx.Rngs(params=self.seed, dropout=self.seed + 1), dropout_rate=0.1,
                                         activation=act_fn)
        model = train_model(model, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64)
        
        # Ensure model is fully trained
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")

        ensemble = MCDropoutEnsemble(model, self.n_perturbations)
        metrics = _evaluate_gym("MC Dropout", ensemble,
                                inputs_id_eval, targets_id_eval, inputs_ood, targets_ood,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class GymSWAG(luigi.Task):
    env             = luigi.Parameter()
    steps           = luigi.IntParameter(default=10000)
    n_perturbations = luigi.IntParameter(default=1000)
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / self.env /
                f"swag_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        inputs_id, targets_id, inputs_id_eval, targets_id_eval, inputs_ood, targets_ood = \
            _load_gym_data(self.input())
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(f"\n=== EXP 4: SWAG ({self.env}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = ProbabilisticRegressionModel(inputs_id.shape[1], targets_id.shape[1], nnx.Rngs(params=self.seed),
                                             hidden_dims=[64, 64], activation=act_fn)
        model, swag_mean, swag_var = train_swag_model(
            model, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64, swag_start=1000)
        
        # Ensure SWAG stats are ready
        jax.block_until_ready(swag_mean)
        jax.block_until_ready(swag_var)

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")

        ensemble = SWAGEnsemble(model, swag_mean, swag_var, self.n_perturbations)
        metrics = _evaluate_gym("SWAG", ensemble,
                                inputs_id_eval, targets_id_eval, inputs_ood, targets_ood,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class GymLaplace(luigi.Task):
    env             = luigi.Parameter()
    steps           = luigi.IntParameter(default=10000)
    n_perturbations = luigi.IntParameter(default=1000)
    subset_size     = luigi.IntParameter(default=4096)
    laplace_priors  = luigi.ListParameter(default=[1.0, 5.0, 10.0, 50.0, 100.0])
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        priors_str = _ps_str(self.laplace_priors)
        return luigi.LocalTarget(
            str(Path("results") / self.env /
                f"laplace_priors{priors_str}_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        inputs_id, targets_id, inputs_id_eval, targets_id_eval, inputs_ood, targets_ood = \
            _load_gym_data(self.input())
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(f"\n=== EXP 5: Laplace ({self.env}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = TransitionModel(inputs_id.shape[1], targets_id.shape[1], nnx.Rngs(params=self.seed),
                                activation=act_fn)
        model = train_model(model, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64)

        actual_subset = min(len(inputs_id), self.subset_size)
        subset_idx = np.random.choice(len(inputs_id), actual_subset, replace=False)
        X_sub = inputs_id[subset_idx]
        Y_sub = targets_id[subset_idx]

        print("Computing KFAC Factors...")
        factors = compute_kfac_factors(model, X_sub, Y_sub, batch_size=128)
        
        # Ensure factors are ready
        for f in jax.tree_util.tree_leaves(factors):
            if hasattr(f, 'block_until_ready'):
                f.block_until_ready()

        setup_time = time.time() - t0
        print(f"Training + KFAC time: {setup_time:.2f}s")

        all_metrics = {}
        base_npz = self.output().path.replace(".json", "")
        for prior in self.laplace_priors:
            lap_ens = LaplaceEnsemble(
                model=model, kfac_factors=factors,
                prior_precision=prior, n_models=self.n_perturbations,
                data_size=actual_subset)
            m = _evaluate_gym(f"Laplace (prior={prior})", lap_ens,
                              inputs_id_eval, targets_id_eval, inputs_ood, targets_ood,
                              sidecar_path=f"{base_npz}_prior{prior}.npz")
            m["train_time"] = setup_time # Consistent, non-cumulative setup time
            all_metrics[str(prior)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class GymPJSVD(luigi.Task):
    env                = luigi.Parameter()
    steps              = luigi.IntParameter(default=10000)
    subset_size        = luigi.IntParameter(default=4096)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=1000)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 80.0, 160.0])
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")
    correction_mode    = luigi.ChoiceParameter(default="affine", choices=["affine", "least_squares"])
    layer_scope        = luigi.ChoiceParameter(default="first", choices=["first", "multi"])
    use_full_span      = luigi.BoolParameter(default=False)
    hidden_dims        = luigi.ListParameter(default=[64, 64])

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        full_str = "_full" if self.use_full_span else ""
        return luigi.LocalTarget(
            str(Path("results") / self.env /
                f"pjsvd_{self.layer_scope}_{self.correction_mode}{full_str}_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        inputs_id, targets_id, inputs_id_eval, targets_id_eval, inputs_ood, targets_ood = \
            _load_gym_data(self.input())
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(f"\n=== Gym PJSVD ({self.env}, scope={self.layer_scope}, mode={self.correction_mode}, full={self.use_full_span}) ===")
        t_start = time.time()
        model = TransitionModel(inputs_id.shape[1], targets_id.shape[1], nnx.Rngs(params=self.seed),
                                activation=act_fn)
        # model depth check for multi-layer
        if self.layer_scope == "multi" and len(self.hidden_dims) < 2:
            print("Skipping Multi-Layer PJSVD: model too shallow.")
            Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output().path, "w") as f:
                json.dump({"skipped": True}, f)
            return

        model = train_model(model, x_tr, y_tr, x_va, y_va)
        
        # End of base training
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        print(f"Base model training time: {time.time()-t_start:.2f}s")

        actual_subset = min(len(inputs_id), self.subset_size)
        subset_idx = np.random.choice(len(inputs_id), actual_subset, replace=False)
        X_sub = inputs_id[subset_idx]

        W1 = model.l1.kernel.get_value()
        b1 = model.l1.bias.get_value()
        W2 = model.l2.kernel.get_value()
        b2 = model.l2.bias.get_value()

        if self.layer_scope == "first":
            def model_fn(w):
                return act_fn(X_sub @ w + b1)
            t_dir = time.time()
            v_opts, sigmas = _find_pjsvd_directions(model_fn, W1, self.n_directions, use_full_span=self.use_full_span, seed=self.seed)
            v_opts.block_until_ready()
            print(f"Direction search time: {time.time()-t_dir:.2f}s")

            h_old = model_fn(W1)
            mu_old = jnp.mean(h_old, axis=0)
            std_old = jnp.std(h_old, axis=0)

            correction_params = {}
            if self.correction_mode == "affine":
                correction_params = {"mu_old": mu_old, "std_old": std_old, "next_w": W2, "next_b": b2}
            else:
                correction_params = {"target_act": h_old @ W2 + b2}

            layer_params = {"l1": {"W": W1, "b": b1}}
            perturbed_layers = ["l1"]

        else: # multi
            W3 = model.l3.kernel.get_value()
            b3 = model.l3.bias.get_value()
            def model_fn_ws(ws):
                w1, w2 = ws
                h1 = act_fn(X_sub @ w1 + b1)
                h2 = act_fn(h1 @ w2 + b2)
                return h2

            solver_fn = find_optimal_perturbation_multi_layer_full if self.use_full_span else find_optimal_perturbation_multi_layer
            
            v_opts_buf = np.zeros((self.n_directions, W1.size + W2.size), dtype=np.float32)
            direction_mask = np.zeros(self.n_directions, dtype=bool)
            sigmas = []

            t_dir = time.time()
            for k in range(self.n_directions):
                v_opts_jax = jnp.array(v_opts_buf)
                mask_jax = jnp.array(direction_mask)
                v_opts_list, sigma = solver_fn(
                    model_fn_ws, [W1, W2], max_iter=500,
                    orthogonal_directions=v_opts_jax, direction_mask=mask_jax,
                    seed=self.seed + k)
                v_opts_buf[k] = np.array(jnp.concatenate([v.flatten() for v in v_opts_list]))
                direction_mask[k] = True
                sigmas.append(float(sigma))
            
            v_opts = jnp.array(v_opts_buf)
            v_opts.block_until_ready()
            print(f"ML Direction search time: {time.time()-t_dir:.2f}s")

            h_old = model_fn_ws([W1, W2])
            mu_old = jnp.mean(h_old, axis=0)
            std_old = jnp.std(h_old, axis=0)

            correction_params = {}
            if self.correction_mode == "affine":
                correction_params = {"mu_old": mu_old, "std_old": std_old, "next_w": W3, "next_b": b3}
            else:
                correction_params = {"target_act": h_old @ W3 + b3}

            layer_params = {"l1": {"W": W1, "b": b1}, "l2": {"W": W2, "b": b2}}
            perturbed_layers = ["l1", "l2"]

        setup_time = time.time() - t_start # Total setup time (base train + direction search)

        rng = np.random.RandomState(self.seed)
        all_z = rng.normal(0, 1, size=(self.n_perturbations, self.n_directions))

        all_metrics = {}
        base_npz = self.output().path.replace(".json", "")
        for p_size in self.perturbation_sizes:
            ens = PJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=perturbed_layers, correction_mode=self.correction_mode,
                activation=act_fn, layer_params=layer_params,
                correction_params=correction_params
            )
            m = _evaluate_gym(f"PJSVD-{self.layer_scope}-{self.correction_mode} (size={p_size})", ens,
                              inputs_id_eval, targets_id_eval, inputs_ood, targets_ood,
                              sidecar_path=f"{base_npz}_ps{p_size}.npz")
            m["train_time"] = setup_time # Consistent, non-cumulative setup time
            all_metrics[str(p_size)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)




class GymSubspaceInference(luigi.Task):
    env             = luigi.Parameter()
    steps           = luigi.IntParameter(default=10000)
    n_perturbations = luigi.IntParameter(default=1000)
    seed            = luigi.IntParameter(default=0)
    activation      = luigi.Parameter(default="relu")
    temperature     = luigi.FloatParameter(default=0.0)

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path("results") / self.env /
                f"subspace_inference_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}_T{self.temperature}.json"))

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        inputs_id, targets_id, inputs_id_eval, targets_id_eval, inputs_ood, targets_ood = \
            _load_gym_data(self.input())
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(f"\n=== EXP: Subspace Inference ({self.env}, n={self.n_perturbations}, act={self.activation}) ===")
        t0 = time.time()
        model = ProbabilisticRegressionModel(inputs_id.shape[1], targets_id.shape[1], nnx.Rngs(params=self.seed),
                                             hidden_dims=[64, 64], activation=act_fn)
        model, swag_mean, pca_components = train_subspace_model(
            model, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64, swag_start=1000, max_rank=20)

        T = self.temperature if self.temperature > 0 else 2.0 * float(np.sqrt(len(inputs_id)))
        ensemble = SubspaceInferenceEnsemble(model, swag_mean, pca_components, self.n_perturbations,
                                             temperature=T, X_train=inputs_id, Y_train=targets_id,
                                             use_ess=True, is_classification=False)
        
        # Ensure PCA or other components are ready
        jax.block_until_ready(swag_mean)
        for c in jax.tree_util.tree_leaves(pca_components):
            if hasattr(c, 'block_until_ready'):
                c.block_until_ready()

        setup_time = time.time() - t0
        print(f"Training time: {setup_time:.2f}s")

        metrics = _evaluate_gym("Subspace Inference", ensemble,
                                inputs_id_eval, targets_id_eval, inputs_ood, targets_ood,
                                sidecar_path=self.output().path.replace(".json", ".npz"))
        metrics["train_time"] = setup_time

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class AllGymExperiments(luigi.WrapperTask):
    """Umbrella task that requires all gym sub-experiments."""
    env                = luigi.Parameter()
    steps              = luigi.IntParameter(default=10000)
    subset_size        = luigi.IntParameter(default=4096)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=1000)
    n_baseline         = luigi.IntParameter(default=5)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 80.0, 160.0])
    laplace_priors     = luigi.ListParameter(default=[1.0, 5.0, 10.0, 50.0, 100.0])
    seed               = luigi.IntParameter(default=0)
    activation         = luigi.Parameter(default="relu")

    def requires(self) -> list[luigi.Task]:
        ml_ps   = [ps / 4 for ps in self.perturbation_sizes]
        shared  = dict(env=self.env, steps=self.steps,
                       seed=self.seed, activation=self.activation)
        tasks = [
            GymStandardEnsemble(n_baseline=self.n_baseline, **shared),
            GymMCDropout(n_perturbations=self.n_perturbations, **shared),
            GymSWAG(n_perturbations=self.n_perturbations, **shared),
            GymLaplace(n_perturbations=self.n_perturbations,
                       subset_size=self.subset_size,
                       laplace_priors=self.laplace_priors, **shared),
            GymSubspaceInference(n_perturbations=self.n_perturbations, **shared),
        ]
        
        # Add all PJSVD variants
        for scope in ["first", "multi"]:
            for mode in ["affine", "least_squares"]:
                for full in [True, False]:
                    ps = self.perturbation_sizes
                    if scope == "multi":
                        ps = [x / 4 for x in ps]
                    tasks.append(GymPJSVD(
                        subset_size=self.subset_size,
                        n_directions=self.n_directions,
                        n_perturbations=self.n_perturbations,
                        perturbation_sizes=ps,
                        layer_scope=scope,
                        correction_mode=mode,
                        use_full_span=full,
                        **shared
                    ))
        return tasks
