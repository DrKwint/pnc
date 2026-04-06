"""Luigi tasks for Gym environment experiments."""

import time
import json
from pathlib import Path

import jax
import luigi
import jax.numpy as jnp
from flax import nnx
import numpy as np
import tree

from util import (
    _get_activation,
    seed_everything,
    _evaluate_gym,
    _load_gym_data,
    _ps_str,
    _find_pjsvd_directions,
    _split_data,
)
from geometry import CalibrationGeometry
from pjsvd import (
    find_optimal_perturbation_multi_layer,
    find_optimal_perturbation_multi_layer_full,
)
from models import (
    TransitionModel,
    MCDropoutTransitionModel,
    ProbabilisticRegressionModel,
)
from data import collect_data, id_policy_random, OODPolicyWrapper
from training import (
    train_model,
    train_swag_model,
    train_probabilistic_model,
    train_subspace_model,
)
from ensembles import (
    PJSVDEnsemble,
    StandardEnsemble,
    MCDropoutEnsemble,
    SWAGEnsemble,
    LaplaceEnsemble,
    SubspaceInferenceEnsemble,
)
from laplace import compute_kfac_factors
import sys
import os

# Parse --cpu early, BEFORE importing jax, so JAX_PLATFORMS is set before JAX initializes any GPU
if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        ""  # Belt-and-suspenders: hide GPU from CUDA too
    )
    sys.argv.remove("--cpu")
else:
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"


def _posthoc_suffix(enabled: bool) -> str:
    return "_vcal" if enabled else ""


def _sample_member_latents(
    rng: np.random.RandomState,
    n_perturbations: int,
    n_directions: int,
    antithetic_pairing: bool = False,
) -> np.ndarray:
    if not antithetic_pairing:
        return rng.normal(0, 1, size=(n_perturbations, n_directions))

    n_pairs = n_perturbations // 2
    base = rng.normal(0, 1, size=(n_pairs, n_directions))
    paired = np.concatenate([base, -base], axis=0)
    if n_perturbations % 2 == 1:
        extra = rng.normal(0, 1, size=(1, n_directions))
        paired = np.concatenate([paired, extra], axis=0)
    return paired


def _get_next_layer_params(model, layer_idx: int) -> tuple[jax.Array, jax.Array]:
    """Return the original next-layer affine map after the given hidden layer."""
    if hasattr(model, "layers"):
        layer = model.layers[layer_idx]
        return layer.kernel.get_value(), layer.bias.get_value()
    layer = getattr(model, f"l{layer_idx + 1}")
    return layer.kernel.get_value(), layer.bias.get_value()


def _add_uncorrected_local_l2_metrics(
    metrics: dict,
    model,
    ensemble,
    dataset: dict,
    layer_idx: int = 1,
) -> None:
    """Add local hidden-state and next-layer preactivation L2 metrics."""
    from util import get_intermediate_state, compute_l2_distance

    w_next_orig, b_next_orig = _get_next_layer_params(model, layer_idx)

    h_orig_id = get_intermediate_state(model, dataset["id_eval"][0], layer_idx=layer_idx)
    h_ens_id = ensemble.predict_intermediate(dataset["id_eval"][0], layer_idx=layer_idx)
    metrics["uncorrected_l2_id_h"] = compute_l2_distance(h_ens_id, h_orig_id)

    z_orig_id = h_orig_id @ w_next_orig + b_next_orig
    z_ens_id = h_ens_id @ w_next_orig + b_next_orig
    metrics["uncorrected_l2_id_z"] = compute_l2_distance(z_ens_id, z_orig_id)

    # Backward-compatible aggregate alias keeps older report paths working.
    metrics["uncorrected_l2_id"] = metrics["uncorrected_l2_id_h"]

    for reg in ["ood_near", "ood_mid", "ood_far"]:
        h_orig = get_intermediate_state(model, dataset[reg][0], layer_idx=layer_idx)
        h_ens = ensemble.predict_intermediate(dataset[reg][0], layer_idx=layer_idx)
        metrics[f"uncorrected_l2_{reg}_h"] = compute_l2_distance(h_ens, h_orig)

        z_orig = h_orig @ w_next_orig + b_next_orig
        z_ens = h_ens @ w_next_orig + b_next_orig
        metrics[f"uncorrected_l2_{reg}_z"] = compute_l2_distance(z_ens, z_orig)
        metrics[f"uncorrected_l2_{reg}"] = metrics[f"uncorrected_l2_{reg}_h"]


def _probabilistic_output_vector(
    model: ProbabilisticRegressionModel,
    hidden: jax.Array,
    apply_hidden_activation: bool = False,
) -> jax.Array:
    """Map a hidden representation to the concatenated probabilistic outputs."""
    if apply_hidden_activation:
        hidden = model.activation(hidden)

    mean = hidden @ model.mean_layer.kernel.get_value() + model.mean_layer.bias.get_value()
    var_logits = hidden @ model.var_layer.kernel.get_value() + model.var_layer.bias.get_value()
    var = jax.nn.softplus(var_logits) + 1e-6
    return jnp.concatenate([mean, var], axis=-1)


class CollectGymData(luigi.Task):
    """Collect ID (train + eval) and OOD transitions for a gym environment."""

    env = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    seed = luigi.IntParameter(default=0)
    policy_preset = luigi.Parameter(default="")

    def _base(self) -> Path:
        return Path("results") / self.env

    def output(self) -> dict[str, luigi.LocalTarget]:
        b = self._base()
        s = f"seed{self.seed}_steps{self.steps}"
        return {
            "id_train": luigi.LocalTarget(str(b / f"data_id_train_{s}.npz")),
            "id_eval": luigi.LocalTarget(str(b / f"data_id_eval_{s}.npz")),
            "ood_near": luigi.LocalTarget(str(b / f"data_ood_near_{s}.npz")),
            "ood_mid": luigi.LocalTarget(str(b / f"data_ood_mid_{s}.npz")),
            "ood_far": luigi.LocalTarget(str(b / f"data_ood_far_{s}.npz")),
            "ood": luigi.LocalTarget(str(b / f"data_ood_{s}.npz")),
        }

    def run(self) -> None:
        seed_everything(self.seed)
        self._base().mkdir(parents=True, exist_ok=True)

        print(f"\n=== Collecting gym data: {self.env} (steps={self.steps}) ===")
        from data import get_policy_for_regime
        
        regimes = [("id_train", "id"), ("id_eval", "id_eval"), ("ood_near", "ood_near"), ("ood_mid", "ood_mid"), ("ood_far", "ood_far")]
        for i, (out_key, reg) in enumerate(regimes):
            policy_fn, metadata = get_policy_for_regime(self.env, reg.replace("_eval", ""), preset=self.policy_preset, strict=True)
            inputs, targets = collect_data(
                self.env, self.steps, policy_fn, seed=self.seed + i
            )
            out_path = self.output()[out_key].path
            np.savez(
                out_path,
                inputs=np.array(inputs),
                targets=np.array(targets),
            )
            metadata["environment"] = self.env
            metadata["regime"] = reg
            metadata["seed"] = self.seed + i
            with open(out_path + ".json", "w") as f:
                json.dump(metadata, f, indent=2)
            if out_key == "ood_far":
                np.savez(
                    self.output()["ood"].path,
                    inputs=np.array(inputs),
                    targets=np.array(targets),
                )
        print("Data saved.")


class GymStandardEnsemble(luigi.Task):
    env = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    n_baseline = luigi.IntParameter(default=5)
    seed = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.env
                / f"standard_ensemble{calib_str}_n{self.n_baseline}_act-{self.activation}_seed{self.seed}.json"
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        dataset = _load_gym_data(self.input())
        inputs_id, targets_id = dataset["id_train"]
        inputs_id_eval, targets_id_eval = dataset["id_eval"]
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(
            f"\n=== EXP 2: Standard Ensemble ({self.env}, n={self.n_baseline}, act={self.activation}) ==="
        )
        t0 = time.time()
        models = []
        for i in range(self.n_baseline):
            m = ProbabilisticRegressionModel(
                inputs_id.shape[1],
                targets_id.shape[1],
                nnx.Rngs(params=self.seed + i),
                hidden_dims=[64, 64],
                activation=act_fn,
            )
            print(f"  Training model {i + 1}/{self.n_baseline}...")
            m = train_probabilistic_model(
                m, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64
            )
            models.append(m)

        # Ensure models are fully trained/initialized
        for m in models:
            for p in jax.tree_util.tree_leaves(nnx.state(m)):
                if hasattr(p, "block_until_ready"):
                    p.block_until_ready()

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")

        ensemble = StandardEnsemble(models)
        metrics = _evaluate_gym(
            "Standard Ensemble",
            ensemble,
            dataset,
            sidecar_path=self.output().path.replace(".json", ".npz"),
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
        )
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class GymMCDropout(luigi.Task):
    env = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    n_perturbations = luigi.IntParameter(default=1000)
    seed = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.env
                / f"mc_dropout{calib_str}_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        dataset = _load_gym_data(self.input())
        inputs_id, targets_id = dataset["id_train"]
        inputs_id_eval, targets_id_eval = dataset["id_eval"]
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(
            f"\n=== EXP 3: MC Dropout ({self.env}, n={self.n_perturbations}, act={self.activation}) ==="
        )
        t0 = time.time()
        model = MCDropoutTransitionModel(
            inputs_id.shape[1],
            targets_id.shape[1],
            nnx.Rngs(params=self.seed, dropout=self.seed + 1),
            dropout_rate=0.1,
            activation=act_fn,
        )
        model = train_model(model, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64)

        # Ensure model is fully trained
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, "block_until_ready"):
                p.block_until_ready()

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")

        ensemble = MCDropoutEnsemble(model, self.n_perturbations)
        metrics = _evaluate_gym(
            "MC Dropout",
            ensemble,
            dataset,
            sidecar_path=self.output().path.replace(".json", ".npz"),
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
        )
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class GymSWAG(luigi.Task):
    env = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    n_perturbations = luigi.IntParameter(default=1000)
    seed = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.env
                / f"swag{calib_str}_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        dataset = _load_gym_data(self.input())
        inputs_id, targets_id = dataset["id_train"]
        inputs_id_eval, targets_id_eval = dataset["id_eval"]
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(
            f"\n=== EXP 4: SWAG ({self.env}, n={self.n_perturbations}, act={self.activation}) ==="
        )
        t0 = time.time()
        model = ProbabilisticRegressionModel(
            inputs_id.shape[1],
            targets_id.shape[1],
            nnx.Rngs(params=self.seed),
            hidden_dims=[64, 64],
            activation=act_fn,
        )
        model, swag_mean, swag_var = train_swag_model(
            model, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64, swag_start=1000
        )

        # Ensure SWAG stats are ready
        jax.block_until_ready(swag_mean)
        jax.block_until_ready(swag_var)

        train_time = time.time() - t0
        print(f"Training time: {train_time:.2f}s")

        ensemble = SWAGEnsemble(model, swag_mean, swag_var, self.n_perturbations)
        metrics = _evaluate_gym(
            "SWAG",
            ensemble,
            dataset,
            sidecar_path=self.output().path.replace(".json", ".npz"),
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
        )
        metrics["train_time"] = train_time

        _add_uncorrected_local_l2_metrics(metrics, model, ensemble, dataset, layer_idx=1)

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class GymLaplace(luigi.Task):
    env = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    n_perturbations = luigi.IntParameter(default=1000)
    subset_size = luigi.IntParameter(default=4096)
    laplace_priors = luigi.ListParameter(default=[1.0, 5.0, 10.0, 50.0, 100.0])
    seed = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        priors_str = _ps_str(self.laplace_priors)
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.env
                / f"laplace{calib_str}_priors{priors_str}_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}.json"
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        dataset = _load_gym_data(self.input())
        inputs_id, targets_id = dataset["id_train"]
        inputs_id_eval, targets_id_eval = dataset["id_eval"]
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(
            f"\n=== EXP 5: Laplace ({self.env}, n={self.n_perturbations}, act={self.activation}) ==="
        )
        t0 = time.time()
        model = TransitionModel(
            inputs_id.shape[1],
            targets_id.shape[1],
            nnx.Rngs(params=self.seed),
            activation=act_fn,
        )
        model = train_model(model, x_tr, y_tr, x_va, y_va, steps=2000, batch_size=64)

        actual_subset = min(len(inputs_id), self.subset_size)
        subset_idx = np.random.choice(len(inputs_id), actual_subset, replace=False)
        X_sub = inputs_id[subset_idx]
        Y_sub = targets_id[subset_idx]

        print("Computing KFAC Factors...")
        factors = compute_kfac_factors(model, X_sub, Y_sub, batch_size=128)

        # Ensure factors are ready
        for f in jax.tree_util.tree_leaves(factors):
            if hasattr(f, "block_until_ready"):
                f.block_until_ready()

        setup_time = time.time() - t0
        print(f"Training + KFAC time: {setup_time:.2f}s")

        all_metrics = {}
        base_npz = self.output().path.replace(".json", "")
        for prior in self.laplace_priors:
            lap_ens = LaplaceEnsemble(
                model=model,
                kfac_factors=factors,
                prior_precision=prior,
                n_models=self.n_perturbations,
                data_size=actual_subset,
            )
            m = _evaluate_gym(
                f"Laplace (prior={prior})",
                lap_ens,
                dataset,
                sidecar_path=f"{base_npz}_prior{prior}.npz",
                calibration_data=(x_va, y_va),
                posthoc_calibrate=self.posthoc_calibrate,
            )
            m["train_time"] = setup_time  # Consistent, non-cumulative setup time

            _add_uncorrected_local_l2_metrics(m, model, lap_ens, dataset, layer_idx=1)

            all_metrics[str(prior)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class GymPJSVD(luigi.Task):
    env = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    subset_size = luigi.IntParameter(default=4096)
    n_directions = luigi.IntParameter(default=40)
    n_perturbations = luigi.IntParameter(default=1000)
    perturbation_sizes = luigi.ListParameter(default=[20.0, 40.0, 80.0, 160.0])
    seed = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")
    correction_mode = luigi.ChoiceParameter(
        default="affine", choices=["affine", "least_squares"]
    )
    layer_scope = luigi.ChoiceParameter(default="first", choices=["first", "multi"])
    pjsvd_family = luigi.ChoiceParameter(default="low", choices=["low", "random"])
    antithetic_pairing = luigi.BoolParameter(default=False)
    member_radius_distribution = luigi.ChoiceParameter(
        default="fixed", choices=["fixed", "lognormal", "two_point"]
    )
    member_radius_std = luigi.FloatParameter(default=0.0)
    member_radius_values = luigi.ListParameter(default=[])
    safe_subspace_backend = luigi.ChoiceParameter(
        default="activation_covariance",
        choices=["activation_covariance", "projected_residual"],
    )
    use_full_span = luigi.BoolParameter(default=False)
    hidden_dims = luigi.ListParameter(default=[64, 64])
    probabilistic_base_model = luigi.BoolParameter(default=False)
    posthoc_calibrate = luigi.BoolParameter(default=False)
    compute_l2 = luigi.BoolParameter(default=True)
    compute_geometry = luigi.BoolParameter(default=True)
    geometry_rank = luigi.IntParameter(default=None)

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        full_str = "_full" if self.use_full_span else ""
        l2_str = "" if self.compute_l2 else "_nol2"
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        prob_str = "_prob" if self.probabilistic_base_model else ""
        anti_str = "_anti" if self.antithetic_pairing else ""
        if self.member_radius_distribution == "fixed":
            radius_str = ""
        else:
            radius_str = (
                f"_mr-{self.member_radius_distribution}"
                f"_mrs{self.member_radius_std:g}"
            )
            if len(self.member_radius_values) > 0:
                radius_str += f"_mrv{_ps_str(self.member_radius_values)}"
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.env
                / f"pjsvd_{self.layer_scope}_{self.correction_mode}_{self.pjsvd_family}_{self.safe_subspace_backend}{full_str}{l2_str}{calib_str}{prob_str}{anti_str}{radius_str}_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_act-{self.activation}_seed{self.seed}.json"
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        dataset = _load_gym_data(self.input())
        inputs_id, targets_id = dataset["id_train"]
        inputs_id_eval, targets_id_eval = dataset["id_eval"]
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(
            f"\n=== Gym PJSVD ({self.env}, scope={self.layer_scope}, mode={self.correction_mode}, full={self.use_full_span}) ==="
        )
        t_start = time.time()
        if self.probabilistic_base_model:
            model = ProbabilisticRegressionModel(
                inputs_id.shape[1],
                targets_id.shape[1],
                nnx.Rngs(params=self.seed),
                hidden_dims=[64, 64],
                activation=act_fn,
            )
        else:
            model = TransitionModel(
                inputs_id.shape[1],
                targets_id.shape[1],
                nnx.Rngs(params=self.seed),
                activation=act_fn,
            )
        # model depth check for multi-layer
        if self.layer_scope == "multi" and len(self.hidden_dims) < 2:
            print("Skipping Multi-Layer PJSVD: model too shallow.")
            Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output().path, "w") as f:
                json.dump({"skipped": True}, f)
            return

        if self.probabilistic_base_model and self.correction_mode == "affine" and self.layer_scope == "multi":
            print("Skipping probabilistic multi-layer affine PJSVD: affine correction expects a single next layer.")
            Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output().path, "w") as f:
                json.dump({"skipped": True}, f)
            return

        if self.probabilistic_base_model:
            model = train_probabilistic_model(model, x_tr, y_tr, x_va, y_va)
        else:
            model = train_model(model, x_tr, y_tr, x_va, y_va)

        # End of base training
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, "block_until_ready"):
                p.block_until_ready()

        print(f"Base model training time: {time.time() - t_start:.2f}s")

        actual_subset = min(len(inputs_id), self.subset_size)
        subset_idx = np.random.choice(len(inputs_id), actual_subset, replace=False)
        X_sub = inputs_id[subset_idx]

        if self.probabilistic_base_model:
            W1 = model.layers[0].kernel.get_value()
            b1 = model.layers[0].bias.get_value()
            W2 = model.layers[1].kernel.get_value()
            b2 = model.layers[1].bias.get_value()
        else:
            W1 = model.l1.kernel.get_value()
            b1 = model.l1.bias.get_value()
            W2 = model.l2.kernel.get_value()
            b2 = model.l2.bias.get_value()

        if self.layer_scope == "first":

            def model_fn(w):
                return act_fn(X_sub @ w + b1)

            # Family-specific selection: low (default), random, all
            if self.pjsvd_family == "random":
                D = W1.size
                rng = np.random.RandomState(self.seed)
                rand_dirs = rng.normal(size=(self.n_directions, D)).astype(np.float32)
                rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12
                v_opts = jnp.array(rand_dirs)
                sigmas = np.ones(self.n_directions, dtype=np.float32)
            else:  # low singular
                t_dir = time.time()
                from pnc import find_pnc_subspace_lanczos

                def get_Y_fn(w, x):
                    return act_fn(x @ w + b1)

                v_opts, sigmas = find_pnc_subspace_lanczos(
                    get_Y_fn,
                    W1,
                    [X_sub],
                    self.n_directions,
                    backend=self.safe_subspace_backend,
                    seed=self.seed,
                )
                v_opts.block_until_ready()
                print(f"Direction search time: {time.time() - t_dir:.2f}s")

            h_old = model_fn(W1)
            mu_old = jnp.mean(h_old, axis=0)
            std_old = jnp.std(h_old, axis=0)

            correction_params = {}
            if self.correction_mode == "affine":
                if self.probabilistic_base_model:
                    raise ValueError("Probabilistic first-layer affine PJSVD is not supported.")
                correction_params = {
                    "mu_old": mu_old,
                    "std_old": std_old,
                    "next_w": W2,
                    "next_b": b2,
                }
            else:
                correction_params = {"target_act": h_old @ W2 + b2}

            layer_params = {"l1": {"W": W1, "b": b1}}
            perturbed_layers = ["l1"]

        else:  # multi
            if self.probabilistic_base_model:
                W_mean = model.mean_layer.kernel.get_value()
                b_mean = model.mean_layer.bias.get_value()
                W_var = model.var_layer.kernel.get_value()
                b_var = model.var_layer.bias.get_value()
            else:
                W3 = model.l3.kernel.get_value()
                b3 = model.l3.bias.get_value()

            def model_fn_ws(ws):
                w1, w2 = ws
                h1 = act_fn(X_sub @ w1 + b1)
                h2 = act_fn(h1 @ w2 + b2)
                return h2

            # Family-specific selection: low (default), random, all
            if self.pjsvd_family == "random":
                D = W1.size + W2.size
                rng = np.random.RandomState(self.seed)
                rand_dirs = rng.normal(size=(self.n_directions, D)).astype(np.float32)
                rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True) + 1e-12
                v_opts = jnp.array(rand_dirs)
                sigmas = np.ones(self.n_directions, dtype=np.float32)
            else:
                # low singular (default): find via lanczos over flattened weights
                w_joint_flat = jnp.concatenate([W1.flatten(), W2.flatten()])

                def get_Y_fn(w_flat, x):
                    w1 = w_flat[: W1.size].reshape(W1.shape)
                    w2 = w_flat[W1.size :].reshape(W2.shape)
                    h1 = act_fn(x @ w1 + b1)
                    return act_fn(h1 @ w2 + b2)

                t_dir = time.time()
                from pnc import find_pnc_subspace_lanczos

                v_opts, sigmas = find_pnc_subspace_lanczos(
                    get_Y_fn,
                    w_joint_flat,
                    [X_sub],
                    self.n_directions,
                    backend=self.safe_subspace_backend,
                    seed=self.seed,
                )
                v_opts.block_until_ready()
                print(f"ML Direction search time: {time.time() - t_dir:.2f}s")

            h_old = model_fn_ws([W1, W2])
            mu_old = jnp.mean(h_old, axis=0)
            std_old = jnp.std(h_old, axis=0)

            correction_params = {}
            if self.correction_mode == "affine":
                if self.probabilistic_base_model:
                    raise ValueError("Probabilistic multi-layer affine PJSVD is not supported.")
                correction_params = {
                    "mu_old": mu_old,
                    "std_old": std_old,
                    "next_w": W3,
                    "next_b": b3,
                }
            else:
                if self.probabilistic_base_model:
                    correction_params = {"target_act": h_old}
                else:
                    correction_params = {"target_act": h_old @ W3 + b3}

            layer_params = {"l1": {"W": W1, "b": b1}, "l2": {"W": W2, "b": b2}}
            perturbed_layers = ["l1", "l2"]

        if self.compute_geometry:
            # We build the calibration affine geometry on the unperturbed activations from the calibration set.
            geom = CalibrationGeometry(h_old, retain_rank=self.geometry_rank)
        else:
            geom = None

        setup_time = (
            time.time() - t_start
        )  # Total setup time (base train + direction search)

        rng = np.random.RandomState(self.seed)
        all_z = _sample_member_latents(
            rng,
            self.n_perturbations,
            self.n_directions,
            antithetic_pairing=self.antithetic_pairing,
        )

        all_metrics = {}
        base_npz = self.output().path.replace(".json", "")
        for p_size in self.perturbation_sizes:
            ens = PJSVDEnsemble(
                base_model=model,
                v_opts=v_opts,
                sigmas=sigmas,
                z_coeffs=all_z,
                perturbation_scale=p_size,
                X_sub=X_sub,
                layers=perturbed_layers,
                correction_mode=self.correction_mode,
                member_radius_distribution=self.member_radius_distribution,
                member_radius_std=self.member_radius_std,
                member_radius_values=self.member_radius_values,
                member_radius_seed=self.seed,
                activation=act_fn,
                layer_params=layer_params,
                correction_params=correction_params,
                tail_is_hidden=self.probabilistic_base_model and self.layer_scope == "multi",
            )
            m = _evaluate_gym(
                f"PJSVD-{self.layer_scope}-{self.correction_mode} (size={p_size})",
                ens,
                dataset,
                sidecar_path=f"{base_npz}_ps{p_size}.npz",
                calibration_data=(x_va, y_va),
                posthoc_calibrate=self.posthoc_calibrate,
            )
            m["train_time"] = setup_time  # Consistent, non-cumulative setup time
            m["antithetic_pairing"] = bool(self.antithetic_pairing)
            m["antithetic_has_unpaired_sample"] = bool(self.antithetic_pairing and (self.n_perturbations % 2 == 1))
            member_radii = np.asarray(ens.member_radii)
            m["member_radius_distribution"] = self.member_radius_distribution
            m["member_radius_mean"] = float(member_radii.mean())
            m["member_radius_std"] = float(member_radii.std())
            m["member_radius_min"] = float(member_radii.min())
            m["member_radius_max"] = float(member_radii.max())

            if self.compute_l2 or self.compute_geometry:
                # Optional intermediate-state analysis for perturbation behavior.
                from util import get_intermediate_state, compute_l2_distance

                layer_idx = 1 if self.layer_scope == "first" else 2
                h_orig_id = get_intermediate_state(model, dataset["id_eval"][0], layer_idx=layer_idx)
                h_perts_id, z_nexts_id = ens.predict_intermediate_and_corrected(dataset["id_eval"][0])
                geom_out = {}
                vmap_dist = None

                if self.probabilistic_base_model:
                    if self.layer_scope == "first":
                        w_next_orig = model.layers[1].kernel.get_value()
                        b_next_orig = model.layers[1].bias.get_value()
                        z_orig_hidden_id = h_orig_id @ w_next_orig + b_next_orig
                        z_uncorr_hidden_id = h_perts_id @ w_next_orig + b_next_orig
                        z_orig_id = _probabilistic_output_vector(
                            model, z_orig_hidden_id, apply_hidden_activation=True
                        )
                        z_uncorr_id = _probabilistic_output_vector(
                            model, z_uncorr_hidden_id, apply_hidden_activation=True
                        )
                        z_corr_id = _probabilistic_output_vector(
                            model, z_nexts_id, apply_hidden_activation=True
                        )
                    else:
                        z_orig_id = _probabilistic_output_vector(model, h_orig_id)
                        z_uncorr_id = _probabilistic_output_vector(model, h_perts_id)
                        z_corr_id = _probabilistic_output_vector(model, z_nexts_id)
                else:
                    w_next_orig = (
                        model.l2.kernel.get_value()
                        if self.layer_scope == "first"
                        else model.l3.kernel.get_value()
                    )
                    b_next_orig = (
                        model.l2.bias.get_value()
                        if self.layer_scope == "first"
                        else model.l3.bias.get_value()
                    )
                    z_orig_id = h_orig_id @ w_next_orig + b_next_orig
                    z_uncorr_id = h_perts_id @ w_next_orig + b_next_orig
                    z_corr_id = z_nexts_id

                if self.compute_l2:
                    m["uncorrected_l2_id_h"] = compute_l2_distance(h_perts_id, h_orig_id)
                    m["uncorrected_l2_id_z"] = compute_l2_distance(z_uncorr_id, z_orig_id)
                    m["corrected_l2_id_z"] = compute_l2_distance(z_corr_id, z_orig_id)

                if self.compute_geometry and geom is not None:
                    vmap_dist = jax.vmap(geom.distance)
                    geom_out["geom_dist_id"] = np.array(vmap_dist(h_perts_id))
                    geom_out["uncorr_l2_id"] = np.array(jnp.linalg.norm(z_uncorr_id - z_orig_id, axis=-1))
                    geom_out["corr_l2_id"] = np.array(jnp.linalg.norm(z_corr_id - z_orig_id, axis=-1))

                for reg in ["ood_near", "ood_mid", "ood_far"]:
                    h_o = get_intermediate_state(model, dataset[reg][0], layer_idx=layer_idx)
                    h_p, z_n = ens.predict_intermediate_and_corrected(dataset[reg][0])

                    if self.probabilistic_base_model:
                        if self.layer_scope == "first":
                            z_orig_hidden = h_o @ w_next_orig + b_next_orig
                            z_uncorr_hidden = h_p @ w_next_orig + b_next_orig
                            z_o = _probabilistic_output_vector(
                                model, z_orig_hidden, apply_hidden_activation=True
                            )
                            z_u = _probabilistic_output_vector(
                                model, z_uncorr_hidden, apply_hidden_activation=True
                            )
                            z_c = _probabilistic_output_vector(
                                model, z_n, apply_hidden_activation=True
                            )
                        else:
                            z_o = _probabilistic_output_vector(model, h_o)
                            z_u = _probabilistic_output_vector(model, h_p)
                            z_c = _probabilistic_output_vector(model, z_n)
                    else:
                        z_o = h_o @ w_next_orig + b_next_orig
                        z_u = h_p @ w_next_orig + b_next_orig
                        z_c = z_n

                    if self.compute_l2:
                        m[f"uncorrected_l2_{reg}_h"] = compute_l2_distance(h_p, h_o)
                        m[f"uncorrected_l2_{reg}_z"] = compute_l2_distance(z_u, z_o)
                        m[f"corrected_l2_{reg}_z"] = compute_l2_distance(z_c, z_o)

                    if self.compute_geometry and geom is not None and vmap_dist is not None:
                        geom_out[f"geom_dist_{reg}"] = np.array(vmap_dist(h_p))
                        geom_out[f"uncorr_l2_{reg}"] = np.array(jnp.linalg.norm(z_u - z_o, axis=-1))
                        geom_out[f"corr_l2_{reg}"] = np.array(jnp.linalg.norm(z_c - z_o, axis=-1))

                if self.compute_geometry and geom is not None:
                    geom_npz = f"{base_npz}_ps{p_size}_geometry.npz"
                    np.savez(geom_npz, **geom_out)

            all_metrics[str(p_size)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class GymSubspaceInference(luigi.Task):
    env = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    n_perturbations = luigi.IntParameter(default=1000)
    seed = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")
    temperature = luigi.FloatParameter(default=0.0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CollectGymData(env=self.env, steps=self.steps, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.env
                / f"subspace_inference{calib_str}_n{self.n_perturbations}_act-{self.activation}_seed{self.seed}_T{self.temperature}.json"
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        act_fn = _get_activation(self.activation)
        dataset = _load_gym_data(self.input())
        inputs_id, targets_id = dataset["id_train"]
        inputs_id_eval, targets_id_eval = dataset["id_eval"]
        x_tr, y_tr, x_va, y_va = _split_data(inputs_id, targets_id)

        print(
            f"\n=== EXP: Subspace Inference ({self.env}, n={self.n_perturbations}, act={self.activation}) ==="
        )
        t0 = time.time()
        model = ProbabilisticRegressionModel(
            inputs_id.shape[1],
            targets_id.shape[1],
            nnx.Rngs(params=self.seed),
            hidden_dims=[64, 64],
            activation=act_fn,
        )
        model, swag_mean, pca_components = train_subspace_model(
            model,
            x_tr,
            y_tr,
            x_va,
            y_va,
            steps=2000,
            batch_size=64,
            swag_start=1000,
            max_rank=20,
        )

        T = (
            self.temperature
            if self.temperature > 0
            else 2.0 * float(np.sqrt(len(inputs_id)))
        )
        ensemble = SubspaceInferenceEnsemble(
            model,
            swag_mean,
            pca_components,
            self.n_perturbations,
            temperature=T,
            X_train=inputs_id,
            Y_train=targets_id,
            use_ess=True,
            is_classification=False,
        )

        # Ensure PCA or other components are ready
        jax.block_until_ready(swag_mean)
        for c in jax.tree_util.tree_leaves(pca_components):
            if hasattr(c, "block_until_ready"):
                c.block_until_ready()

        setup_time = time.time() - t0
        print(f"Training time: {setup_time:.2f}s")

        metrics = _evaluate_gym(
            "Subspace Inference",
            ensemble,
            dataset,
            sidecar_path=self.output().path.replace(".json", ".npz"),
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
        )
        metrics["train_time"] = setup_time

        _add_uncorrected_local_l2_metrics(metrics, model, ensemble, dataset, layer_idx=1)

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class AllGymExperiments(luigi.WrapperTask):
    """Umbrella task that requires all gym sub-experiments."""

    env = luigi.Parameter()
    steps = luigi.IntParameter(default=10000)
    subset_size = luigi.IntParameter(default=10000)
    n_directions = luigi.IntParameter(default=40)
    n_perturbations = luigi.IntParameter(default=128)
    n_baseline = luigi.IntParameter(default=5)
    perturbation_sizes = luigi.ListParameter(default=[2, 4, 8, 16, 32, 64, 128])
    laplace_priors = luigi.ListParameter(
        default=[1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    )
    seed = luigi.IntParameter(default=0)
    activation = luigi.Parameter(default="relu")
    pjsvd_family = luigi.ChoiceParameter(default="low", choices=["low", "random"])
    antithetic_pairing = luigi.BoolParameter(default=False)
    member_radius_distribution = luigi.ChoiceParameter(
        default="fixed", choices=["fixed", "lognormal", "two_point"]
    )
    member_radius_std = luigi.FloatParameter(default=0.0)
    member_radius_values = luigi.ListParameter(default=[])
    posthoc_calibrate = luigi.BoolParameter(default=False)
    probabilistic_pjsvd = luigi.BoolParameter(default=False)

    def requires(self) -> list[luigi.Task]:
        shared = dict(
            env=self.env,
            steps=self.steps,
            seed=self.seed,
            activation=self.activation,
            posthoc_calibrate=self.posthoc_calibrate,
        )
        tasks = [
            GymStandardEnsemble(n_baseline=self.n_baseline, **shared),
            GymMCDropout(n_perturbations=self.n_perturbations, **shared),
            GymSWAG(n_perturbations=self.n_perturbations, **shared),
            GymLaplace(
                n_perturbations=self.n_perturbations,
                subset_size=self.subset_size,
                laplace_priors=self.laplace_priors,
                **shared,
            ),
            GymSubspaceInference(n_perturbations=self.n_perturbations, **shared),
        ]

        # Add all PJSVD variants
        for pjsvd_family in ["low", "random"]:
            ps = self.perturbation_sizes
            tasks.append(
                GymPJSVD(
                    subset_size=self.subset_size,
                    n_directions=self.n_directions,
                    n_perturbations=self.n_perturbations,
                    perturbation_sizes=ps,
                    pjsvd_family=pjsvd_family,
                    antithetic_pairing=self.antithetic_pairing,
                    layer_scope="multi",
                    correction_mode="least_squares",
                    use_full_span=True,
                    safe_subspace_backend="projected_residual",
                    probabilistic_base_model=self.probabilistic_pjsvd,
                    member_radius_distribution=self.member_radius_distribution,
                    member_radius_std=self.member_radius_std,
                    member_radius_values=self.member_radius_values,
                    **shared,
                )
            )
        return tasks


class AllGymExperimentsMultiSeed(luigi.WrapperTask):
    """Umbrella task that requires AllGymExperiments across multiple seeds."""

    envs = luigi.ListParameter(default=["HalfCheetah-v5", "Hopper-v5", "Ant-v5"])
    steps = luigi.IntParameter(default=10000)
    subset_size = luigi.IntParameter(default=10000)
    n_directions = luigi.IntParameter(default=40)
    n_perturbations = luigi.IntParameter(default=100)
    n_baseline = luigi.IntParameter(default=10)
    perturbation_sizes = luigi.ListParameter(default=[1, 2, 4, 8, 16, 32, 64])
    laplace_priors = luigi.ListParameter(
        default=[10.0, 100.0, 1000.0, 10000.0, 100000.0]
    )
    seeds = luigi.ListParameter(default=[0, 10, 200])
    activation = luigi.Parameter(default="relu")
    pjsvd_family = luigi.ChoiceParameter(default="low", choices=["low", "random"])
    antithetic_pairing = luigi.BoolParameter(default=False)
    member_radius_distribution = luigi.ChoiceParameter(
        default="fixed", choices=["fixed", "lognormal", "two_point"]
    )
    member_radius_std = luigi.FloatParameter(default=0.0)
    member_radius_values = luigi.ListParameter(default=[])
    posthoc_calibrate = luigi.BoolParameter(default=False)
    probabilistic_pjsvd = luigi.BoolParameter(default=False)

    def requires(self) -> list[luigi.Task]:
        return tree.flatten([
            [
                AllGymExperiments(
                    env=env,
                    steps=self.steps,
                    subset_size=self.subset_size,
                    n_directions=self.n_directions,
                    n_perturbations=self.n_perturbations,
                    n_baseline=self.n_baseline,
                    perturbation_sizes=self.perturbation_sizes,
                    laplace_priors=self.laplace_priors,
                    seed=seed,
                    activation=self.activation,
                    pjsvd_family=self.pjsvd_family,
                    antithetic_pairing=self.antithetic_pairing,
                    member_radius_distribution=self.member_radius_distribution,
                    member_radius_std=self.member_radius_std,
                    member_radius_values=self.member_radius_values,
                    posthoc_calibrate=self.posthoc_calibrate,
                    probabilistic_pjsvd=self.probabilistic_pjsvd,
                )
                for env in self.envs
            ]
            for seed in self.seeds
        ])
