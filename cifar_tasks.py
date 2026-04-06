"""Luigi tasks for CIFAR-10/100 baselines and PnC experiments on PreAct ResNet-18."""
import itertools
import time
import json
import pickle
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

import luigi
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from jaxtyping import Array

from util import (
    seed_everything,
    _evaluate_cifar,
    _evaluate_cifar_logits,
    _fit_posthoc_temperature,
    _make_cifar_protocol_splits,
    _predict_cifar_logits,
    _ps_str,
    _split_data,
)
from models import PreActResNet18, MCDropoutPreActResNet18, _PreActBasicBlock
from data import load_cifar10, load_cifar100
from data import load_openood_cifar_benchmark
from training import train_resnet_model, train_resnet_swag
from ensembles import SWAGEnsemble, PnCEnsemble, MultiBlockPnCEnsemble, LLLAEnsemble
from openood_eval import evaluate_openood_cifar

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def preact_resnet18_block_indices() -> list[tuple[int, int]]:
    """Ordered (stage_idx, block_idx) for all 8 residual blocks (stage1..stage4, 2 each)."""
    return [(s, b) for s in range(4) for b in range(2)]


def _posthoc_suffix(enabled: bool) -> str:
    return "_vcal" if enabled else ""


@nnx.jit
def _forward_block_jit(blk: _PreActBasicBlock, h: jax.Array, w1: jax.Array) -> tuple[jax.Array, jax.Array]:
    """JITed forward pass for one block, returning (h_next, T_conv2)."""
    out_bn1 = blk.bn1(h, use_running_average=True)
    out_relu1 = jax.nn.relu(out_bn1)
    # y_raw uses the passed-in w1 (which might be orig OR perturbed later)
    y_raw = jax.lax.conv_general_dilated(
        lhs=out_relu1,
        rhs=w1.transpose(3, 2, 0, 1),
        window_strides=blk.conv1.strides,
        padding="SAME",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    y_bn2 = blk.bn2(y_raw, use_running_average=True)
    y_relu2 = jax.nn.relu(y_bn2)
    t = blk.conv2(y_relu2)
    
    if blk.downsample is not None:
        identity = blk.downsample(out_relu1)
    else:
        identity = h
    return (t + identity), t

def compute_cifar_block_preacts(
    model: PreActResNet18,
    X_data: jax.Array,
    chunk_sz: int,
    target_stage_idx: int,
    target_block_idx: int,
    w1_orig: jax.Array,
) -> tuple[list[jax.Array], list[jax.Array]]:
    """
    Return (pre_act_chunks, T_orig_chunks) for the target block: inputs to bn1,
    and conv2 outputs on the unperturbed forward (MAP), chunked.
    """
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    n_ch = int(np.ceil(len(X_data) / chunk_sz))
    
    pa_chunks, t_chunks = [], []
    
    # Process through stem once
    @nnx.jit
    def run_stem_jit(x):
        return model.stem(x)
    
    for i in range(n_ch):
        x_batch = X_data[i * chunk_sz : (i + 1) * chunk_sz]
        h = run_stem_jit(x_batch)
        for s_idx, stage in enumerate(stages):
            for b_idx, blk in enumerate(stage):
                if s_idx == target_stage_idx and b_idx == target_block_idx:
                    pa_chunks.append(h)
                    _, t = _forward_block_jit(blk, h, w1_orig)
                    t_chunks.append(t)
                    break
                # Only run the full block if we haven't reached target
                h, _ = _forward_block_jit(blk, h, blk.conv1.kernel.value)
            if s_idx == target_stage_idx:
                break
    return pa_chunks, t_chunks


def _load_cifar_dataset(dataset: str):
    """Load CIFAR-10 or CIFAR-100 and return `(x_train, y_train, x_test, y_test, n_classes)`."""
    if dataset.lower() == 'cifar10':
        x_tr, y_tr, x_te, y_te = load_cifar10()
        n_cls = 10
    elif dataset.lower() == 'cifar100':
        x_tr, y_tr, x_te, y_te = load_cifar100()
        n_cls = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return x_tr, y_tr, x_te, y_te, n_cls


def preact_resnet18_block_indices() -> list[tuple[int, int]]:
    """Ordered `(stage_idx, block_idx)` pairs for all 8 PreAct ResNet-18 residual blocks."""
    return [(stage_idx, block_idx) for stage_idx in range(4) for block_idx in range(2)]


def make_cifar_block_get_Y_fn(target_blk):
    """Return conv2-design features for a perturbed block's conv1 weights."""
    def get_Y_fn(w1, h_in):
        out_bn1 = target_blk.bn1(h_in, use_running_average=True)
        out_relu1 = jax.nn.relu(out_bn1)
        y_raw = jax.lax.conv_general_dilated(
            lhs=out_relu1,
            rhs=w1.transpose(3, 2, 0, 1),
            window_strides=tuple(target_blk.conv1.strides),
            padding="SAME",
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        y_bn2 = target_blk.bn2(y_raw, use_running_average=True)
        y_relu2 = jax.nn.relu(y_bn2)
        from pnc import extract_patches
        kh = target_blk.conv2.kernel_size[0]
        stride = target_blk.conv2.strides[0]
        return extract_patches(y_relu2, k=kh, strides=stride)

    return get_Y_fn


def _fmt_recipe_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _cifar_recipe_suffix(task) -> str:
    aug = "fc"
    if getattr(task, "cutout_size", 0) > 0:
        aug += f"co{int(task.cutout_size)}"
    else:
        aug += "co0"

    nesterov = "n1" if getattr(task, "nesterov", True) else "n0"
    return (
        f"_opt{task.optimizer}"
        f"_lr{task.lr:.0e}"
        f"_wd{task.weight_decay:.0e}"
        f"_bs{task.batch_size}"
        f"_wu{task.warmup_epochs}"
        f"_mom{_fmt_recipe_float(task.momentum)}"
        f"_{nesterov}"
        f"_aug{aug}"
        f"_ls{_fmt_recipe_float(task.label_smoothing)}"
    )


def _cifar_task_kwargs(task) -> dict:
    return {
        "batch_size": task.batch_size,
        "lr": task.lr,
        "weight_decay": task.weight_decay,
        "optimizer": task.optimizer,
        "momentum": task.momentum,
        "nesterov": task.nesterov,
        "warmup_epochs": task.warmup_epochs,
        "cutout_size": task.cutout_size,
        "label_smoothing": task.label_smoothing,
    }


def _cifar_trainer_kwargs(task) -> dict:
    return {
        "batch_size": task.batch_size,
        "lr": task.lr,
        "weight_decay": task.weight_decay,
        "optimizer_name": task.optimizer,
        "momentum": task.momentum,
        "nesterov": task.nesterov,
        "warmup_epochs": task.warmup_epochs,
        "cutout_size": task.cutout_size,
        "label_smoothing": task.label_smoothing,
    }


def _cifar_swag_suffix(task, *, include_bn_refresh: bool) -> str:
    suffix = (
        f"_sws{task.swag_start_epoch}"
        f"_swf{task.swag_collect_freq}"
        f"_swr{task.swag_max_rank}"
    )
    if include_bn_refresh:
        suffix += f"_bnr{int(task.swag_use_bn_refresh)}_bns{task.bn_refresh_subset_size}"
    return suffix


class CIFARRecipeMixin:
    batch_size = luigi.IntParameter(default=128)
    lr = luigi.FloatParameter(default=0.1)
    weight_decay = luigi.FloatParameter(default=5e-4)
    optimizer = luigi.Parameter(default="sgd")
    momentum = luigi.FloatParameter(default=0.9)
    nesterov = luigi.BoolParameter(default=True)
    warmup_epochs = luigi.IntParameter(default=5)
    cutout_size = luigi.IntParameter(default=8)
    label_smoothing = luigi.FloatParameter(default=0.0)

    def train_recipe_suffix(self) -> str:
        return _cifar_recipe_suffix(self)

    def task_recipe_kwargs(self) -> dict:
        return _cifar_task_kwargs(self)

    def trainer_recipe_kwargs(self) -> dict:
        return _cifar_trainer_kwargs(self)


class CIFAROpenOODMixin:
    openood_root = luigi.Parameter(default="openood_data")
    openood_max_examples_per_dataset = luigi.IntParameter(default=0)

    def openood_loader_kwargs(self) -> dict:
        max_examples = int(self.openood_max_examples_per_dataset)
        return {
            "root_dir": self.openood_root,
            "max_examples_per_dataset": None if max_examples <= 0 else max_examples,
        }


def _protocol_list_suffix(values) -> str:
    return "-".join(str(v) for v in values)


def _pnc_protocol_suffix(task) -> str:
    return (
        f"_modes{_protocol_list_suffix(task.candidate_block_modes)}"
        f"_dirs{_protocol_list_suffix(task.candidate_direction_methods)}"
        f"_k{_protocol_list_suffix(task.candidate_n_directions)}"
        f"_n{_protocol_list_suffix(task.candidate_n_perturbations)}"
        f"_ps{_protocol_list_suffix(task.candidate_perturbation_sizes)}"
        f"_lr{_protocol_list_suffix(task.candidate_lambda_regs)}"
        f"_sel{task.selection_metric}"
        f"_accdrop{_fmt_recipe_float(task.max_validation_accuracy_drop)}"
        f"_vsplit{_fmt_recipe_float(task.val_model_select_split)}"
        f"_bnsplit{_fmt_recipe_float(task.train_bn_refresh_split)}"
    )


def _pnc_protocol_candidate_configs(task) -> list[dict[str, object]]:
    configs = []
    for block_mode, direction_method, n_directions, n_perturbations, perturbation_size, lambda_reg in itertools.product(
        task.candidate_block_modes,
        task.candidate_direction_methods,
        task.candidate_n_directions,
        task.candidate_n_perturbations,
        task.candidate_perturbation_sizes,
        task.candidate_lambda_regs,
    ):
        configs.append(
            {
                "block_mode": str(block_mode),
                "direction_method": str(direction_method),
                "random_directions": str(direction_method) == "random",
                "n_directions": int(n_directions),
                "n_perturbations": int(n_perturbations),
                "perturbation_size": float(perturbation_size),
                "lambda_reg": float(lambda_reg),
            }
        )
    return configs


def _select_best_pnc_candidate(
    candidate_results: list[dict[str, object]],
    *,
    selection_metric: str,
    reference_accuracy: float,
    max_validation_accuracy_drop: float,
) -> dict[str, object]:
    if not candidate_results:
        raise ValueError("candidate_results must be non-empty")

    def _eligible(entry: dict[str, object]) -> bool:
        val_metrics = entry["val_metrics"]
        return float(val_metrics["accuracy"]) >= (reference_accuracy - max_validation_accuracy_drop)

    eligible = [entry for entry in candidate_results if _eligible(entry)]
    constraint_relaxed = False
    pool = eligible
    if not pool:
        pool = candidate_results
        constraint_relaxed = True

    if selection_metric == "nll_within_acc_drop":
        key_fn = lambda entry: (
            float(entry["val_metrics"]["nll"]),
            -float(entry["val_metrics"]["accuracy"]),
            float(entry["val_metrics"]["ece"]),
        )
        criterion_name = "best_validation_nll_subject_to_accuracy_drop"
    elif selection_metric == "nll":
        key_fn = lambda entry: (
            float(entry["val_metrics"]["nll"]),
            -float(entry["val_metrics"]["accuracy"]),
        )
        criterion_name = "best_validation_nll"
    elif selection_metric == "ece":
        key_fn = lambda entry: (
            float(entry["val_metrics"]["ece"]),
            float(entry["val_metrics"]["nll"]),
        )
        criterion_name = "best_validation_ece"
    elif selection_metric == "brier":
        key_fn = lambda entry: (
            float(entry["val_metrics"]["brier"]),
            float(entry["val_metrics"]["nll"]),
        )
        criterion_name = "best_validation_brier"
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")

    selected = min(pool, key=key_fn)
    return {
        "selected_candidate": selected,
        "selection_summary": {
            "selection_metric": selection_metric,
            "selection_criterion": criterion_name,
            "reference_validation_accuracy": float(reference_accuracy),
            "max_validation_accuracy_drop": float(max_validation_accuracy_drop),
            "n_candidates_considered": int(len(candidate_results)),
            "n_candidates_meeting_accuracy_constraint": int(len(eligible)),
            "accuracy_constraint_relaxed": bool(constraint_relaxed),
        },
    }


def _evaluate_cifar_ensemble_with_frozen_temperature(
    ensemble_name: str,
    ensemble,
    inputs: np.ndarray,
    targets: Array,
    *,
    n_classes: int,
    batch_size: int,
    temperature: float,
    sidecar_path: str | None = None,
) -> dict[str, float]:
    t0 = time.time()
    logits = _predict_cifar_logits(ensemble, inputs, batch_size=batch_size)
    jax.block_until_ready(logits)
    eval_time = time.time() - t0
    metrics = _evaluate_cifar_logits(
        ensemble_name,
        logits,
        targets,
        n_classes=n_classes,
        temperature=temperature,
        eval_time=eval_time,
        sidecar_path=sidecar_path,
    )
    metrics["posthoc_temperature"] = metrics.pop("temperature")
    return metrics


def _build_single_block_pnc_ensemble(
    model: PreActResNet18,
    train_fit_inputs: np.ndarray,
    *,
    target_stage_idx: int,
    target_block_idx: int,
    n_directions: int,
    n_perturbations: int,
    perturbation_scale: float,
    subset_size: int,
    chunk_size: int,
    lambda_reg: float,
    random_directions: bool,
    seed: int,
) -> tuple[PnCEnsemble, dict[str, object]]:
    actual_sub = min(len(train_fit_inputs), subset_size)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_fit_inputs), actual_sub, replace=False)
    x_sub = train_fit_inputs[idx]

    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    target_blk = stages[target_stage_idx][target_block_idx]
    w1_orig = target_blk.conv1.kernel.value
    w2_orig = target_blk.conv2.kernel.value

    pre_act_chunks, t_orig_chunks = compute_cifar_block_preacts(
        model,
        x_sub,
        chunk_size,
        target_stage_idx,
        target_block_idx,
        w1_orig,
    )
    get_y_fn = make_cifar_block_get_Y_fn(target_blk)

    if random_directions:
        from pnc import find_random_directions

        v_opts, sigmas = find_random_directions(w1_orig.size, n_directions, seed=seed)
    else:
        from pnc import find_pnc_subspace_lanczos

        v_opts, sigmas = find_pnc_subspace_lanczos(
            get_y_fn,
            w1_orig,
            pre_act_chunks,
            K=n_directions,
            seed=seed,
        )

    rng_z = np.random.RandomState(seed + 17)
    z_coeffs = rng_z.normal(0, 1, size=(n_perturbations, n_directions))
    ens = PnCEnsemble(
        base_model=model,
        v_opts=v_opts,
        sigmas=sigmas,
        z_coeffs=z_coeffs,
        perturbation_scale=perturbation_scale,
        get_Y_fn=get_y_fn,
        w1_orig=w1_orig,
        w2_orig=w2_orig,
        chunks=pre_act_chunks,
        T_orig_chunks=t_orig_chunks,
        target_stage_idx=target_stage_idx,
        target_block_idx=target_block_idx,
        lambda_reg=lambda_reg,
    )
    return ens, {
        "builder": "single_block",
        "subset_size_used": int(actual_sub),
        "target_stage_idx": int(target_stage_idx),
        "target_block_idx": int(target_block_idx),
        "random_directions": bool(random_directions),
    }


def _build_multi_block_pnc_ensemble(
    model: PreActResNet18,
    train_fit_inputs: np.ndarray,
    *,
    n_directions: int,
    n_perturbations: int,
    perturbation_scale: float,
    subset_size: int,
    chunk_size: int,
    lambda_reg: float,
    sigma_sq_weights: bool,
    random_directions: bool,
    seed: int,
) -> tuple[MultiBlockPnCEnsemble, dict[str, object]]:
    actual_sub = min(len(train_fit_inputs), subset_size)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_fit_inputs), actual_sub, replace=False)
    x_sub = train_fit_inputs[idx]

    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    block_indices = preact_resnet18_block_indices()

    @jax.jit
    def run_stem_jit(x):
        return model.stem(x)

    def _stem_chunks(x_data):
        n_chunks = int(np.ceil(len(x_data) / chunk_size))
        return [run_stem_jit(x_data[i * chunk_size:(i + 1) * chunk_size]) for i in range(n_chunks)]

    def _block_forward_chunks(blk, in_chunks, w1, w2):
        block_inputs = []
        block_targets = []
        next_chunks = []
        for h in in_chunks:
            block_inputs.append(h)
            out_bn1 = blk.bn1(h, use_running_average=True)
            out_relu1 = jax.nn.relu(out_bn1)
            y_raw = jax.lax.conv_general_dilated(
                lhs=out_relu1,
                rhs=w1.transpose(3, 2, 0, 1),
                window_strides=tuple(blk.conv1.strides),
                padding="SAME",
                dimension_numbers=("NHWC", "OIHW", "NHWC"),
            )
            y_bn2 = blk.bn2(y_raw, use_running_average=True)
            y_relu2 = jax.nn.relu(y_bn2)
            t_chunk = jax.lax.conv_general_dilated(
                lhs=y_relu2,
                rhs=w2.transpose(3, 2, 0, 1),
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "OIHW", "NHWC"),
            )
            block_targets.append(t_chunk)
            identity = h if blk.downsample is None else blk.downsample(out_relu1)
            next_chunks.append(t_chunk + identity)
        return block_inputs, block_targets, next_chunks

    from pnc import find_pnc_subspace_lanczos, find_random_directions

    h_calib_chunks = _stem_chunks(x_sub)
    block_specs = []
    for block_flat_idx, (stage_idx, block_idx) in enumerate(block_indices):
        target_blk = stages[stage_idx][block_idx]
        w1_orig = target_blk.conv1.kernel.value
        w2_orig = target_blk.conv2.kernel.value
        get_y_fn = make_cifar_block_get_Y_fn(target_blk)

        calib_chunks, calib_targets, h_calib_chunks = _block_forward_chunks(
            target_blk, h_calib_chunks, w1_orig, w2_orig
        )

        if random_directions:
            v_opts, sigmas = find_random_directions(
                w1_orig.size,
                n_directions,
                seed=seed + block_flat_idx,
            )
        else:
            v_opts, sigmas = find_pnc_subspace_lanczos(
                get_y_fn,
                w1_orig,
                calib_chunks,
                K=n_directions,
                seed=seed + block_flat_idx,
            )

        block_specs.append(
            {
                "stage_idx": stage_idx,
                "block_idx": block_idx,
                "w1_orig": w1_orig,
                "w2_orig": w2_orig,
                "v_opts": v_opts,
                "sigmas": sigmas,
                "chunks": calib_chunks,
                "T_orig_chunks": calib_targets,
                "get_Y_fn": get_y_fn,
            }
        )
        jax.clear_caches()

    rng_z = np.random.RandomState(seed + 17)
    z_coeffs = rng_z.normal(
        0,
        1,
        size=(n_perturbations, len(block_indices), n_directions),
    )
    ens = MultiBlockPnCEnsemble(
        base_model=model,
        block_specs=block_specs,
        z_coeffs=z_coeffs,
        perturbation_scale=float(perturbation_scale),
        lambda_reg=lambda_reg,
        sigma_sq_weights=sigma_sq_weights,
    )
    return ens, {
        "builder": "multi_block",
        "subset_size_used": int(actual_sub),
        "n_blocks": int(len(block_indices)),
        "random_directions": bool(random_directions),
        "sigma_sq_weights": bool(sigma_sq_weights),
    }


def _load_cifar_openood_context(task) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    benchmark = load_openood_cifar_benchmark(task.dataset, **task.openood_loader_kwargs())
    x_train_full = benchmark["id_train"]["inputs"]
    y_train_full = benchmark["id_train"]["targets"]
    x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)
    n_cls = 10 if task.dataset.lower() == "cifar10" else 100
    return benchmark, x_tr, y_tr, x_va, y_va, n_cls








class CIFARStandardEnsemble(CIFARRecipeMixin, luigi.Task):
    """Evaluate a deep ensemble of independently trained PreAct ResNet-18 models."""
    dataset      = luigi.Parameter(default='cifar10')
    epochs       = luigi.IntParameter(default=200)
    n_models     = luigi.IntParameter(default=5)
    seed         = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> list[luigi.Task]:
        return [CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, seed=self.seed + i,
            **self.task_recipe_kwargs(),
        ) for i in range(self.n_models)]

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_standard_ensemble{calib_str}_n{self.n_models}_e{self.epochs}'
                f'{recipe}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        _, _, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(f'\n=== CIFAR Standard Ensemble (n={self.n_models}) ===')
        t0 = time.time()
        models = []
        for i, inp in enumerate(self.input()):
            m = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed + i))
            with open(inp.path, 'rb') as f:
                ckpt = pickle.load(f)
            nnx.update(m, ckpt['state'])
            models.append(m)
        print(f'Model load time: {time.time()-t0:.2f}s')

        class _InfBatch:
            def __init__(self, ms): self.ms = ms
            def predict(self, x):
                ys = [m(x, use_running_average=True) for m in self.ms]
                return jnp.stack(ys, axis=0)

        ens = _InfBatch(models)
        train_time = time.time() - t0
        
        # Ensure models are ready for eval (actually evaluation will block, but for train_time it's good to block here)
        for m in models:
            for p in jax.tree_util.tree_leaves(nnx.state(m)):
                if hasattr(p, 'block_until_ready'):
                    p.block_until_ready()

        metrics = _evaluate_cifar('Standard Ensemble', ens, x_te, y_te, n_cls,
                                  calibration_data=(x_va, y_va),
                                  posthoc_calibrate=self.posthoc_calibrate)
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)


class CIFARTrainMCDropoutPreActResNet18(CIFARRecipeMixin, luigi.Task):
    """Train and checkpoint an MC Dropout PreAct ResNet-18."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=200)
    dropout_rate    = luigi.FloatParameter(default=0.1)
    seed            = luigi.IntParameter(default=0)

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'preact_resnet18_mcdropout_train_dr{self.dropout_rate}_e{self.epochs}{recipe}_seed{self.seed}.pkl'))

    def run(self) -> None:
        seed_everything(self.seed)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(f'\n=== CIFAR Train MC Dropout PreAct ResNet-18 ({self.dataset}, dr={self.dropout_rate}, epochs={self.epochs}) ===')
        t0 = time.time()
        model = MCDropoutPreActResNet18(n_classes=n_cls, dropout_rate=self.dropout_rate, rngs=nnx.Rngs(self.seed))
        model = train_resnet_model(model, x_tr, y_tr, x_va, y_va, epochs=self.epochs,
                                   patience=self.epochs, **self.trainer_recipe_kwargs())
        
        # Ensure model is ready
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        print(f'Training time: {time.time()-t0:.2f}s')

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'wb') as f:
            pickle.dump({'state': state, 'train_recipe': self.task_recipe_kwargs()}, f)
        print(f'Checkpoint saved to {self.output().path}')


class CIFARPreActMCDropout(CIFARRecipeMixin, luigi.Task):
    """Evaluate MC Dropout uncertainty from a trained PreAct ResNet-18 checkpoint."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=200)
    n_perturbations = luigi.IntParameter(default=50)
    dropout_rate    = luigi.FloatParameter(default=0.1)
    seed            = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainMCDropoutPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, dropout_rate=self.dropout_rate,
            seed=self.seed, **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_mc_dropout{calib_str}_n{self.n_perturbations}_dr{self.dropout_rate}'
                f'_e{self.epochs}{recipe}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        _, _, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(f'\n=== CIFAR MC Dropout (n={self.n_perturbations}, dr={self.dropout_rate}) ===')
        t0 = time.time()
        model = MCDropoutPreActResNet18(n_classes=n_cls, dropout_rate=self.dropout_rate, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])
        print(f'Model load time: {time.time()-t0:.2f}s')

        class _MCEns:
            def __init__(self, m, n): self.m, self.n = m, n
            def predict(self, x):
                return jnp.stack([self.m(x, use_running_average=True, deterministic=False)
                                   for _ in range(self.n)], axis=0)

        ens = _MCEns(model, self.n_perturbations)
        train_time = time.time() - t0
        
        # Ensure model is ready
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        metrics = _evaluate_cifar('MC Dropout', ens, x_te, y_te, n_cls,
                                  calibration_data=(x_va, y_va),
                                  posthoc_calibrate=self.posthoc_calibrate)
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)


class CIFARTrainSWAGPreActResNet18(CIFARRecipeMixin, luigi.Task):
    """Train a PreAct ResNet-18 checkpoint and collect full low-rank-plus-diagonal SWAG statistics."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=200)
    swag_start_epoch = luigi.IntParameter(default=160)
    swag_collect_freq = luigi.IntParameter(default=1)
    swag_max_rank   = luigi.IntParameter(default=20)
    seed            = luigi.IntParameter(default=0)

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        swag = _cifar_swag_suffix(self, include_bn_refresh=False)
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'preact_resnet18_swag_train_e{self.epochs}{recipe}{swag}_seed{self.seed}.pkl'))

    def run(self) -> None:
        seed_everything(self.seed)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(
            f'\n=== CIFAR Train Full SWAG PreAct ResNet-18 '
            f'({self.dataset}, epochs={self.epochs}, swag_start={self.swag_start_epoch}, '
            f'freq={self.swag_collect_freq}, rank={self.swag_max_rank}) ==='
        )
        t0 = time.time()
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        model, swag_mean, swag_var, swag_cov_mat_sqrt, swag_metadata = train_resnet_swag(
            model,
            x_tr,
            y_tr,
            x_va,
            y_va,
            epochs=self.epochs,
            swag_start_epoch=self.swag_start_epoch,
            swag_collect_freq=self.swag_collect_freq,
            swag_max_rank=self.swag_max_rank,
            **self.trainer_recipe_kwargs(),
        )
        
        # Ensure stats are ready
        jax.block_until_ready(swag_mean)
        jax.block_until_ready(swag_var)
        jax.block_until_ready(swag_cov_mat_sqrt)

        print(f'Training time: {time.time()-t0:.2f}s')

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'wb') as f:
            pickle.dump(
                {
                    'state': state,
                    'swag_mean': swag_mean,
                    'swag_var': swag_var,
                    'swag_cov_mat_sqrt': swag_cov_mat_sqrt,
                    'swag_metadata': swag_metadata,
                    'train_recipe': self.task_recipe_kwargs(),
                },
                f,
            )
        print(f'Checkpoint saved to {self.output().path}')


class CIFARPreActSWAG(CIFARRecipeMixin, luigi.Task):
    """Evaluate full low-rank-plus-diagonal SWAG uncertainty from a trained PreAct ResNet-18 checkpoint."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=200)
    n_perturbations = luigi.IntParameter(default=50)
    swag_start_epoch = luigi.IntParameter(default=160)
    swag_collect_freq = luigi.IntParameter(default=1)
    swag_max_rank   = luigi.IntParameter(default=20)
    swag_use_bn_refresh = luigi.BoolParameter(default=True)
    bn_refresh_subset_size = luigi.IntParameter(default=2048)
    seed            = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainSWAGPreActResNet18(
            dataset=self.dataset,
            epochs=self.epochs,
            swag_start_epoch=self.swag_start_epoch,
            swag_collect_freq=self.swag_collect_freq,
            swag_max_rank=self.swag_max_rank,
            seed=self.seed,
            **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        swag = _cifar_swag_suffix(self, include_bn_refresh=True)
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_swag{calib_str}_n{self.n_perturbations}_e{self.epochs}'
                f'{recipe}{swag}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(
            f'\n=== CIFAR Full SWAG (n={self.n_perturbations}, '
            f'rank={self.swag_max_rank}, '
            f'bn_refresh={self.swag_use_bn_refresh}, bn_subset={self.bn_refresh_subset_size}) ==='
        )
        t0 = time.time()
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))

        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])
        swag_mean = ckpt['swag_mean']
        swag_var = ckpt['swag_var']
        swag_cov_mat_sqrt = ckpt.get('swag_cov_mat_sqrt')
        swag_metadata = ckpt.get('swag_metadata', {})
        print(f'Model load time: {time.time()-t0:.2f}s')

        if self.swag_use_bn_refresh:
            subset_size = min(self.bn_refresh_subset_size, len(x_tr))
            subset_rng = np.random.RandomState(self.seed)
            subset_idx = subset_rng.choice(len(x_tr), size=subset_size, replace=False)
            bn_refresh_inputs = x_tr[subset_idx]
        else:
            bn_refresh_inputs = None

        ens = SWAGEnsemble(
            model,
            swag_mean,
            swag_var,
            self.n_perturbations,
            swag_cov_mat_sqrt=swag_cov_mat_sqrt,
            bn_refresh_inputs=bn_refresh_inputs,
            bn_refresh_batch_size=min(self.batch_size, max(1, self.bn_refresh_subset_size)),
            use_bn_refresh=self.swag_use_bn_refresh,
            seed=self.seed,
        )

        metrics = _evaluate_cifar('SWAG', ens, x_te, y_te, n_cls,
                                  calibration_data=(x_va, y_va),
                                  posthoc_calibrate=self.posthoc_calibrate)
        metrics["train_time"] = time.time() - t0
        metrics["swag_metadata"] = swag_metadata
        metrics["swag_use_bn_refresh"] = bool(self.swag_use_bn_refresh)
        metrics["bn_refresh_subset_size"] = int(0 if bn_refresh_inputs is None else len(bn_refresh_inputs))
        metrics["val_size"] = int(len(x_va))
        
        # Ensure model is ready
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)

class CIFARTrainPreActResNet18(CIFARRecipeMixin, luigi.Task):
    """Train and checkpoint the base PreAct ResNet-18 learner for CIFAR."""
    dataset      = luigi.Parameter(default='cifar10')
    epochs       = luigi.IntParameter(default=200)
    seed         = luigi.IntParameter(default=0)

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'preact_resnet18_train_e{self.epochs}{recipe}_seed{self.seed}.pkl'))

    def run(self) -> None:
        
        seed_everything(self.seed)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.dataset.lower() == 'cifar10':
            x_train_full, y_train_full, x_te, y_te = load_cifar10()
            n_cls = 10
        elif self.dataset.lower() == 'cifar100':
            x_train_full, y_train_full, x_te, y_te = load_cifar100()
            n_cls = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(f'\\n=== CIFAR Train PreAct ResNet-18 ({self.dataset}, epochs={self.epochs}) ===')
        t0 = time.time()
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        model = train_resnet_model(
            model, x_tr, y_tr, x_va, y_va,
            epochs=self.epochs,
            patience=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_name=self.optimizer,
            momentum=self.momentum,
            nesterov=self.nesterov,
            warmup_epochs=min(self.warmup_epochs, max(0, self.epochs - 1)),
            cutout_size=self.cutout_size,
            label_smoothing=self.label_smoothing,
        )
        
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        print(f'Training time: {time.time()-t0:.2f}s')

        class _SingleEns:
            def __init__(self, m): self.m = m
            def predict(self, x):
                return jnp.expand_dims(self.m(x, use_running_average=True), axis=0)

        ens = _SingleEns(model)
        metrics = _evaluate_cifar(
            'PreAct ResNet-18',
            ens,
            x_te,
            y_te,
            n_cls,
            calibration_data=(x_va, y_va),
            posthoc_calibrate=False,
        )

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'wb') as f:
            pickle.dump({'state': state, 'metrics': metrics, 'train_recipe': self.task_recipe_kwargs()}, f)
        print(f'Checkpoint saved to {self.output().path}')


class CIFAREvalPreActResNet18(CIFARRecipeMixin, luigi.Task):
    """Evaluate a trained PreAct ResNet-18 checkpoint, optionally with post-hoc calibration."""
    dataset      = luigi.Parameter(default='cifar10')
    epochs       = luigi.IntParameter(default=200)
    seed         = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset,
            epochs=self.epochs,
            seed=self.seed,
            **self.task_recipe_kwargs(),
        )

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        recipe = self.train_recipe_suffix()
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_preact_resnet18{calib_str}_e{self.epochs}{recipe}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        _, _, x_va, y_va = _split_data(x_train_full, y_train_full)

        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])

        class _SingleEns:
            def __init__(self, m): self.m = m
            def predict(self, x):
                return jnp.expand_dims(self.m(x, use_running_average=True), axis=0)

        ens = _SingleEns(model)
        metrics = _evaluate_cifar(
            'PreAct ResNet-18',
            ens,
            x_te,
            y_te,
            n_cls,
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
        )

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)




class CIFARPnC(CIFARRecipeMixin, luigi.Task):
    """Evaluate single-block perturb-and-correct on a chosen PreAct ResNet-18 block."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=200)
    n_directions       = luigi.IntParameter(default=10)
    n_perturbations    = luigi.IntParameter(default=50)
    perturbation_sizes = luigi.ListParameter(default=[10.0, 50.0, 100.0, 200.0])
    subset_size        = luigi.IntParameter(default=1024)
    chunk_size         = luigi.IntParameter(default=1024)
    target_stage_idx   = luigi.IntParameter(default=3) # stage4 is index 3
    target_block_idx   = luigi.IntParameter(default=1) # block1 is index 1
    random_directions  = luigi.BoolParameter(default=False)
    seed               = luigi.IntParameter(default=0)
    lambda_reg         = luigi.FloatParameter(default=1e-3)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        recipe = self.train_recipe_suffix()
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        suffix = "_random" if self.random_directions else ""
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'pnc_single_block{calib_str}_s{self.target_stage_idx}b{self.target_block_idx}'
                f'_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_lr{self.lambda_reg}'
                f'_e{self.epochs}{recipe}_subsetsize{self.subset_size}_chunksize{self.chunk_size}'
                f'_seed{self.seed}{suffix}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        if self.dataset.lower() == 'cifar10':
            x_train_full, y_train_full, x_te, y_te = load_cifar10()
            n_cls = 10
        elif self.dataset.lower() == 'cifar100':
            x_train_full, y_train_full, x_te, y_te = load_cifar100()
            n_cls = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])

        print(
            f'\\n=== CIFAR Single-block PnC Stage {self.target_stage_idx} '
            f'Block {self.target_block_idx} (K={self.n_directions}, n={self.n_perturbations}) ==='
        )
        t_start = time.time()
        
        actual_sub = min(len(x_tr), self.subset_size)
        rng   = np.random.RandomState(self.seed)
        idx   = rng.choice(len(x_tr), actual_sub, replace=False)
        X_sub = x_tr[idx]
        
        stages = [model.stage1, model.stage2, model.stage3, model.stage4]
        target_blk = stages[self.target_stage_idx][self.target_block_idx]

        w1_orig = target_blk.conv1.kernel.value
        w2_orig = target_blk.conv2.kernel.value

        print('  Precomputing calibration-set block inputs...')
        pre_act_chunks, T_orig_chunks = compute_cifar_block_preacts(
            model, X_sub, self.chunk_size,
            self.target_stage_idx, self.target_block_idx, w1_orig,
        )

        print('  Precomputing test-set block inputs...')
        te_pre_act_chunks, te_T_orig_chunks = compute_cifar_block_preacts(
            model, x_te, self.chunk_size,
            self.target_stage_idx, self.target_block_idx, w1_orig,
        )

        # get_Y_fn: RAW (pre-BN2) conv1 patches; W2 ridge absorbs BN2 + perturbation.
        get_Y_fn = make_cifar_block_get_Y_fn(target_blk)

        t0 = time.time()
        if self.random_directions:
            print(f'  Generating {self.n_directions} random directions...')
            from pnc import find_random_directions
            v_opts, sigmas = find_random_directions(w1_orig.size, self.n_directions, seed=self.seed)
        else:
            print(f'  Finding {self.n_directions} directions via Lanczos over chunks...')
            from pnc import find_pnc_subspace_lanczos
            v_opts, sigmas = find_pnc_subspace_lanczos(
                get_Y_fn, w1_orig, pre_act_chunks,
                K=self.n_directions, seed=self.seed
            )
        print(f'  Direction finding: {time.time()-t0:.2f}s')
        
        setup_time = time.time() - t_start

        rng_z = np.random.RandomState(self.seed)
        all_z = rng_z.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        all_metrics = {}

        for p_size in self.perturbation_sizes:
            print(f'  Building PnC ensemble (p_scale={p_size}) ...')
            t1 = time.time()
            ens = PnCEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, get_Y_fn=get_Y_fn, w1_orig=w1_orig, w2_orig=w2_orig,
                chunks=pre_act_chunks, T_orig_chunks=T_orig_chunks,
                target_stage_idx=self.target_stage_idx, target_block_idx=self.target_block_idx,
                lambda_reg=self.lambda_reg
            )
            print(f'  Precompute time: {time.time()-t1:.2f}s')
            m = _evaluate_cifar(f'PnC (scale={p_size})', ens, x_te, y_te, n_cls,
                                sidecar_path=self.output().path.replace('.json', f'_ps{p_size}.npz'),
                                calibration_data=(x_va, y_va),
                                posthoc_calibrate=self.posthoc_calibrate)
            m["train_time"] = setup_time

            # ── Block-output shift diagnostics (calib & test) ──────────────
            # Calibration arrays already computed during _precompute_corrections.
            raw_calib, corr_calib = ens.calib_raw_arr, ens.calib_corr_arr

            def _diag_stats(raw_arr, corr_arr, prefix):
                r = {
                    f"{prefix}_raw_shift_mean":  float(raw_arr.mean()),
                    f"{prefix}_raw_shift_std":   float(raw_arr.std()),
                    f"{prefix}_corr_shift_mean": float(corr_arr.mean()),
                    f"{prefix}_corr_shift_std":  float(corr_arr.std()),
                    f"{prefix}_reduction_pct":   float(
                        (1.0 - corr_arr.mean() / (raw_arr.mean() + 1e-12)) * 100),
                }
                return r

            # Compute test diagnostics for this perturbation scale
            raw_test, corr_test = ens.compute_shift_diagnostics(
                te_pre_act_chunks, te_T_orig_chunks, label="test")
            m.update(_diag_stats(raw_test, corr_test, "diag_test"))
            
            all_metrics[str(p_size)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(all_metrics, f, indent=2)


class CIFARMultiBlockPnC(CIFARRecipeMixin, luigi.Task):
    """Evaluate multi-block perturb-and-correct across all PreAct ResNet-18 residual blocks."""

    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    n_directions = luigi.IntParameter(default=10)
    n_perturbations = luigi.IntParameter(default=50)
    perturbation_sizes = luigi.ListParameter(default=[10.0, 50.0, 100.0, 200.0])
    subset_size = luigi.IntParameter(default=1024)
    chunk_size = luigi.IntParameter(default=128)
    lambda_reg = luigi.FloatParameter(default=1e-3)
    sigma_sq_weights = luigi.BoolParameter(default=False)
    random_directions = luigi.BoolParameter(default=False)
    seed = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        recipe = self.train_recipe_suffix()
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        suffix = "_random" if self.random_directions else ""
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.dataset
                / (
                    f"pnc_multi_block{calib_str}_k{self.n_directions}_n{self.n_perturbations}"
                    f"_ps{ps}_lr{self.lambda_reg}_e{self.epochs}{recipe}"
                    f"_subsetsize{self.subset_size}_chunksize{self.chunk_size}_seed{self.seed}{suffix}.json"
                )
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        if self.dataset.lower() == "cifar10":
            x_train_full, y_train_full, x_te, y_te = load_cifar10()
            n_cls = 10
        elif self.dataset.lower() == "cifar100":
            x_train_full, y_train_full, x_te, y_te = load_cifar100()
            n_cls = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["state"])

        print(
            f"\n=== CIFAR Multi-block PnC ({len(preact_resnet18_block_indices())} blocks, "
            f"K={self.n_directions}, n={self.n_perturbations}) ==="
        )
        t_start = time.time()

        actual_sub = min(len(x_tr), self.subset_size)
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(x_tr), actual_sub, replace=False)
        X_sub = x_tr[idx]

        stages = [model.stage1, model.stage2, model.stage3, model.stage4]
        block_indices = preact_resnet18_block_indices()
        from pnc import find_pnc_subspace_lanczos, find_random_directions

        @jax.jit
        def run_stem_jit(x):
            return model.stem(x)

        def _stem_chunks(X_data):
            n_chunks = int(np.ceil(len(X_data) / self.chunk_size))
            return [run_stem_jit(X_data[i * self.chunk_size:(i + 1) * self.chunk_size]) for i in range(n_chunks)]

        def _block_forward_chunks(blk, in_chunks, w1, w2):
            block_inputs = []
            block_targets = []
            next_chunks = []
            for h in in_chunks:
                block_inputs.append(h)
                out_bn1 = blk.bn1(h, use_running_average=True)
                out_relu1 = jax.nn.relu(out_bn1)
                y_raw = jax.lax.conv_general_dilated(
                    lhs=out_relu1,
                    rhs=w1.transpose(3, 2, 0, 1),
                    window_strides=tuple(blk.conv1.strides),
                    padding="SAME",
                    dimension_numbers=("NHWC", "OIHW", "NHWC"),
                )
                y_bn2 = blk.bn2(y_raw, use_running_average=True)
                y_relu2 = jax.nn.relu(y_bn2)
                t_chunk = jax.lax.conv_general_dilated(
                    lhs=y_relu2,
                    rhs=w2.transpose(3, 2, 0, 1),
                    window_strides=(1, 1),
                    padding="SAME",
                    dimension_numbers=("NHWC", "OIHW", "NHWC"),
                )
                block_targets.append(t_chunk)
                identity = h
                if blk.downsample is not None:
                    identity = blk.downsample(out_relu1)
                next_chunks.append(t_chunk + identity)
            return block_inputs, block_targets, next_chunks

        h_calib_chunks = _stem_chunks(X_sub)
        h_test_chunks = _stem_chunks(x_te)

        block_specs = []
        test_chunks_by_block = []
        test_targets_by_block = []

        for block_flat_idx, (stage_idx, block_idx) in enumerate(block_indices):
            target_blk = stages[stage_idx][block_idx]
            w1_orig = target_blk.conv1.kernel.value
            w2_orig = target_blk.conv2.kernel.value
            get_Y_fn = make_cifar_block_get_Y_fn(target_blk)

            calib_chunks, calib_targets, h_calib_chunks = _block_forward_chunks(
                target_blk, h_calib_chunks, w1_orig, w2_orig
            )
            test_chunks, test_targets, h_test_chunks = _block_forward_chunks(
                target_blk, h_test_chunks, w1_orig, w2_orig
            )

            print(f"  Finding directions for block {block_flat_idx} (stage={stage_idx}, block={block_idx})...")
            if self.random_directions:
                v_opts, sigmas = find_random_directions(
                    w1_orig.size,
                    self.n_directions,
                    seed=self.seed + block_flat_idx,
                )
            else:
                v_opts, sigmas = find_pnc_subspace_lanczos(
                    get_Y_fn,
                    w1_orig,
                    calib_chunks,
                    K=self.n_directions,
                    seed=self.seed + block_flat_idx,
                )

            block_specs.append(
                {
                    "stage_idx": stage_idx,
                    "block_idx": block_idx,
                    "w1_orig": w1_orig,
                    "w2_orig": w2_orig,
                    "v_opts": v_opts,
                    "sigmas": sigmas,
                    "chunks": calib_chunks,
                    "T_orig_chunks": calib_targets,
                    "get_Y_fn": get_Y_fn,
                }
            )
            test_chunks_by_block.append(test_chunks)
            test_targets_by_block.append(test_targets)
            jax.clear_caches()

        setup_time = time.time() - t_start

        rng_z = np.random.RandomState(self.seed + 999)
        all_z = rng_z.normal(
            0,
            1,
            size=(self.n_perturbations, len(block_indices), self.n_directions),
        )

        all_metrics = {}

        for p_size in self.perturbation_sizes:
            print(f"  Building multi-block PnC ensemble (p_scale={p_size}) ...")
            t1 = time.time()
            ens = MultiBlockPnCEnsemble(
                base_model=model,
                block_specs=block_specs,
                z_coeffs=all_z,
                perturbation_scale=float(p_size),
                lambda_reg=self.lambda_reg,
                sigma_sq_weights=self.sigma_sq_weights,
            )
            print(f"  Precompute time: {time.time() - t1:.2f}s")

            m = _evaluate_cifar(
                f"Multi-block PnC (scale={p_size})",
                ens,
                x_te,
                y_te,
                n_cls,
                sidecar_path=self.output().path.replace(".json", f"_ps{p_size}.npz"),
                calibration_data=(x_va, y_va),
                posthoc_calibrate=self.posthoc_calibrate,
            )
            m["train_time"] = setup_time

            raw_calib, corr_calib = ens.calib_raw_arr, ens.calib_corr_arr
            raw_test, corr_test = ens.compute_shift_diagnostics(
                test_chunks_by_block, test_targets_by_block, label="test"
            )

            def _diag_stats(raw_arr, corr_arr, prefix):
                return {
                    f"{prefix}_raw_shift_mean": float(raw_arr.mean()),
                    f"{prefix}_raw_shift_std": float(raw_arr.std()),
                    f"{prefix}_corr_shift_mean": float(corr_arr.mean()),
                    f"{prefix}_corr_shift_std": float(corr_arr.std()),
                    f"{prefix}_reduction_pct": float(
                        (1.0 - corr_arr.mean() / (raw_arr.mean() + 1e-12)) * 100
                    ),
                }

            m.update(_diag_stats(raw_calib, corr_calib, "diag_calib"))
            m.update(_diag_stats(raw_test, corr_test, "diag_test"))
            all_metrics[str(p_size)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class CIFARSelectPnCPaperProtocol(CIFARRecipeMixin, luigi.Task):
    """Run a paper-clean CIFAR PnC protocol with validation-only config selection."""

    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    candidate_block_modes = luigi.ListParameter(default=["single", "multi"])
    candidate_direction_methods = luigi.ListParameter(default=["lanczos", "random"])
    candidate_n_directions = luigi.ListParameter(default=[10, 20])
    candidate_n_perturbations = luigi.ListParameter(default=[30])
    candidate_perturbation_sizes = luigi.ListParameter(default=[10.0, 50.0, 100.0])
    candidate_lambda_regs = luigi.ListParameter(default=[1e-3, 1e-2])
    subset_size = luigi.IntParameter(default=1024)
    chunk_size = luigi.IntParameter(default=128)
    target_stage_idx = luigi.IntParameter(default=3)
    target_block_idx = luigi.IntParameter(default=1)
    sigma_sq_weights = luigi.BoolParameter(default=False)
    selection_metric = luigi.Parameter(default="nll_within_acc_drop")
    max_validation_accuracy_drop = luigi.FloatParameter(default=0.02)
    val_model_select_split = luigi.FloatParameter(default=0.1)
    train_bn_refresh_split = luigi.FloatParameter(default=0.1)
    posthoc_calibrate = luigi.BoolParameter(default=True)
    seed = luigi.IntParameter(default=0)

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        protocol = _pnc_protocol_suffix(self)
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.dataset
                / (
                    f"pnc_protocol_select{calib_str}_e{self.epochs}{recipe}{protocol}"
                    f"_subsetsize{self.subset_size}_chunksize{self.chunk_size}"
                    f"_s{self.target_stage_idx}b{self.target_block_idx}"
                    f"_seed{self.seed}.json"
                )
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        protocol_splits = _make_cifar_protocol_splits(
            x_train_full,
            y_train_full,
            x_te,
            y_te,
            val_model_select_split=self.val_model_select_split,
            train_bn_refresh_split=self.train_bn_refresh_split,
            seed=self.seed,
        )
        x_fit, y_fit = protocol_splits["train_fit"]
        x_bn, y_bn = protocol_splits["train_bn_refresh"]
        x_val, y_val = protocol_splits["val_model_select"]
        x_test, y_test = protocol_splits["test_report"]

        print(
            f"\n=== CIFAR PnC Paper Protocol ({self.dataset}) ===\n"
            f"  train_fit={len(x_fit)} | train_bn_refresh={len(x_bn)} | "
            f"val_model_select={len(x_val)} | test_report={len(x_test)}"
        )

        t_train = time.time()
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        model = train_resnet_model(
            model,
            x_fit,
            y_fit,
            x_val,
            y_val,
            epochs=self.epochs,
            patience=self.epochs,
            **self.trainer_recipe_kwargs(),
        )
        base_train_time = time.time() - t_train

        class _SingleEns:
            def __init__(self, m):
                self.m = m

            def predict(self, x):
                return jnp.expand_dims(self.m(x, use_running_average=True), axis=0)

        baseline_ens = _SingleEns(model)
        baseline_val_logits = _predict_cifar_logits(baseline_ens, x_val, batch_size=self.batch_size)
        jax.block_until_ready(baseline_val_logits)
        baseline_temperature = 1.0
        if self.posthoc_calibrate:
            baseline_temperature = _fit_posthoc_temperature(baseline_val_logits, y_val)
        baseline_val_metrics = _evaluate_cifar_logits(
            "PreAct ResNet-18 validation baseline",
            baseline_val_logits,
            y_val,
            n_classes=n_cls,
            temperature=baseline_temperature,
        )
        baseline_val_metrics["posthoc_temperature"] = baseline_val_metrics.pop("temperature")

        candidate_results = []
        candidate_configs = _pnc_protocol_candidate_configs(self)
        for candidate_idx, config in enumerate(candidate_configs):
            candidate_seed = self.seed + 1000 + candidate_idx
            print(
                f"\n[selection] Candidate {candidate_idx + 1}/{len(candidate_configs)} "
                f"{config['block_mode']} {config['direction_method']} "
                f"K={config['n_directions']} n={config['n_perturbations']} "
                f"ps={config['perturbation_size']} lambda={config['lambda_reg']}"
            )
            build_start = time.time()
            builder_model = nnx.clone(model)
            if config["block_mode"] == "single":
                ens, builder_metadata = _build_single_block_pnc_ensemble(
                    builder_model,
                    x_fit,
                    target_stage_idx=self.target_stage_idx,
                    target_block_idx=self.target_block_idx,
                    n_directions=config["n_directions"],
                    n_perturbations=config["n_perturbations"],
                    perturbation_scale=config["perturbation_size"],
                    subset_size=self.subset_size,
                    chunk_size=self.chunk_size,
                    lambda_reg=config["lambda_reg"],
                    random_directions=config["random_directions"],
                    seed=candidate_seed,
                )
            elif config["block_mode"] == "multi":
                ens, builder_metadata = _build_multi_block_pnc_ensemble(
                    builder_model,
                    x_fit,
                    n_directions=config["n_directions"],
                    n_perturbations=config["n_perturbations"],
                    perturbation_scale=config["perturbation_size"],
                    subset_size=self.subset_size,
                    chunk_size=self.chunk_size,
                    lambda_reg=config["lambda_reg"],
                    sigma_sq_weights=self.sigma_sq_weights,
                    random_directions=config["random_directions"],
                    seed=candidate_seed,
                )
            else:
                raise ValueError(f"Unknown block_mode: {config['block_mode']}")

            val_start = time.time()
            val_logits = _predict_cifar_logits(ens, x_val, batch_size=self.batch_size)
            jax.block_until_ready(val_logits)
            val_eval_time = time.time() - val_start
            candidate_temperature = 1.0
            if self.posthoc_calibrate:
                candidate_temperature = _fit_posthoc_temperature(val_logits, y_val)
            val_metrics = _evaluate_cifar_logits(
                f"PnC validation candidate {candidate_idx + 1}",
                val_logits,
                y_val,
                n_classes=n_cls,
                temperature=candidate_temperature,
                eval_time=val_eval_time,
            )
            val_metrics["posthoc_temperature"] = val_metrics.pop("temperature")
            candidate_results.append(
                {
                    "candidate_id": int(candidate_idx),
                    "candidate_seed": int(candidate_seed),
                    "config": dict(config),
                    "builder_metadata": builder_metadata,
                    "build_time": float(time.time() - build_start),
                    "val_metrics": val_metrics,
                }
            )

        selection = _select_best_pnc_candidate(
            candidate_results,
            selection_metric=self.selection_metric,
            reference_accuracy=float(baseline_val_metrics["accuracy"]),
            max_validation_accuracy_drop=self.max_validation_accuracy_drop,
        )
        selected_candidate = selection["selected_candidate"]
        selected_config = dict(selected_candidate["config"])
        selected_seed = int(selected_candidate["candidate_seed"])
        selected_temperature = float(selected_candidate["val_metrics"]["posthoc_temperature"])

        print(
            f"\n[selection] Selected candidate {selected_candidate['candidate_id']} "
            f"with validation {selection['selection_summary']['selection_criterion']}"
        )

        selected_model = nnx.clone(model)
        if selected_config["block_mode"] == "single":
            selected_ens, selected_builder_metadata = _build_single_block_pnc_ensemble(
                selected_model,
                x_fit,
                target_stage_idx=self.target_stage_idx,
                target_block_idx=self.target_block_idx,
                n_directions=selected_config["n_directions"],
                n_perturbations=selected_config["n_perturbations"],
                perturbation_scale=selected_config["perturbation_size"],
                subset_size=self.subset_size,
                chunk_size=self.chunk_size,
                lambda_reg=selected_config["lambda_reg"],
                random_directions=selected_config["random_directions"],
                seed=selected_seed,
            )
        else:
            selected_ens, selected_builder_metadata = _build_multi_block_pnc_ensemble(
                selected_model,
                x_fit,
                n_directions=selected_config["n_directions"],
                n_perturbations=selected_config["n_perturbations"],
                perturbation_scale=selected_config["perturbation_size"],
                subset_size=self.subset_size,
                chunk_size=self.chunk_size,
                lambda_reg=selected_config["lambda_reg"],
                sigma_sq_weights=self.sigma_sq_weights,
                random_directions=selected_config["random_directions"],
                seed=selected_seed,
            )

        test_metrics = _evaluate_cifar_ensemble_with_frozen_temperature(
            "Selected PnC test report",
            selected_ens,
            x_test,
            y_test,
            n_classes=n_cls,
            batch_size=self.batch_size,
            temperature=selected_temperature,
            sidecar_path=self.output().path.replace(".json", "_selected_test.npz"),
        )

        results = {
            "protocol_metadata": {
                **protocol_splits["metadata"],
                "dataset": self.dataset,
                "roles": {
                    "train_fit": "fit base model and PnC corrections",
                    "train_bn_refresh": "reserved training-only split; unused for PnC in this protocol",
                    "val_model_select": "hyperparameter selection and post-hoc temperature scaling",
                    "test_report": "final reporting only after config is frozen",
                },
                "selection_policy": {
                    "pnc_hyperparameter_selection": "id_validation_only",
                    "openood_usage": "evaluation_only",
                    "ood_validation_permitted": False,
                    "ood_threshold_tuning_permitted": False,
                },
                "posthoc_calibrate": bool(self.posthoc_calibrate),
                "selection_metric": self.selection_metric,
            },
            "train_recipe": self.task_recipe_kwargs(),
            "base_train_time": float(base_train_time),
            "base_validation_metrics": baseline_val_metrics,
            "candidate_results": candidate_results,
            "selection": selection["selection_summary"],
            "selected_config": selected_config,
            "selected_candidate_id": int(selected_candidate["candidate_id"]),
            "selected_validation_metrics": selected_candidate["val_metrics"],
            "selected_builder_metadata": selected_builder_metadata,
            "selected_test_metrics": test_metrics,
            "bn_refresh_split_size": int(len(x_bn)),
            "bn_refresh_targets_size": int(len(y_bn)),
        }

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(results, f, indent=2)


class CIFARLLLA(CIFARRecipeMixin, luigi.Task):
    """Evaluate last-layer Laplace uncertainty for a trained PreAct ResNet-18."""
    dataset          = luigi.Parameter(default='cifar10')
    epochs           = luigi.IntParameter(default=200)
    n_perturbations  = luigi.IntParameter(default=50)
    prior_precision  = luigi.FloatParameter(default=1.0)
    seed             = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_llla{calib_str}_n{self.n_perturbations}_prec{self.prior_precision}'
                f'_e{self.epochs}{recipe}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(f'\n=== CIFAR LLLA (n={self.n_perturbations}, prior_prec={self.prior_precision}) ===')
        t0 = time.time()

        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])

        @jax.jit
        def get_features(x):
            h = model.stem(x)
            h = model._run_stages(h, use_running_average=True)
            h = model.final_bn(h, use_running_average=True)
            h = jax.nn.relu(h)
            h = jnp.mean(h, axis=(1, 2))
            return h

        @jax.jit
        def compute_batch_ggn(x_hat, probs):
            H = jax.vmap(lambda p: jnp.diag(p) - jnp.outer(p, p))(probs)
            return jnp.einsum('si,sj,skm->ikjm', x_hat, x_hat, H)

        D = 512
        K = n_cls
        G = jnp.zeros((D + 1, K, D + 1, K))

        print("  Computing GGN over training set...")
        for i in range(0, len(x_tr), self.batch_size):
            x_batch = x_tr[i:i+self.batch_size]
            feats = get_features(x_batch)
            logits = model.fc(feats)
            probs = jax.nn.softmax(logits)
            X_hat = jnp.concatenate([feats, jnp.ones((feats.shape[0], 1))], axis=-1)
            G += compute_batch_ggn(X_hat, probs)

        G_flat = G.reshape((D + 1) * K, (D + 1) * K)

        print("  Inverting GGN...")
        precision = G_flat + self.prior_precision * jnp.eye(G_flat.shape[0])
        covariance = jnp.linalg.inv(precision)

        fc_state = nnx.state(model.fc)
        ens = LLLAEnsemble(model, fc_state, covariance, self.n_perturbations, self.seed)

        train_time = time.time() - t0
        print("  Evaluating LLLA Ensemble...")
        metrics = _evaluate_cifar('LLLA', ens, x_te, y_te, n_cls,
                                  calibration_data=(x_va, y_va),
                                  posthoc_calibrate=self.posthoc_calibrate)
        metrics["train_time"] = train_time

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)


class AllCIFARExperiments(CIFARRecipeMixin, luigi.WrapperTask):
    """Current CIFAR workflow: train the base learner, run baselines, then run single- and multi-block PnC."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=200)
    n_perturbations    = luigi.IntParameter(default=100)
    n_models           = luigi.IntParameter(default=5)
    n_directions       = luigi.IntParameter(default=40)
    perturbation_sizes = luigi.ListParameter(default=[0.01, 0.05, 0.1, 0.5])
    seed               = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)
    random_directions = luigi.BoolParameter(default=False)

    def requires(self) -> list[luigi.Task]:
        shared = dict(
            dataset=self.dataset,
            epochs=self.epochs,
            seed=self.seed,
            posthoc_calibrate=self.posthoc_calibrate,
            **self.task_recipe_kwargs(),
        )
        return [
            CIFAREvalPreActResNet18(**shared),
            CIFARStandardEnsemble(n_models=self.n_models, **shared),
            CIFARPreActMCDropout(n_perturbations=self.n_perturbations, **shared),
            CIFARPreActSWAG(n_perturbations=self.n_perturbations, **shared),
            CIFARPnC(
                n_directions=self.n_directions,
                n_perturbations=self.n_perturbations,
                perturbation_sizes=self.perturbation_sizes,
                random_directions=self.random_directions,
                **shared,
            ),
            CIFARMultiBlockPnC(
                n_directions=self.n_directions,
                n_perturbations=self.n_perturbations,
                perturbation_sizes=self.perturbation_sizes,
                random_directions=self.random_directions,
                **shared,
            ),
            CIFARLLLA(n_perturbations=self.n_perturbations, **shared),
        ]


class CIFAROpenOODPreActResNet18(CIFARRecipeMixin, CIFAROpenOODMixin, luigi.Task):
    """Evaluate the base PreAct ResNet-18 under the OpenOOD v1.5 CIFAR protocol."""
    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    seed = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset,
            epochs=self.epochs,
            seed=self.seed,
            **self.task_recipe_kwargs(),
        )

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        recipe = self.train_recipe_suffix()
        max_str = "" if self.openood_max_examples_per_dataset <= 0 else f"_oomax{self.openood_max_examples_per_dataset}"
        return luigi.LocalTarget(
            str(Path("results") / self.dataset / f"openood_v1p5_preact_resnet18{calib_str}_e{self.epochs}{recipe}_seed{self.seed}{max_str}.json")
        )

    def run(self) -> None:
        seed_everything(self.seed)
        benchmark, _, _, x_va, y_va, n_cls = _load_cifar_openood_context(self)
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["state"])

        class _SingleEns:
            def __init__(self, m):
                self.m = m

            def predict(self, x):
                return jnp.expand_dims(self.m(x, use_running_average=True), axis=0)

        metrics = evaluate_openood_cifar(
            "PreAct ResNet-18",
            _SingleEns(model),
            benchmark,
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
            batch_size=self.batch_size,
        )
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class CIFAROpenOODStandardEnsemble(CIFARRecipeMixin, CIFAROpenOODMixin, luigi.Task):
    """Evaluate the standard CIFAR deep ensemble under OpenOOD v1.5."""
    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    n_models = luigi.IntParameter(default=5)
    seed = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> list[luigi.Task]:
        return [
            CIFARTrainPreActResNet18(
                dataset=self.dataset,
                epochs=self.epochs,
                seed=self.seed + i,
                **self.task_recipe_kwargs(),
            )
            for i in range(self.n_models)
        ]

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        recipe = self.train_recipe_suffix()
        max_str = "" if self.openood_max_examples_per_dataset <= 0 else f"_oomax{self.openood_max_examples_per_dataset}"
        return luigi.LocalTarget(
            str(Path("results") / self.dataset / f"openood_v1p5_standard_ensemble{calib_str}_n{self.n_models}_e{self.epochs}{recipe}_seed{self.seed}{max_str}.json")
        )

    def run(self) -> None:
        seed_everything(self.seed)
        benchmark, _, _, x_va, y_va, n_cls = _load_cifar_openood_context(self)
        models = []
        for i, inp in enumerate(self.input()):
            model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed + i))
            with open(inp.path, "rb") as f:
                ckpt = pickle.load(f)
            nnx.update(model, ckpt["state"])
            models.append(model)

        class _InfBatch:
            def __init__(self, ms):
                self.ms = ms

            def predict(self, x):
                return jnp.stack([m(x, use_running_average=True) for m in self.ms], axis=0)

        metrics = evaluate_openood_cifar(
            "Standard Ensemble",
            _InfBatch(models),
            benchmark,
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
            batch_size=self.batch_size,
        )
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class CIFAROpenOODMCDropout(CIFARRecipeMixin, CIFAROpenOODMixin, luigi.Task):
    """Evaluate MC Dropout under the OpenOOD v1.5 CIFAR protocol."""
    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    n_perturbations = luigi.IntParameter(default=50)
    dropout_rate = luigi.FloatParameter(default=0.1)
    seed = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainMCDropoutPreActResNet18(
            dataset=self.dataset,
            epochs=self.epochs,
            dropout_rate=self.dropout_rate,
            seed=self.seed,
            **self.task_recipe_kwargs(),
        )

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        recipe = self.train_recipe_suffix()
        max_str = "" if self.openood_max_examples_per_dataset <= 0 else f"_oomax{self.openood_max_examples_per_dataset}"
        return luigi.LocalTarget(
            str(Path("results") / self.dataset / f"openood_v1p5_mc_dropout{calib_str}_n{self.n_perturbations}_dr{self.dropout_rate}_e{self.epochs}{recipe}_seed{self.seed}{max_str}.json")
        )

    def run(self) -> None:
        seed_everything(self.seed)
        benchmark, _, _, x_va, y_va, n_cls = _load_cifar_openood_context(self)
        model = MCDropoutPreActResNet18(n_classes=n_cls, dropout_rate=self.dropout_rate, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["state"])

        class _MCEns:
            def __init__(self, m, n):
                self.m = m
                self.n = n

            def predict(self, x):
                return jnp.stack(
                    [self.m(x, use_running_average=True, deterministic=False) for _ in range(self.n)],
                    axis=0,
                )

        metrics = evaluate_openood_cifar(
            "MC Dropout",
            _MCEns(model, self.n_perturbations),
            benchmark,
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
            batch_size=self.batch_size,
        )
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class CIFAROpenOODSWAG(CIFARRecipeMixin, CIFAROpenOODMixin, luigi.Task):
    """Evaluate full SWAG under the OpenOOD v1.5 CIFAR protocol."""
    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    n_perturbations = luigi.IntParameter(default=50)
    swag_start_epoch = luigi.IntParameter(default=160)
    swag_collect_freq = luigi.IntParameter(default=1)
    swag_max_rank = luigi.IntParameter(default=20)
    swag_use_bn_refresh = luigi.BoolParameter(default=True)
    bn_refresh_subset_size = luigi.IntParameter(default=2048)
    seed = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainSWAGPreActResNet18(
            dataset=self.dataset,
            epochs=self.epochs,
            swag_start_epoch=self.swag_start_epoch,
            swag_collect_freq=self.swag_collect_freq,
            swag_max_rank=self.swag_max_rank,
            seed=self.seed,
            **self.task_recipe_kwargs(),
        )

    def output(self) -> luigi.LocalTarget:
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        recipe = self.train_recipe_suffix()
        swag = _cifar_swag_suffix(self, include_bn_refresh=True)
        max_str = "" if self.openood_max_examples_per_dataset <= 0 else f"_oomax{self.openood_max_examples_per_dataset}"
        return luigi.LocalTarget(
            str(Path("results") / self.dataset / f"openood_v1p5_swag{calib_str}_n{self.n_perturbations}_e{self.epochs}{recipe}{swag}_seed{self.seed}{max_str}.json")
        )

    def run(self) -> None:
        seed_everything(self.seed)
        benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(self)
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["state"])
        subset_size = min(self.bn_refresh_subset_size, len(x_tr))
        bn_refresh_inputs = None
        if self.swag_use_bn_refresh and subset_size > 0:
            subset_rng = np.random.RandomState(self.seed)
            subset_idx = subset_rng.choice(len(x_tr), size=subset_size, replace=False)
            bn_refresh_inputs = x_tr[subset_idx]
        ens = SWAGEnsemble(
            model,
            ckpt["swag_mean"],
            ckpt["swag_var"],
            self.n_perturbations,
            swag_cov_mat_sqrt=ckpt.get("swag_cov_mat_sqrt"),
            bn_refresh_inputs=bn_refresh_inputs,
            bn_refresh_batch_size=min(self.batch_size, max(1, self.bn_refresh_subset_size)),
            use_bn_refresh=self.swag_use_bn_refresh,
            seed=self.seed,
        )
        metrics = evaluate_openood_cifar(
            "SWAG",
            ens,
            benchmark,
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
            batch_size=self.batch_size,
        )
        metrics["swag_metadata"] = ckpt.get("swag_metadata", {})
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class CIFAROpenOODPnC(CIFARRecipeMixin, CIFAROpenOODMixin, luigi.Task):
    """Evaluate single-block PnC under the OpenOOD v1.5 CIFAR protocol."""
    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    n_directions = luigi.IntParameter(default=10)
    n_perturbations = luigi.IntParameter(default=50)
    perturbation_sizes = luigi.ListParameter(default=[10.0, 50.0, 100.0, 200.0])
    subset_size = luigi.IntParameter(default=1024)
    chunk_size = luigi.IntParameter(default=1024)
    target_stage_idx = luigi.IntParameter(default=3)
    target_block_idx = luigi.IntParameter(default=1)
    random_directions = luigi.BoolParameter(default=False)
    seed = luigi.IntParameter(default=0)
    lambda_reg = luigi.FloatParameter(default=1e-3)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs())

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        recipe = self.train_recipe_suffix()
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        suffix = "_random" if self.random_directions else ""
        max_str = "" if self.openood_max_examples_per_dataset <= 0 else f"_oomax{self.openood_max_examples_per_dataset}"
        return luigi.LocalTarget(
            str(Path("results") / self.dataset / f"openood_v1p5_pnc_single_block{calib_str}_s{self.target_stage_idx}b{self.target_block_idx}_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_lr{self.lambda_reg}_e{self.epochs}{recipe}_subsetsize{self.subset_size}_chunksize{self.chunk_size}_seed{self.seed}{suffix}{max_str}.json")
        )

    def run(self) -> None:
        seed_everything(self.seed)
        benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(self)
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["state"])

        all_metrics = {}
        for p_size in self.perturbation_sizes:
            ens, build_meta = _build_single_block_pnc_ensemble(
                nnx.clone(model),
                x_tr,
                target_stage_idx=self.target_stage_idx,
                target_block_idx=self.target_block_idx,
                n_directions=self.n_directions,
                n_perturbations=self.n_perturbations,
                perturbation_scale=float(p_size),
                subset_size=self.subset_size,
                chunk_size=self.chunk_size,
                lambda_reg=self.lambda_reg,
                random_directions=self.random_directions,
                seed=self.seed,
            )
            metrics = evaluate_openood_cifar(
                f"PnC scale={p_size}",
                ens,
                benchmark,
                calibration_data=(x_va, y_va),
                posthoc_calibrate=self.posthoc_calibrate,
                batch_size=self.batch_size,
            )
            metrics["builder_metadata"] = build_meta
            all_metrics[str(p_size)] = metrics

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class CIFAROpenOODMultiBlockPnC(CIFARRecipeMixin, CIFAROpenOODMixin, luigi.Task):
    """Evaluate multi-block PnC under the OpenOOD v1.5 CIFAR protocol."""
    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    n_directions = luigi.IntParameter(default=10)
    n_perturbations = luigi.IntParameter(default=50)
    perturbation_sizes = luigi.ListParameter(default=[10.0, 50.0, 100.0, 200.0])
    subset_size = luigi.IntParameter(default=1024)
    chunk_size = luigi.IntParameter(default=128)
    lambda_reg = luigi.FloatParameter(default=1e-3)
    sigma_sq_weights = luigi.BoolParameter(default=False)
    random_directions = luigi.BoolParameter(default=False)
    seed = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs())

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        recipe = self.train_recipe_suffix()
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        suffix = "_random" if self.random_directions else ""
        max_str = "" if self.openood_max_examples_per_dataset <= 0 else f"_oomax{self.openood_max_examples_per_dataset}"
        return luigi.LocalTarget(
            str(Path("results") / self.dataset / f"openood_v1p5_pnc_multi_block{calib_str}_k{self.n_directions}_n{self.n_perturbations}_ps{ps}_lr{self.lambda_reg}_e{self.epochs}{recipe}_subsetsize{self.subset_size}_chunksize{self.chunk_size}_seed{self.seed}{suffix}{max_str}.json")
        )

    def run(self) -> None:
        seed_everything(self.seed)
        benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(self)
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["state"])

        all_metrics = {}
        for p_size in self.perturbation_sizes:
            ens, build_meta = _build_multi_block_pnc_ensemble(
                nnx.clone(model),
                x_tr,
                n_directions=self.n_directions,
                n_perturbations=self.n_perturbations,
                perturbation_scale=float(p_size),
                subset_size=self.subset_size,
                chunk_size=self.chunk_size,
                lambda_reg=self.lambda_reg,
                sigma_sq_weights=self.sigma_sq_weights,
                random_directions=self.random_directions,
                seed=self.seed,
            )
            metrics = evaluate_openood_cifar(
                f"Multi-block PnC scale={p_size}",
                ens,
                benchmark,
                calibration_data=(x_va, y_va),
                posthoc_calibrate=self.posthoc_calibrate,
                batch_size=self.batch_size,
            )
            metrics["builder_metadata"] = build_meta
            all_metrics[str(p_size)] = metrics

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)


class CIFAROpenOODLLLA(CIFARRecipeMixin, CIFAROpenOODMixin, luigi.Task):
    """Evaluate LLLA under the OpenOOD v1.5 CIFAR protocol."""
    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    n_perturbations = luigi.IntParameter(default=50)
    prior_precision = luigi.FloatParameter(default=1.0)
    seed = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs())

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        calib_str = _posthoc_suffix(self.posthoc_calibrate)
        max_str = "" if self.openood_max_examples_per_dataset <= 0 else f"_oomax{self.openood_max_examples_per_dataset}"
        return luigi.LocalTarget(
            str(Path("results") / self.dataset / f"openood_v1p5_llla{calib_str}_n{self.n_perturbations}_prec{self.prior_precision}_e{self.epochs}{recipe}_seed{self.seed}{max_str}.json")
        )

    def run(self) -> None:
        seed_everything(self.seed)
        benchmark, x_tr, _, x_va, y_va, n_cls = _load_cifar_openood_context(self)
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["state"])

        @jax.jit
        def get_features(x):
            h = model.stem(x)
            h = model._run_stages(h, use_running_average=True)
            h = model.final_bn(h, use_running_average=True)
            h = jax.nn.relu(h)
            h = jnp.mean(h, axis=(1, 2))
            return h

        @jax.jit
        def compute_batch_ggn(x_hat, probs):
            H = jax.vmap(lambda p: jnp.diag(p) - jnp.outer(p, p))(probs)
            return jnp.einsum("si,sj,skm->ikjm", x_hat, x_hat, H)

        D = 512
        K = n_cls
        G = jnp.zeros((D + 1, K, D + 1, K))
        for i in range(0, len(x_tr), self.batch_size):
            x_batch = x_tr[i:i + self.batch_size]
            feats = get_features(x_batch)
            logits = model.fc(feats)
            probs = jax.nn.softmax(logits)
            x_hat = jnp.concatenate([feats, jnp.ones((feats.shape[0], 1))], axis=-1)
            G += compute_batch_ggn(x_hat, probs)
        G_flat = G.reshape((D + 1) * K, (D + 1) * K)
        precision = G_flat + self.prior_precision * jnp.eye(G_flat.shape[0])
        covariance = jnp.linalg.inv(precision)
        fc_state = nnx.state(model.fc)
        ens = LLLAEnsemble(model, fc_state, covariance, self.n_perturbations, self.seed)

        metrics = evaluate_openood_cifar(
            "LLLA",
            ens,
            benchmark,
            calibration_data=(x_va, y_va),
            posthoc_calibrate=self.posthoc_calibrate,
            batch_size=self.batch_size,
        )
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(metrics, f, indent=2)


class AllCIFAROpenOODExperiments(CIFARRecipeMixin, CIFAROpenOODMixin, luigi.WrapperTask):
    """Evaluate the main CIFAR baselines and PnC variants on OpenOOD v1.5."""
    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=200)
    n_perturbations = luigi.IntParameter(default=100)
    n_models = luigi.IntParameter(default=5)
    n_directions = luigi.IntParameter(default=40)
    perturbation_sizes = luigi.ListParameter(default=[0.01, 0.05, 0.1, 0.5])
    seed = luigi.IntParameter(default=0)
    posthoc_calibrate = luigi.BoolParameter(default=False)
    random_directions = luigi.BoolParameter(default=False)

    def requires(self) -> list[luigi.Task]:
        shared = dict(
            dataset=self.dataset,
            epochs=self.epochs,
            seed=self.seed,
            posthoc_calibrate=self.posthoc_calibrate,
            openood_root=self.openood_root,
            openood_max_examples_per_dataset=self.openood_max_examples_per_dataset,
            **self.task_recipe_kwargs(),
        )
        return [
            CIFAROpenOODPreActResNet18(**shared),
            CIFAROpenOODStandardEnsemble(n_models=self.n_models, **shared),
            CIFAROpenOODMCDropout(n_perturbations=self.n_perturbations, **shared),
            CIFAROpenOODSWAG(n_perturbations=self.n_perturbations, **shared),
            CIFAROpenOODPnC(
                n_directions=self.n_directions,
                n_perturbations=self.n_perturbations,
                perturbation_sizes=self.perturbation_sizes,
                random_directions=self.random_directions,
                **shared,
            ),
            CIFAROpenOODMultiBlockPnC(
                n_directions=self.n_directions,
                n_perturbations=self.n_perturbations,
                perturbation_sizes=self.perturbation_sizes,
                random_directions=self.random_directions,
                **shared,
            ),
            CIFAROpenOODLLLA(n_perturbations=self.n_perturbations, **shared),
        ]
