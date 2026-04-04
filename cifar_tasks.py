"""Luigi tasks for CIFAR-10/100 baselines and PnC experiments on PreAct ResNet-18."""
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

from util import seed_everything, _evaluate_cifar, _ps_str, _split_data
from models import PreActResNet18, MCDropoutPreActResNet18
from data import load_cifar10, load_cifar100
from training import train_resnet_model, train_resnet_swag
from ensembles import (
    SWAGEnsemble,
    PnCEnsemble,
    MultiBlockPnCEnsemble,
    LLLAEnsemble,
)


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
        return extract_patches(y_relu2, k=3, strides=1)

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
    suffix = f"_sws{task.swag_start_epoch}_swf{task.swag_collect_freq}"
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








class CIFARStandardEnsemble(CIFARRecipeMixin, luigi.Task):
    """Evaluate a deep ensemble of independently trained PreAct ResNet-18 models."""
    dataset      = luigi.Parameter(default='cifar10')
    epochs       = luigi.IntParameter(default=200)
    n_models     = luigi.IntParameter(default=5)
    seed         = luigi.IntParameter(default=0)

    def requires(self) -> list[luigi.Task]:
        return [CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, seed=self.seed + i,
            **self.task_recipe_kwargs(),
        ) for i in range(self.n_models)]

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_standard_ensemble_n{self.n_models}_e{self.epochs}{recipe}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        _, _, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

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

        metrics = _evaluate_cifar('Standard Ensemble', ens, x_te, y_te, n_cls)
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

    def requires(self) -> luigi.Task:
        return CIFARTrainMCDropoutPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, dropout_rate=self.dropout_rate,
            seed=self.seed, **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_mc_dropout_n{self.n_perturbations}_dr{self.dropout_rate}'
                f'_e{self.epochs}{recipe}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        _, _, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

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

        metrics = _evaluate_cifar('MC Dropout', ens, x_te, y_te, n_cls)
        metrics["train_time"] = train_time
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)


class CIFARTrainSWAGPreActResNet18(CIFARRecipeMixin, luigi.Task):
    """Train a PreAct ResNet-18 checkpoint and collect diagonal-SWAG statistics."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=200)
    swag_start_epoch = luigi.IntParameter(default=160)
    swag_collect_freq = luigi.IntParameter(default=1)
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
            f'\n=== CIFAR Train Diagonal SWAG PreAct ResNet-18 '
            f'({self.dataset}, epochs={self.epochs}, swag_start={self.swag_start_epoch}, '
            f'freq={self.swag_collect_freq}) ==='
        )
        t0 = time.time()
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        model, swag_mean, swag_var, swag_metadata = train_resnet_swag(
            model,
            x_tr,
            y_tr,
            x_va,
            y_va,
            epochs=self.epochs,
            swag_start_epoch=self.swag_start_epoch,
            swag_collect_freq=self.swag_collect_freq,
            **self.trainer_recipe_kwargs(),
        )
        
        # Ensure stats are ready
        jax.block_until_ready(swag_mean)
        jax.block_until_ready(swag_var)

        print(f'Training time: {time.time()-t0:.2f}s')

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'wb') as f:
            pickle.dump(
                {
                    'state': state,
                    'swag_mean': swag_mean,
                    'swag_var': swag_var,
                    'swag_metadata': swag_metadata,
                    'train_recipe': self.task_recipe_kwargs(),
                },
                f,
            )
        print(f'Checkpoint saved to {self.output().path}')


class CIFARPreActSWAG(CIFARRecipeMixin, luigi.Task):
    """Evaluate diagonal-SWAG uncertainty from a trained PreAct ResNet-18 checkpoint."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=200)
    n_perturbations = luigi.IntParameter(default=50)
    swag_start_epoch = luigi.IntParameter(default=160)
    swag_collect_freq = luigi.IntParameter(default=1)
    swag_use_bn_refresh = luigi.BoolParameter(default=True)
    bn_refresh_subset_size = luigi.IntParameter(default=2048)
    seed            = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainSWAGPreActResNet18(
            dataset=self.dataset,
            epochs=self.epochs,
            swag_start_epoch=self.swag_start_epoch,
            swag_collect_freq=self.swag_collect_freq,
            seed=self.seed,
            **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        swag = _cifar_swag_suffix(self, include_bn_refresh=True)
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_swag_n{self.n_perturbations}_e{self.epochs}{recipe}{swag}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(
            f'\n=== CIFAR Diagonal SWAG (n={self.n_perturbations}, '
            f'bn_refresh={self.swag_use_bn_refresh}, bn_subset={self.bn_refresh_subset_size}) ==='
        )
        t0 = time.time()
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))

        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])
        swag_mean = ckpt['swag_mean']
        swag_var = ckpt['swag_var']
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
            bn_refresh_inputs=bn_refresh_inputs,
            bn_refresh_batch_size=min(self.batch_size, max(1, self.bn_refresh_subset_size)),
            use_bn_refresh=self.swag_use_bn_refresh,
            seed=self.seed,
        )

        metrics = _evaluate_cifar('SWAG', ens, x_te, y_te, n_cls)
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
        metrics = _evaluate_cifar('PreAct ResNet-18', ens, x_te, y_te, n_cls)

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'wb') as f:
            pickle.dump({'state': state, 'metrics': metrics, 'train_recipe': self.task_recipe_kwargs()}, f)
        print(f'Checkpoint saved to {self.output().path}')




class CIFARPnC(CIFARRecipeMixin, luigi.Task):
    """Evaluate single-block perturb-and-correct on a chosen PreAct ResNet-18 block."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=200)
    n_directions       = luigi.IntParameter(default=10)
    n_perturbations    = luigi.IntParameter(default=50)
    perturbation_sizes = luigi.ListParameter(default=[10.0, 50.0, 100.0, 200.0])
    subset_size        = luigi.IntParameter(default=1024)
    chunk_size         = luigi.IntParameter(default=128)
    target_stage_idx   = luigi.IntParameter(default=3) # stage4 is index 3
    target_block_idx   = luigi.IntParameter(default=1) # block1 is index 1
    seed               = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        recipe = self.train_recipe_suffix()
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'pnc_single_block_s{self.target_stage_idx}b{self.target_block_idx}_k{self.n_directions}_n{self.n_perturbations}'
                f'_ps{ps}_e{self.epochs}{recipe}_subsetsize{self.subset_size}_seed{self.seed}.json'))

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
        
        # ── Precompute stage inputs ──────────────────────────────────────
        stages = [model.stage1, model.stage2, model.stage3, model.stage4]
        target_stage = stages[self.target_stage_idx]
        target_blk   = target_stage[self.target_block_idx]

        w1_orig = target_blk.conv1.kernel.value
        w2_orig = target_blk.conv2.kernel.value

        def _compute_block_preacts(X_data, chunk_sz):
            """Return (pre_act_chunks, T_orig_chunks) for the target block."""
            n_ch = int(np.ceil(len(X_data) / chunk_sz))
            batches = [X_data[i*chunk_sz:(i+1)*chunk_sz] for i in range(n_ch)]
            pa_chunks, t_chunks = [], []
            for x_batch in batches:
                h = model.stem(x_batch)
                for s_idx, stage in enumerate(stages):
                    for b_idx, blk in enumerate(stage):
                        if s_idx == self.target_stage_idx and b_idx == self.target_block_idx:
                            pa_chunks.append(h)
                            out_bn1   = blk.bn1(h, use_running_average=True)
                            out_relu1 = jax.nn.relu(out_bn1)
                            y_raw     = jax.lax.conv_general_dilated(
                                lhs=out_relu1, rhs=w1_orig.transpose(3, 2, 0, 1),
                                window_strides=blk.conv1.strides, padding='SAME',
                                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
                            y_bn2   = blk.bn2(y_raw, use_running_average=True)
                            y_relu2 = jax.nn.relu(y_bn2)
                            t_chunks.append(blk.conv2(y_relu2))
                            break
                        else:
                            h = blk(h, use_running_average=True)
                    if s_idx == self.target_stage_idx:
                        break
            return pa_chunks, t_chunks

        print('  Precomputing calibration-set block inputs...')
        pre_act_chunks, T_orig_chunks = _compute_block_preacts(X_sub, self.chunk_size)

        print('  Precomputing test-set block inputs...')
        te_pre_act_chunks, te_T_orig_chunks = _compute_block_preacts(x_te, self.chunk_size)
                    
        # get_Y_fn returns RAW (pre-BN2, pre-ReLU) Conv1 output patches.
        # BN2 is intentionally excluded so that the regression features are
        # directly sensitive to w1 perturbations.  W2_new absorbs both the
        # BN2 normalisation and the perturbation effect via least-squares.
        def get_Y_fn(w1, h_in):
            out_bn1 = target_blk.bn1(h_in, use_running_average=True)
            out_relu1 = jax.nn.relu(out_bn1)
            y_raw = jax.lax.conv_general_dilated(
                lhs=out_relu1, rhs=w1.transpose(3, 2, 0, 1),
                window_strides=target_blk.conv1.strides, padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            # No BN2 / ReLU — raw conv1 output so perturbations remain visible
            from pnc import extract_patches
            Y = extract_patches(y_raw, k=3, strides=1)
            return Y

        t0 = time.time()
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
                target_stage_idx=self.target_stage_idx, target_block_idx=self.target_block_idx
            )
            print(f'  Precompute time: {time.time()-t1:.2f}s')
            m = _evaluate_cifar(f'PnC (scale={p_size})', ens, x_te, y_te, n_cls,
                                sidecar_path=self.output().path.replace('.json', f'_ps{p_size}.npz'))
            m["train_time"] = setup_time

            # ── Block-output shift diagnostics (calib & test) ──────────────
            # Calibration arrays already computed during _precompute_corrections.
            raw_calib, corr_calib = ens.calib_raw_arr, ens.calib_corr_arr
            raw_test,  corr_test  = ens.compute_shift_diagnostics(
                te_pre_act_chunks, te_T_orig_chunks, label="test")

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

            m.update(_diag_stats(raw_calib, corr_calib, "diag_calib"))
            m.update(_diag_stats(raw_test,  corr_test,  "diag_test"))
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
    seed = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        recipe = self.train_recipe_suffix()
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.dataset
                / (
                    f"pnc_multi_block_k{self.n_directions}_n{self.n_perturbations}"
                    f"_ps{ps}_lr{self.lambda_reg}_e{self.epochs}{recipe}"
                    f"_subsetsize{self.subset_size}_chunksize{self.chunk_size}_seed{self.seed}.json"
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
        from pnc import find_pnc_subspace_lanczos

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


class CIFARLLLA(CIFARRecipeMixin, luigi.Task):
    """Evaluate last-layer Laplace uncertainty for a trained PreAct ResNet-18."""
    dataset          = luigi.Parameter(default='cifar10')
    epochs           = luigi.IntParameter(default=200)
    n_perturbations  = luigi.IntParameter(default=50)
    prior_precision  = luigi.FloatParameter(default=1.0)
    seed             = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs()
        )

    def output(self) -> luigi.LocalTarget:
        recipe = self.train_recipe_suffix()
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'baseline_llla_n{self.n_perturbations}_prec{self.prior_precision}'
                f'_e{self.epochs}{recipe}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        print(f'\n=== CIFAR LLLA (n={self.n_perturbations}, prior_prec={self.prior_precision}) ===')
        t0 = time.time()
        
        # 1. Load MAP model
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])
        
        # 2. Extract features and compute GGN
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
            # Batch of Fisher matrices: H_i = diag(p_i) - p_i p_i^T
            # (B, K, K)
            H = jax.vmap(lambda p: jnp.diag(p) - jnp.outer(p, p))(probs)
            # Accumulate sum_s (x_hat_s x_hat_s^T) \otimes H_s
            return jnp.einsum('si,sj,skm->ikjm', x_hat, x_hat, H)

        D = 512 # feature dim
        K = n_cls
        G = jnp.zeros((D + 1, K, D + 1, K))
        
        print("  Computing GGN over training set...")
        for i in range(0, len(x_tr), self.batch_size):
            x_batch = x_tr[i:i+self.batch_size]
            feats = get_features(x_batch) # (B, D)
            logits = model.fc(feats) # (B, K)
            probs = jax.nn.softmax(logits) # (B, K)
            X_hat = jnp.concatenate([feats, jnp.ones((feats.shape[0], 1))], axis=-1) # (B, D+1)
            G += compute_batch_ggn(X_hat, probs)

        # Flatten G to ((D+1)K, (D+1)K) matching ravel_pytree order
        G_flat = G.reshape((D + 1) * K, (D + 1) * K)
        
        print("  Inverting GGN...")
        # Add prior precision (lambda I)
        precision = G_flat + self.prior_precision * jnp.eye(G_flat.shape[0])
        covariance = jnp.linalg.inv(precision)
        
        # 4. Create LLLA Ensemble
        fc_state = nnx.state(model.fc)
        ens = LLLAEnsemble(model, fc_state, covariance, self.n_perturbations, self.seed)
        
        train_time = time.time() - t0
        print("  Evaluating LLLA Ensemble...")
        metrics = _evaluate_cifar('LLLA', ens, x_te, y_te, n_cls)
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

    def requires(self) -> list[luigi.Task]:
        shared = dict(dataset=self.dataset, epochs=self.epochs, seed=self.seed, **self.task_recipe_kwargs())
        return [
            CIFARTrainPreActResNet18(**shared),
            CIFARStandardEnsemble(n_models=self.n_models, **shared),
            CIFARPreActMCDropout(n_perturbations=self.n_perturbations, **shared),
            CIFARPreActSWAG(n_perturbations=self.n_perturbations, **shared),
            CIFARPnC(
                n_directions=self.n_directions,
                n_perturbations=self.n_perturbations,
                perturbation_sizes=self.perturbation_sizes,
                **shared,
            ),
            CIFARMultiBlockPnC(
                n_directions=self.n_directions,
                n_perturbations=self.n_perturbations,
                perturbation_sizes=self.perturbation_sizes,
                **shared,
            ),
            CIFARLLLA(n_perturbations=self.n_perturbations, **shared),
        ]
