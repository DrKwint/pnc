"""Luigi tasks for CIFAR-10/100 classification experiments (ResNet-50, BatchNorm Refit PJSVD)."""
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
from pjsvd import find_pjsvd_directions_randomized_svd
from models import PreActResNet18, MCDropoutPreActResNet18, _PreActBasicBlock
from data import load_cifar10, load_cifar100
from training import train_resnet_model
from ensembles import SWAGEnsemble, PJSVDEnsemble, PnCEnsemble, MultiBlockPnCEnsemble, LLLAEnsemble

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


def preact_resnet18_block_indices() -> list[tuple[int, int]]:
    """Ordered (stage_idx, block_idx) for all 8 residual blocks (stage1..stage4, 2 each)."""
    return [(s, b) for s in range(4) for b in range(2)]


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


def make_cifar_block_get_Y_fn(target_blk: nnx.Module):
    """get_Y_fn(w1, h_in) -> patch features of post-bn2-relu conv1 output, for PnC ridge."""

    def get_Y_fn(w1, h_in):
        out_bn1 = target_blk.bn1(h_in, use_running_average=True)
        out_relu1 = jax.nn.relu(out_bn1)
        y_raw = jax.lax.conv_general_dilated(
            lhs=out_relu1,
            rhs=w1.transpose(3, 2, 0, 1),
            window_strides=target_blk.conv1.strides,
            padding="SAME",
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        y_bn2 = target_blk.bn2(y_raw, use_running_average=True)
        y_relu2 = jax.nn.relu(y_bn2)
        from pnc import extract_patches
        kh = target_blk.conv2.kernel_size[0]
        s = target_blk.conv2.strides[0]
        Y = extract_patches(y_relu2, k=kh, strides=s)
        return Y

    return get_Y_fn




def _load_cifar_dataset(dataset: str):
    """Load cifar10 or cifar100 using native data loading; return (x_train, y_train, x_test, y_test, n_classes)."""
    if dataset.lower() == 'cifar10':
        x_tr, y_tr, x_te, y_te = load_cifar10()
        n_cls = 10
    elif dataset.lower() == 'cifar100':
        x_tr, y_tr, x_te, y_te = load_cifar100()
        n_cls = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return x_tr, y_tr, x_te, y_te, n_cls


def _extract_orbax_values(d, path=""):
    if isinstance(d, dict) and 'value' in d:
        return d['value']
    if isinstance(d, dict):
        return {k: _extract_orbax_values(v, f"{path}.{k}" if path else k) for k, v in d.items()}
    return d


def _dict_to_state(d, path=""):
    from flax.nnx.statelib import State

    if isinstance(d, dict):
        # stage1/stage2/stage3 are nnx.List-like and represented as State with integer keys
        if path in ['stage1', 'stage2', 'stage3']:
            state_dict = {}
            for k, v in d.items():
                if k.isdigit():
                    state_dict[int(k)] = _dict_to_state(v, f"{path}.{k}")
                else:
                    raise ValueError(f"Unexpected non-numeric key '{k}' in {path} state")
            return State(state_dict)

        state_dict = {k: _dict_to_state(v, f"{path}.{k}" if path else k) for k, v in d.items()}
        return State(state_dict)

    return d








class CIFARStandardEnsemble(luigi.Task):
    """Train N independent ResNet-50s and evaluate as a deep ensemble."""
    dataset      = luigi.Parameter(default='cifar10')
    epochs       = luigi.IntParameter(default=100)
    n_models     = luigi.IntParameter(default=5)
    batch_size   = luigi.IntParameter(default=128)
    lr           = luigi.FloatParameter(default=1e-3)
    weight_decay = luigi.FloatParameter(default=1e-4)
    seed         = luigi.IntParameter(default=0)

    def requires(self) -> list[luigi.Task]:
        return [CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, seed=self.seed + i
        ) for i in range(self.n_models)]

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'standard_ensemble_n{self.n_models}_e{self.epochs}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

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


class CIFARTrainMCDropoutPreActResNet18(luigi.Task):
    """Train and checkpoint an MCDropoutPreActResNet18."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=100)
    dropout_rate    = luigi.FloatParameter(default=0.1)
    batch_size      = luigi.IntParameter(default=128)
    lr              = luigi.FloatParameter(default=1e-3)
    weight_decay    = luigi.FloatParameter(default=1e-4)
    seed            = luigi.IntParameter(default=0)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'preact_resnet18_mcdropout_dr{self.dropout_rate}_e{self.epochs}_lr{self.lr:.0e}_wd{self.weight_decay:.0e}_seed{self.seed}.pkl'))

    def run(self) -> None:
        seed_everything(self.seed)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(f'\n=== CIFAR Train MC Dropout Model ({self.dataset}, dr={self.dropout_rate}, epochs={self.epochs}) ===')
        t0 = time.time()
        model = MCDropoutPreActResNet18(n_classes=n_cls, dropout_rate=self.dropout_rate, rngs=nnx.Rngs(self.seed))
        model = train_resnet_model(model, x_tr, y_tr, x_va, y_va, epochs=self.epochs,
                                   batch_size=self.batch_size, lr=self.lr,
                                   weight_decay=self.weight_decay, patience=self.epochs)
        
        # Ensure model is ready
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()

        print(f'Training time: {time.time()-t0:.2f}s')

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'wb') as f:
            pickle.dump({'state': state}, f)
        print(f'Checkpoint saved to {self.output().path}')


class CIFARPreActMCDropout(luigi.Task):
    """Load MCDropoutPreActResNet18 and sample N times for uncertainty."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=100)
    n_perturbations = luigi.IntParameter(default=50)
    dropout_rate    = luigi.FloatParameter(default=0.1)
    batch_size      = luigi.IntParameter(default=128)
    lr              = luigi.FloatParameter(default=1e-3)
    weight_decay    = luigi.FloatParameter(default=1e-4)
    seed            = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainMCDropoutPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, dropout_rate=self.dropout_rate,
            batch_size=self.batch_size, lr=self.lr, weight_decay=self.weight_decay,
            seed=self.seed
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'mc_dropout_n{self.n_perturbations}_dr{self.dropout_rate}'
                f'_e{self.epochs}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

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


class CIFARTrainSWAGPreActResNet18(luigi.Task):
    """Train PreActResNet-18 with SWAG and save statistics."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=100)
    batch_size      = luigi.IntParameter(default=128)
    lr              = luigi.FloatParameter(default=1e-3)
    weight_decay    = luigi.FloatParameter(default=1e-4)
    seed            = luigi.IntParameter(default=0)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'preact_resnet18_swag_e{self.epochs}_lr{self.lr:.0e}_wd{self.weight_decay:.0e}_seed{self.seed}.pkl'))

    def run(self) -> None:
        seed_everything(self.seed)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        x_train_full, y_train_full, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        print(f'\n=== CIFAR Train SWAG Model ({self.dataset}, epochs={self.epochs}) ===')
        t0 = time.time()
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        warmup_ep = max(1, self.epochs * 2 // 3)
        model = train_resnet_model(model, x_tr, y_tr, x_va, y_va, epochs=warmup_ep,
                                   batch_size=self.batch_size, lr=self.lr,
                                   weight_decay=self.weight_decay, patience=warmup_ep)

        params       = nnx.state(model, nnx.Param)
        swag_mean    = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        swag_sq_mean = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        n_swag = 0
        for _ in range(self.epochs - warmup_ep):
            model = train_resnet_model(model, x_tr, y_tr, x_va, y_va, epochs=1,
                                       batch_size=self.batch_size, lr=self.lr * 0.01,
                                       weight_decay=self.weight_decay,
                                       warmup_epochs=0, patience=1)
            cur = nnx.state(model, nnx.Param)
            n   = float(n_swag + 1)
            swag_mean    = jax.tree.map(lambda m, p: (m * n_swag + p) / n, swag_mean, cur)
            swag_sq_mean = jax.tree.map(lambda m, p: (m * n_swag + p**2) / n, swag_sq_mean, cur)
            n_swag += 1
        swag_var = jax.tree.map(lambda sq, m: jnp.maximum(sq - m**2, 1e-8), swag_sq_mean, swag_mean)
        
        # Ensure stats are ready
        jax.block_until_ready(swag_mean)
        jax.block_until_ready(swag_var)

        print(f'Training time: {time.time()-t0:.2f}s')

        state = nnx.state(model)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'wb') as f:
            pickle.dump({'state': state, 'swag_mean': swag_mean, 'swag_var': swag_var}, f)
        print(f'Checkpoint saved to {self.output().path}')


class CIFARPreActSWAG(luigi.Task):
    """Load SWAG stats and sample N models for uncertainty."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=100)
    n_perturbations = luigi.IntParameter(default=50)
    batch_size      = luigi.IntParameter(default=128)
    lr              = luigi.FloatParameter(default=1e-3)
    weight_decay    = luigi.FloatParameter(default=1e-4)
    seed            = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainSWAGPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, seed=self.seed
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'swag_n{self.n_perturbations}_e{self.epochs}_seed{self.seed}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        print(f'\n=== CIFAR SWAG (n={self.n_perturbations}) ===')
        t0 = time.time()
        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))

        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])
        swag_mean = ckpt['swag_mean']
        swag_var = ckpt['swag_var']
        print(f'Model load time: {time.time()-t0:.2f}s')

        ens = SWAGEnsemble(model, swag_mean, swag_var, self.n_perturbations)
        def _patched_predict(x):
            ys = []
            for _ in range(ens.n_models):
                sm = ens._sample_model()
                ys.append(sm(x, use_running_average=True))
            return jnp.stack(ys, axis=0)
        ens.predict = _patched_predict

        metrics = _evaluate_cifar('SWAG', ens, x_te, y_te, n_cls)
        metrics["train_time"] = time.time() - t0
        
        # Ensure model is ready
        for p in jax.tree_util.tree_leaves(nnx.state(model)):
            if hasattr(p, 'block_until_ready'):
                p.block_until_ready()
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)


class CIFARPJSVD(luigi.Task):
    """Single-layer BatchNorm Refit PJSVD on ResNet-50 stem conv."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=100)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[0.005, 0.01, 0.05, 0.1])
    subset_size        = luigi.IntParameter(default=512)
    n_oversampling     = luigi.IntParameter(default=10)
    random_directions  = luigi.BoolParameter(default=False)
    seed               = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(dataset=self.dataset, epochs=self.epochs, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        suffix = "_random" if self.random_directions else ""
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'pjsvd_bnrefit_k{self.n_directions}_n{self.n_perturbations}'
                f'_ps{ps}_e{self.epochs}_seed{self.seed}{suffix}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])

        print(f'\n=== CIFAR PJSVD BN Refit (K={self.n_directions}, n={self.n_perturbations}) ===')
        t_start = time.time()
        actual_sub = min(len(x_tr), self.subset_size)
        rng   = np.random.RandomState(self.seed)
        idx   = rng.choice(len(x_tr), actual_sub, replace=False)
        X_sub = x_tr[idx]

        W_stem = model.stem.kernel.value  # (3, 3, 3, 64)

        def model_fn_stem(w):
            return jax.lax.conv_general_dilated(
                lhs=X_sub, rhs=w.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))

        t0 = time.time()
        if self.random_directions:
            print(f'  Generating {self.n_directions} random directions...')
            from pnc import find_random_directions
            v_opts, sigmas = find_random_directions(W_stem.size, self.n_directions, seed=self.seed)
        else:
            print(f'  Finding {self.n_directions} directions via randomized SVD '
                  f'(oversampling={self.n_oversampling})...')
            v_opts, sigmas = find_pjsvd_directions_randomized_svd(
                model_fn_stem, W_stem,
                n_directions=self.n_directions,
                n_oversampling=self.n_oversampling,
                use_full_span=True, seed=self.seed)
        v_opts.block_until_ready()
        print(f'  Direction finding: {time.time()-t0:.2f}s')
        
        setup_time = time.time() - t_start

        rng_z = np.random.RandomState(self.seed)
        all_z = rng_z.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        all_metrics = {}

        for p_size in self.perturbation_sizes:
            print(f'  Building ensemble (p_scale={p_size}) ...')
            t1 = time.time()
            raw_orig = model.stem(X_sub)
            z_orig   = model.stage1[0].bn1(raw_orig, use_running_average=True)
            ens = PJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=["l1"], correction_mode="bn_refit",
                layer_params={"l1": {"W": W_stem}},
                correction_params={"targets": [z_orig]}
            )
            print(f'  Precompute time: {time.time()-t1:.2f}s')
            m = _evaluate_cifar(f'PJSVD BN Refit (scale={p_size})', ens, x_te, y_te, n_cls,
                                sidecar_path=self.output().path.replace('.json', f'_ps{p_size}.npz'))
            m["train_time"] = setup_time
            all_metrics[str(p_size)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(all_metrics, f, indent=2)


class CIFARMLPJSVDv(luigi.Task):
    """Multi-layer BatchNorm Refit PJSVD: stem + stage1-block0-conv1."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=100)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    subset_size        = luigi.IntParameter(default=512)
    n_oversampling     = luigi.IntParameter(default=10)
    random_directions  = luigi.BoolParameter(default=False)
    seed               = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(dataset=self.dataset, epochs=self.epochs, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        suffix = "_random" if self.random_directions else ""
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'ml_pjsvd_bnrefit_k{self.n_directions}_n{self.n_perturbations}'
                f'_ps{ps}_e{self.epochs}_seed{self.seed}{suffix}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])

        print(f'\n=== CIFAR ML-PJSVD BN Refit (K={self.n_directions}, n={self.n_perturbations}) ===')
        t_start = time.time()
        actual_sub = min(len(x_tr), self.subset_size)
        rng   = np.random.RandomState(self.seed)
        idx   = rng.choice(len(x_tr), actual_sub, replace=False)
        X_sub = x_tr[idx]

        W_stem = model.stem.kernel.value           # (3,3,3,64)
        W_s1c1 = model.stage1[0].conv1.kernel.value  # (1,1,64,64)
        W_joint_flat = jnp.concatenate([W_stem.reshape(-1), W_s1c1.reshape(-1)])

        def model_fn_flat(w_flat):
            w_st = w_flat[:W_stem.size].reshape(W_stem.shape)
            w_sc = w_flat[W_stem.size:].reshape(W_s1c1.shape)
            h_st = jax.lax.conv_general_dilated(
                lhs=X_sub, rhs=w_st.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            h_st_bn  = model.stage1[0].bn1(h_st, use_running_average=True)
            h_st_act = jax.nn.relu(h_st_bn)
            h_sc = jax.lax.conv_general_dilated(
                lhs=h_st_act, rhs=w_sc.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            return h_sc

        t0 = time.time()
        if self.random_directions:
            print(f'  Generating {self.n_directions} random multi-layer directions...')
            from pnc import find_random_directions
            v_opts, sigmas = find_random_directions(W_joint_flat.size, self.n_directions, seed=self.seed)
        else:
            print(f'  Finding {self.n_directions} multi-layer directions via randomized SVD...')
            v_opts, sigmas = find_pjsvd_directions_randomized_svd(
                model_fn_flat, W_joint_flat,
                n_directions=self.n_directions,
                n_oversampling=self.n_oversampling,
                use_full_span=True, seed=self.seed)
        v_opts.block_until_ready()
        print(f'  Direction finding: {time.time()-t0:.2f}s')
        
        setup_time = time.time() - t_start

        rng_z = np.random.RandomState(self.seed)
        all_z = rng_z.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        all_metrics = {}

        for p_size in self.perturbation_sizes:
            print(f'  Building ML ensemble (p_scale={p_size}) ...')
            t1 = time.time()
            raw_stem_orig = model.stem.conv(X_sub)
            z_stem_orig   = model.stem.bn(raw_stem_orig, use_running_average=True)
            h_st_orig     = jax.nn.relu(z_stem_orig)
            blk0 = model.stage1[0]
            raw_s1c1_orig = blk0.conv1.conv(h_st_orig)
            z_s1c1_orig   = blk0.conv1.bn(raw_s1c1_orig, use_running_average=True)

            ens = PJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, X_sub=X_sub,
                layers=["l1", "l2"], correction_mode="bn_refit",
                layer_params={"l1": {"W": W_stem}, "l2": {"W": W_s1c1}},
                correction_params={"targets": [z_stem_orig, z_s1c1_orig]}
            )
            print(f'  Precompute time: {time.time()-t1:.2f}s')
            m = _evaluate_cifar(f'ML-PJSVD BN Refit (scale={p_size})', ens, x_te, y_te, n_cls,
                                sidecar_path=self.output().path.replace('.json', f'_ps{p_size}.npz'))
            m["train_time"] = setup_time
            all_metrics[str(p_size)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(all_metrics, f, indent=2)


class CIFARTrainPreActResNet18(luigi.Task):
    """Train and checkpoint a PreAct ResNet-18 on CIFAR-10."""
    dataset      = luigi.Parameter(default='cifar10')
    epochs       = luigi.IntParameter(default=100)
    batch_size   = luigi.IntParameter(default=128)
    lr           = luigi.FloatParameter(default=1e-3)
    weight_decay = luigi.FloatParameter(default=1e-4)
    seed         = luigi.IntParameter(default=0)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'preact_resnet18_e{self.epochs}_lr{self.lr:.0e}_wd{self.weight_decay:.0e}'
                f'_seed{self.seed}.pkl'))

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
            epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay,
            warmup_epochs=min(5, max(0, self.epochs - 1)),
            patience=self.epochs)
        
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
            pickle.dump({'state': state, 'metrics': metrics}, f)
        print(f'Checkpoint saved to {self.output().path}')




class CIFARPnC(luigi.Task):
    """Perturb-and-Correct on a specified ResNet-18 block using chunked operators."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=100)
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

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(dataset=self.dataset, epochs=self.epochs, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        suffix = "_random" if self.random_directions else ""
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'pnc_s{self.target_stage_idx}b{self.target_block_idx}_k{self.n_directions}_n{self.n_perturbations}'
                f'_ps{ps}_lr{self.lambda_reg}_e{self.epochs}_subsetsize{self.subset_size}_seed{self.seed}{suffix}.json'))

    def run(self) -> None:
        seed_everything(self.seed)
        if self.dataset.lower() == 'cifar10':
            from data import load_cifar10
            x_train_full, y_train_full, x_te, y_te = load_cifar10()
            n_cls = 10
        elif self.dataset.lower() == 'cifar100':
            from data import load_cifar100
            x_train_full, y_train_full, x_te, y_te = load_cifar100()
            n_cls = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
            
        from util import _split_data
        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])

        print(f'\\n=== CIFAR PnC Stage {self.target_stage_idx} Block {self.target_block_idx} (K={self.n_directions}, n={self.n_perturbations}) ===')
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
                                sidecar_path=self.output().path.replace('.json', f'_ps{p_size}.npz'))
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


class CIFARMultiBlockPnC(luigi.Task):
    """
    Multi-block PnC (approach B): Lanczos + ridge independently per residual block;
    MAP calibration per block. z_coeffs shape (N, 8, K). Ridge solves show a tqdm bar.
    """

    dataset = luigi.Parameter(default="cifar10")
    epochs = luigi.IntParameter(default=100)
    n_directions = luigi.IntParameter(default=16)
    n_perturbations = luigi.IntParameter(default=32)
    perturbation_sizes = luigi.ListParameter(default=[1.0, 5.0, 10.0, 50.0])
    subset_size = luigi.IntParameter(default=1024)
    chunk_size = luigi.IntParameter(default=64)
    lambda_reg = luigi.FloatParameter(default=1e-3)
    random_directions = luigi.BoolParameter(default=False)
    seed = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(dataset=self.dataset, epochs=self.epochs, seed=self.seed)

    def output(self) -> luigi.LocalTarget:
        ps = _ps_str(self.perturbation_sizes)
        suffix = "_random" if self.random_directions else ""
        return luigi.LocalTarget(
            str(
                Path("results")
                / self.dataset
                / f"pnc_allblocks_k{self.n_directions}_n{self.n_perturbations}"
                f"_ps{ps}_e{self.epochs}_subsetsize{self.subset_size}_seed{self.seed}{suffix}.json"
            )
        )

    def run(self) -> None:
        seed_everything(self.seed)
        if self.dataset.lower() == "cifar10":
            from data import load_cifar10

            x_train_full, y_train_full, x_te, y_te = load_cifar10()
            n_cls = 10
        elif self.dataset.lower() == "cifar100":
            from data import load_cifar100

            x_train_full, y_train_full, x_te, y_te = load_cifar100()
            n_cls = 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        from util import _split_data

        x_tr, y_tr, x_va, y_va = _split_data(x_train_full, y_train_full)

        model = PreActResNet18(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, "rb") as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt["state"])

        print(
            f"\\n=== CIFAR Multi-block PnC (8 blocks, K={self.n_directions}, n={self.n_perturbations}) ==="
        )
        t_start = time.time()

        actual_sub = min(len(x_tr), self.subset_size)
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(x_tr), actual_sub, replace=False)
        X_sub = x_tr[idx]

        stages = [model.stage1, model.stage2, model.stage3, model.stage4]
        block_indices = preact_resnet18_block_indices()

        # Sequential processing to avoid redundant forward passes and excessive memory usage
        v_opts_list: list = []
        sigmas_list: list = []
        w1_list: list = []
        w2_list: list = []
        get_Y_fns: list = []
        
        # Solved weights per block per member: (n_blk, n_mem, 3) where 3 is (w1, w2, b2)
        block_member_weights = []
        
        # Full diagnostics (n_blk, n_mem)
        raw_calib_full = []
        corr_calib_full = []
        raw_test_full = []
        corr_test_full = []

        from pnc import find_pnc_subspace_lanczos, solve_chunked_conv2_correction

        # Initialize activations from stem
        @nnx.jit
        def run_stem_jit(x):
            return model.stem(x)

        n_ch_c = int(np.ceil(len(X_sub) / self.chunk_size))
        n_ch_te = int(np.ceil(len(x_te) / self.chunk_size))

        h_calib_chunks = [run_stem_jit(X_sub[i*self.chunk_size:(i+1)*self.chunk_size]) for i in range(n_ch_c)]
        h_test_chunks = [run_stem_jit(x_te[i*self.chunk_size:(i+1)*self.chunk_size]) for i in range(n_ch_te)]
        
        n_mem = self.n_perturbations
        rng_z = np.random.RandomState(self.seed + 999)
        all_z = rng_z.normal(0, 1, size=(n_mem, len(block_indices), self.n_directions))

        for b_idx_flat, (s_idx, b_idx) in enumerate(tqdm(
            block_indices, desc="Multi-block PnC: Lanczos + Ridge + Diag", unit="block"
        )):
            target_blk = stages[s_idx][b_idx]
            w1_orig = target_blk.conv1.kernel.value
            w2_orig = target_blk.conv2.kernel.value
            w1_list.append(w1_orig)
            w2_list.append(w2_orig)
            
            get_Y_fn = make_cifar_block_get_Y_fn(target_blk)
            get_Y_fns.append(get_Y_fn)

            # 1. Run Lanczos (Calibration input is current h_calib_chunks)
            t0 = time.time()
            if self.random_directions:
                from pnc import find_random_directions
                v_opts, sigmas = find_random_directions(
                    w1_orig.size,
                    self.n_directions,
                    seed=self.seed + 17 * s_idx + 31 * b_idx,
                )
            else:
                v_opts, sigmas = find_pnc_subspace_lanczos(
                    get_Y_fn,
                    w1_orig,
                    h_calib_chunks,
                    K=self.n_directions,
                    seed=self.seed + 17 * s_idx + 31 * b_idx,
                )
            v_opts_list.append(v_opts)
            sigmas_list.append(sigmas)

            # 2. Advance h_calib and collect T_c for Ridge solve
            T_c_chunks = []
            new_h_c_chunks = []
            for h in h_calib_chunks:
                h_next, t = _forward_block_jit(target_blk, h, w1_orig)
                new_h_c_chunks.append(h_next)
                T_c_chunks.append(t)

            # 3. Solve Ridge for each member (sequential to save VRAM)
            members_this_block = []
            kh, kw, C_in, C_out = w2_orig.shape
            w2_flat_orig = w2_orig.reshape(-1, C_out)
            
            def _coeffs_from_z(z_row, sigmas, p_scale):
                safe_sigmas = sigmas + 1e-6
                coeffs = z_row / safe_sigmas
                norm = np.linalg.norm(coeffs) + 1e-12
                return (coeffs / norm) * p_scale

            for i in range(n_mem):
                z_row = all_z[i, b_idx_flat, :]
                coeffs = _coeffs_from_z(z_row, sigmas, self.perturbation_sizes[0])
                dp = coeffs @ v_opts
                w1_pert = w1_orig + dp.reshape(w1_orig.shape)
                
                w2_pert, b2_pert = solve_chunked_conv2_correction(
                    get_Y_fn, w1_pert, w2_orig, h_calib_chunks, T_c_chunks,
                    lambda_reg=self.lambda_reg
                )
                members_this_block.append((w1_pert, w2_pert, b2_pert))

            block_member_weights.append(members_this_block)

            # 4. Compute Diagnostics and Advance h_test Chunks simultaneously
            raw_c_norms, corr_c_norms = [], []
            for w1_p, w2_p, b2_p in members_this_block:
                w2_p_f = w2_p.reshape(-1, C_out)
                
                raw_err_sq, corr_err_sq, n_tot = 0.0, 0.0, 0
                for ch, to in zip(h_calib_chunks, T_c_chunks):
                    n = ch.shape[0]
                    Y_p = get_Y_fn(w1_p, ch)
                    diff_raw = (Y_p @ w2_flat_orig).reshape(to.shape) - to
                    diff_corr = (Y_p @ w2_p_f + b2_p).reshape(to.shape) - to
                    raw_err_sq += float(jnp.sum(diff_raw**2))
                    corr_err_sq += float(jnp.sum(diff_corr**2))
                    n_tot += n
                raw_c_norms.append(np.sqrt(raw_err_sq / n_tot))
                corr_c_norms.append(np.sqrt(corr_err_sq / n_tot))
            
            raw_calib_full.append(raw_c_norms)
            corr_calib_full.append(corr_c_norms)

            # Process test chunks one by one
            new_h_te_chunks = []
            raw_te_err_sq_sum = np.zeros(n_mem)
            corr_te_err_sq_sum = np.zeros(n_mem)
            total_te_samples = 0

            @nnx.jit
            def diag_test_step(h_ch, w1_p, w2_p_f, b2_p, t_orig):
                Y_p = get_Y_fn(w1_p, h_ch)
                t_raw = (Y_p @ w2_flat_orig).reshape(t_orig.shape)
                t_corr = (Y_p @ w2_p_f + b2_p).reshape(t_orig.shape)
                return jnp.sum((t_raw - t_orig)**2), jnp.sum((t_corr - t_orig)**2)

            for h in h_test_chunks:
                h_next, t_orig = _forward_block_jit(target_blk, h, w1_orig)
                new_h_te_chunks.append(h_next)
                
                n = h.shape[0]
                total_te_samples += n
                
                for i in range(n_mem):
                    w1_p, w2_p, b2_p = members_this_block[i]
                    w2_p_f = w2_p.reshape(-1, C_out)
                    r_sq, c_sq = diag_test_step(h, w1_p, w2_p_f, b2_p, t_orig)
                    raw_te_err_sq_sum[i] += float(r_sq)
                    corr_te_err_sq_sum[i] += float(c_sq)
                
                del t_orig

            raw_test_full.append(np.sqrt(raw_te_err_sq_sum / total_te_samples))
            corr_test_full.append(np.sqrt(corr_te_err_sq_sum / total_te_samples))

            # Update h_chunks
            h_calib_chunks = new_h_c_chunks
            h_test_chunks = new_h_te_chunks
            
            del T_c_chunks, new_h_c_chunks
            jax.clear_caches()

        setup_time = time.time() - t_start

        # Transpose weights: (n_blk, n_mem) -> (n_mem, n_blk)
        members_transposed = [[block_member_weights[b][i] for b in range(len(block_indices))] for i in range(n_mem)]
        
        # Aggregated diagnostics: (n_mem,) mean over blocks
        final_raw_calib = np.mean(raw_calib_full, axis=0)
        final_corr_calib = np.mean(corr_calib_full, axis=0)
        final_raw_test = np.mean(raw_test_full, axis=0)
        final_corr_test = np.mean(corr_test_full, axis=0)

        # Prepare specs for the ensemble (weights already solved, chunks not needed)
        block_specs_empty = []
        for b in range(len(block_indices)):
            block_specs_empty.append({
                "stage_idx": block_indices[b][0], "block_idx": block_indices[b][1],
                "w1_orig": w1_list[b], "w2_orig": w2_list[b],
                "v_opts": v_opts_list[b], "sigmas": sigmas_list[b],
                "chunks": [], "T_orig_chunks": [], "get_Y_fn": get_Y_fns[b]
            })

        all_metrics = {}

        for p_size in self.perturbation_sizes:
            print(f"  Building multi-block PnC ensemble (p_scale={p_size}) ...")
            t1 = time.time()
            ens = MultiBlockPnCEnsemble(
                base_model=model,
                block_specs=block_specs_empty,
                z_coeffs=all_z,
                perturbation_scale=float(p_size),
                members=members_transposed,
                raw_calib_arr=final_raw_calib,
                corr_calib_arr=final_corr_calib,
            )
            print(f"  Precompute (re-scaling): {time.time() - t1:.2f}s")
            
            m = _evaluate_cifar(
                f"Multi-block PnC (scale={p_size})",
                ens, x_te, y_te, n_cls,
                sidecar_path=self.output().path.replace(".json", f"_ps{p_size}.npz"),
            )
            m["train_time"] = setup_time

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

            # Scale diagnostics by p_size/original_p_size if p_size != first size
            diag_scale = p_size / self.perturbation_sizes[0]
            m.update(_diag_stats(final_raw_calib * diag_scale, final_corr_calib * diag_scale, "diag_calib"))
            m.update(_diag_stats(final_raw_test * diag_scale, final_corr_test * diag_scale, "diag_test"))
            all_metrics[str(p_size)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, "w") as f:
            json.dump(all_metrics, f, indent=2)




class CIFARLLLA(luigi.Task):
    """Last-Layer Laplace Approximation for PreAct ResNet-18."""
    dataset          = luigi.Parameter(default='cifar10')
    epochs           = luigi.IntParameter(default=100)
    n_perturbations  = luigi.IntParameter(default=50)
    batch_size       = luigi.IntParameter(default=128)
    lr               = luigi.FloatParameter(default=1e-3)
    weight_decay     = luigi.FloatParameter(default=1e-4)
    prior_precision  = luigi.FloatParameter(default=1.0)
    seed             = luigi.IntParameter(default=0)

    def requires(self) -> luigi.Task:
        return CIFARTrainPreActResNet18(
            dataset=self.dataset, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, seed=self.seed
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'llla_n{self.n_perturbations}_prec{self.prior_precision}'
                f'_e{self.epochs}_seed{self.seed}.json'))

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


class AllCIFARExperiments(luigi.WrapperTask):
    """Umbrella task: runs all CIFAR experiments for a given dataset."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=100)
    n_perturbations    = luigi.IntParameter(default=100)
    n_models           = luigi.IntParameter(default=5)
    n_directions       = luigi.IntParameter(default=40)
    perturbation_sizes = luigi.ListParameter(default=[0.01, 0.05, 0.1, 0.5])
    seed               = luigi.IntParameter(default=0)

    def requires(self) -> list[luigi.Task]:
        shared = dict(dataset=self.dataset, epochs=self.epochs, seed=self.seed)
        ml_ps  = [ps / 5 for ps in self.perturbation_sizes]
        return [
            CIFARTrainPreActResNet18(**shared),
            CIFARStandardEnsemble(n_models=self.n_models, **shared),
            CIFARPreActMCDropout(n_perturbations=self.n_perturbations, **shared),
            CIFARPreActSWAG(n_perturbations=self.n_perturbations, **shared),
            CIFARPJSVD(n_directions=self.n_directions,
                       n_perturbations=self.n_perturbations,
                       perturbation_sizes=self.perturbation_sizes, **shared),
            CIFARMLPJSVDv(n_directions=self.n_directions,
                           n_perturbations=self.n_perturbations,
                           perturbation_sizes=ml_ps, **shared),
            CIFARPnC(n_directions=self.n_directions,
                           n_perturbations=self.n_perturbations,
                           perturbation_sizes=self.perturbation_sizes, **shared),
            CIFARLLLA(n_perturbations=self.n_perturbations, **shared),
        ]
