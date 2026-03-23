"""Luigi tasks for CIFAR-10/100 classification experiments (ResNet-50, BatchNorm Refit PJSVD)."""
import time
import json
import pickle
from pathlib import Path

import luigi
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from util import seed_everything, _evaluate_cifar, _ps_str
from pjsvd import find_pjsvd_directions_randomized_svd
from models import ResNet50, MCDropoutResNet50
from data import load_cifar10, load_cifar100
from training import train_resnet_model
from ensembles import SWAGEnsemble, BatchNormRefitPJSVDEnsemble, MLBatchNormRefitPJSVDEnsemble


def _load_cifar_dataset(dataset: str):
    """Load cifar10 or cifar100; return (x_train, y_train, x_test, y_test, n_classes)."""
    dataset = dataset.lower()
    if dataset == 'cifar10':
        x_tr, y_tr, x_te, y_te = load_cifar10()
        return x_tr, y_tr, x_te, y_te, 10
    elif dataset == 'cifar100':
        x_tr, y_tr, x_te, y_te = load_cifar100()
        return x_tr, y_tr, x_te, y_te, 100
    else:
        raise ValueError(f"Unknown CIFAR dataset: {dataset}. Choose cifar10 or cifar100.")


class CIFARTrainBaseModel(luigi.Task):
    """Train and checkpoint a ResNet-50 on CIFAR-10 or CIFAR-100."""
    dataset      = luigi.Parameter(default='cifar10')
    epochs       = luigi.IntParameter(default=100)
    batch_size   = luigi.IntParameter(default=128)
    lr           = luigi.FloatParameter(default=1e-3)
    weight_decay = luigi.FloatParameter(default=1e-4)
    seed         = luigi.IntParameter(default=0)

    def output(self):
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'resnet50_e{self.epochs}_lr{self.lr:.0e}_wd{self.weight_decay:.0e}'
                f'_seed{self.seed}.pkl'))

    def run(self):
        seed_everything(self.seed)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        print(f'\n=== CIFAR Train Base Model ({self.dataset}, epochs={self.epochs}) ===')

        t0 = time.time()
        model = ResNet50(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        model = train_resnet_model(
            model, x_tr, y_tr,
            epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay)
        print(f'Training time: {time.time()-t0:.2f}s')

        class _SingleEns:
            def __init__(self, m): self.m = m
            def predict(self, x):
                return jnp.expand_dims(self.m(x, use_running_average=True), axis=0)

        ens = _SingleEns(model)
        metrics = _evaluate_cifar('ResNet-50 (base)', ens, x_te, y_te, n_cls)

        state = nnx.state(model)
        with open(self.output().path, 'wb') as f:
            pickle.dump({'state': state, 'metrics': metrics}, f)
        print(f'Checkpoint saved to {self.output().path}')


class CIFARStandardEnsemble(luigi.Task):
    """Train N independent ResNet-50s and evaluate as a deep ensemble."""
    dataset      = luigi.Parameter(default='cifar10')
    epochs       = luigi.IntParameter(default=100)
    n_models     = luigi.IntParameter(default=5)
    batch_size   = luigi.IntParameter(default=128)
    lr           = luigi.FloatParameter(default=1e-3)
    weight_decay = luigi.FloatParameter(default=1e-4)
    seed         = luigi.IntParameter(default=0)

    def requires(self):
        return [CIFARTrainBaseModel(
            dataset=self.dataset, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, seed=self.seed + i
        ) for i in range(self.n_models)]

    def output(self):
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'standard_ensemble_n{self.n_models}_e{self.epochs}_seed{self.seed}.json'))

    def run(self):
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        print(f'\n=== CIFAR Standard Ensemble (n={self.n_models}) ===')
        t0 = time.time()
        models = []
        for i, inp in enumerate(self.input()):
            m = ResNet50(n_classes=n_cls, rngs=nnx.Rngs(self.seed + i))
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
        metrics = _evaluate_cifar('Standard Ensemble', ens, x_te, y_te, n_cls)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)


class CIFARTrainMCDropoutModel(luigi.Task):
    """Train and checkpoint an MCDropoutResNet50."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=100)
    dropout_rate    = luigi.FloatParameter(default=0.1)
    batch_size      = luigi.IntParameter(default=128)
    lr              = luigi.FloatParameter(default=1e-3)
    weight_decay    = luigi.FloatParameter(default=1e-4)
    seed            = luigi.IntParameter(default=0)

    def output(self):
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'resnet50_mcdropout_dr{self.dropout_rate}_e{self.epochs}_lr{self.lr:.0e}_wd{self.weight_decay:.0e}_seed{self.seed}.pkl'))

    def run(self):
        seed_everything(self.seed)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        print(f'\n=== CIFAR Train MC Dropout Model ({self.dataset}, dr={self.dropout_rate}, epochs={self.epochs}) ===')
        t0 = time.time()
        model = MCDropoutResNet50(n_classes=n_cls, dropout_rate=self.dropout_rate, rngs=nnx.Rngs(self.seed))
        model = train_resnet_model(model, x_tr, y_tr, epochs=self.epochs,
                                   batch_size=self.batch_size, lr=self.lr,
                                   weight_decay=self.weight_decay)
        print(f'Training time: {time.time()-t0:.2f}s')

        state = nnx.state(model)
        with open(self.output().path, 'wb') as f:
            pickle.dump({'state': state}, f)
        print(f'Checkpoint saved to {self.output().path}')


class CIFARMCDropout(luigi.Task):
    """Load MCDropoutResNet50 and sample N times for uncertainty."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=100)
    n_perturbations = luigi.IntParameter(default=50)
    dropout_rate    = luigi.FloatParameter(default=0.1)
    batch_size      = luigi.IntParameter(default=128)
    lr              = luigi.FloatParameter(default=1e-3)
    weight_decay    = luigi.FloatParameter(default=1e-4)
    seed            = luigi.IntParameter(default=0)

    def requires(self):
        return CIFARTrainMCDropoutModel(
            dataset=self.dataset, epochs=self.epochs, dropout_rate=self.dropout_rate,
            batch_size=self.batch_size, lr=self.lr, weight_decay=self.weight_decay,
            seed=self.seed
        )

    def output(self):
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'mc_dropout_n{self.n_perturbations}_dr{self.dropout_rate}'
                f'_e{self.epochs}_seed{self.seed}.json'))

    def run(self):
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        print(f'\n=== CIFAR MC Dropout (n={self.n_perturbations}, dr={self.dropout_rate}) ===')
        t0 = time.time()
        model = MCDropoutResNet50(n_classes=n_cls, dropout_rate=self.dropout_rate, rngs=nnx.Rngs(self.seed))
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
        metrics = _evaluate_cifar('MC Dropout', ens, x_te, y_te, n_cls)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)


class CIFARTrainSWAGModel(luigi.Task):
    """Train ResNet-50 with SWAG and save statistics."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=100)
    batch_size      = luigi.IntParameter(default=128)
    lr              = luigi.FloatParameter(default=1e-3)
    weight_decay    = luigi.FloatParameter(default=1e-4)
    seed            = luigi.IntParameter(default=0)

    def output(self):
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'resnet50_swag_e{self.epochs}_lr{self.lr:.0e}_wd{self.weight_decay:.0e}_seed{self.seed}.pkl'))

    def run(self):
        seed_everything(self.seed)
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        print(f'\n=== CIFAR Train SWAG Model ({self.dataset}, epochs={self.epochs}) ===')
        t0 = time.time()
        model = ResNet50(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        warmup_ep = max(1, self.epochs * 2 // 3)
        model = train_resnet_model(model, x_tr, y_tr, epochs=warmup_ep,
                                   batch_size=self.batch_size, lr=self.lr,
                                   weight_decay=self.weight_decay)

        params       = nnx.state(model, nnx.Param)
        swag_mean    = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        swag_sq_mean = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        n_swag = 0
        for _ in range(self.epochs - warmup_ep):
            model = train_resnet_model(model, x_tr, y_tr, epochs=1,
                                       batch_size=self.batch_size, lr=self.lr * 0.01,
                                       weight_decay=self.weight_decay,
                                       warmup_epochs=0)
            cur = nnx.state(model, nnx.Param)
            n   = float(n_swag + 1)
            swag_mean    = jax.tree.map(lambda m, p: (m * n_swag + p) / n, swag_mean, cur)
            swag_sq_mean = jax.tree.map(lambda m, p: (m * n_swag + p**2) / n, swag_sq_mean, cur)
            n_swag += 1
        swag_var = jax.tree.map(lambda sq, m: jnp.maximum(sq - m**2, 1e-8), swag_sq_mean, swag_mean)
        print(f'Training time: {time.time()-t0:.2f}s')

        state = nnx.state(model)
        with open(self.output().path, 'wb') as f:
            pickle.dump({'state': state, 'swag_mean': swag_mean, 'swag_var': swag_var}, f)
        print(f'Checkpoint saved to {self.output().path}')


class CIFARSWAG(luigi.Task):
    """Load SWAG stats and sample N models for uncertainty."""
    dataset         = luigi.Parameter(default='cifar10')
    epochs          = luigi.IntParameter(default=100)
    n_perturbations = luigi.IntParameter(default=50)
    batch_size      = luigi.IntParameter(default=128)
    lr              = luigi.FloatParameter(default=1e-3)
    weight_decay    = luigi.FloatParameter(default=1e-4)
    seed            = luigi.IntParameter(default=0)

    def requires(self):
        return CIFARTrainSWAGModel(
            dataset=self.dataset, epochs=self.epochs, batch_size=self.batch_size,
            lr=self.lr, weight_decay=self.weight_decay, seed=self.seed
        )

    def output(self):
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'swag_n{self.n_perturbations}_e{self.epochs}_seed{self.seed}.json'))

    def run(self):
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        print(f'\n=== CIFAR SWAG (n={self.n_perturbations}) ===')
        t0 = time.time()
        model = ResNet50(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        
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
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(metrics, f, indent=2)


class CIFARPJSVD(luigi.Task):
    """Single-layer BatchNorm Refit PJSVD on ResNet-50 stem conv."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=100)
    n_directions       = luigi.IntParameter(default=40)
    n_perturbations    = luigi.IntParameter(default=100)
    perturbation_sizes = luigi.ListParameter(default=[0.01, 0.05, 0.1, 0.5])
    subset_size        = luigi.IntParameter(default=512)
    n_oversampling     = luigi.IntParameter(default=10)
    seed               = luigi.IntParameter(default=0)

    def requires(self):
        return CIFARTrainBaseModel(dataset=self.dataset, epochs=self.epochs, seed=self.seed)

    def output(self):
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'pjsvd_bnrefit_k{self.n_directions}_n{self.n_perturbations}'
                f'_ps{ps}_e{self.epochs}_seed{self.seed}.json'))

    def run(self):
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        model = ResNet50(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])

        print(f'\n=== CIFAR PJSVD BN Refit (K={self.n_directions}, n={self.n_perturbations}) ===')
        actual_sub = min(len(x_tr), self.subset_size)
        idx   = np.random.choice(len(x_tr), actual_sub, replace=False)
        X_sub = x_tr[idx]

        W_stem = model.stem.conv.kernel.value  # (3, 3, 3, 64)

        def model_fn_stem(w):
            return jax.lax.conv_general_dilated(
                lhs=X_sub, rhs=w.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))

        t0 = time.time()
        print(f'  Finding {self.n_directions} directions via randomized SVD '
              f'(oversampling={self.n_oversampling})...')
        v_opts, sigmas = find_pjsvd_directions_randomized_svd(
            model_fn_stem, W_stem,
            n_directions=self.n_directions,
            n_oversampling=self.n_oversampling,
            use_full_span=True, seed=self.seed)
        print(f'  Direction finding: {time.time()-t0:.2f}s')

        all_z = np.random.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        all_metrics = {}

        for p_size in self.perturbation_sizes:
            print(f'  Building ensemble (p_scale={p_size}) ...')
            t1 = time.time()
            ens = BatchNormRefitPJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, W_stem=W_stem, X_sub=X_sub)
            print(f'  Precompute time: {time.time()-t1:.2f}s')
            m = _evaluate_cifar(f'PJSVD BN Refit (scale={p_size})', ens, x_te, y_te, n_cls,
                                sidecar_path=self.output().path.replace('.json', f'_ps{p_size}.npz'))
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
    perturbation_sizes = luigi.ListParameter(default=[0.005, 0.01, 0.05, 0.1])
    subset_size        = luigi.IntParameter(default=512)
    n_oversampling     = luigi.IntParameter(default=10)
    seed               = luigi.IntParameter(default=0)

    def requires(self):
        return CIFARTrainBaseModel(dataset=self.dataset, epochs=self.epochs, seed=self.seed)

    def output(self):
        ps = _ps_str(self.perturbation_sizes)
        return luigi.LocalTarget(
            str(Path('results') / self.dataset /
                f'ml_pjsvd_bnrefit_k{self.n_directions}_n{self.n_perturbations}'
                f'_ps{ps}_e{self.epochs}_seed{self.seed}.json'))

    def run(self):
        seed_everything(self.seed)
        x_tr, y_tr, x_te, y_te, n_cls = _load_cifar_dataset(self.dataset)

        model = ResNet50(n_classes=n_cls, rngs=nnx.Rngs(self.seed))
        with open(self.input().path, 'rb') as f:
            ckpt = pickle.load(f)
        nnx.update(model, ckpt['state'])

        print(f'\n=== CIFAR ML-PJSVD BN Refit (K={self.n_directions}, n={self.n_perturbations}) ===')
        actual_sub = min(len(x_tr), self.subset_size)
        idx   = np.random.choice(len(x_tr), actual_sub, replace=False)
        X_sub = x_tr[idx]

        W_stem = model.stem.conv.kernel.value           # (3,3,3,64)
        W_s1c1 = model.stage1[0].conv1.conv.kernel.value  # (1,1,64,64)
        W_joint_flat = jnp.concatenate([W_stem.reshape(-1), W_s1c1.reshape(-1)])

        def model_fn_flat(w_flat):
            w_st = w_flat[:W_stem.size].reshape(W_stem.shape)
            w_sc = w_flat[W_stem.size:].reshape(W_s1c1.shape)
            h_st = jax.lax.conv_general_dilated(
                lhs=X_sub, rhs=w_st.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            h_st_bn  = model.stem.bn(h_st, use_running_average=True)
            h_st_act = jax.nn.relu(h_st_bn)
            h_sc = jax.lax.conv_general_dilated(
                lhs=h_st_act, rhs=w_sc.transpose(3, 2, 0, 1),
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
            return h_sc

        t0 = time.time()
        print(f'  Finding {self.n_directions} multi-layer directions via randomized SVD...')
        v_opts, sigmas = find_pjsvd_directions_randomized_svd(
            model_fn_flat, W_joint_flat,
            n_directions=self.n_directions,
            n_oversampling=self.n_oversampling,
            use_full_span=True, seed=self.seed)
        print(f'  Direction finding: {time.time()-t0:.2f}s')

        all_z = np.random.normal(0, 1, size=(self.n_perturbations, self.n_directions))
        all_metrics = {}

        for p_size in self.perturbation_sizes:
            print(f'  Building ML ensemble (p_scale={p_size}) ...')
            t1 = time.time()
            ens = MLBatchNormRefitPJSVDEnsemble(
                base_model=model, v_opts=v_opts, sigmas=sigmas, z_coeffs=all_z,
                perturbation_scale=p_size, W_stem=W_stem, W_s1c1=W_s1c1, X_sub=X_sub)
            print(f'  Precompute time: {time.time()-t1:.2f}s')
            m = _evaluate_cifar(f'ML-PJSVD BN Refit (scale={p_size})', ens, x_te, y_te, n_cls,
                                sidecar_path=self.output().path.replace('.json', f'_ps{p_size}.npz'))
            all_metrics[str(p_size)] = m

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output().path, 'w') as f:
            json.dump(all_metrics, f, indent=2)


class AllCIFARExperiments(luigi.WrapperTask):
    """Umbrella task: runs all CIFAR experiments for a given dataset."""
    dataset            = luigi.Parameter(default='cifar10')
    epochs             = luigi.IntParameter(default=100)
    n_perturbations    = luigi.IntParameter(default=100)
    n_models           = luigi.IntParameter(default=5)
    n_directions       = luigi.IntParameter(default=40)
    perturbation_sizes = luigi.ListParameter(default=[0.01, 0.05, 0.1, 0.5])
    seed               = luigi.IntParameter(default=0)

    def requires(self):
        shared = dict(dataset=self.dataset, epochs=self.epochs, seed=self.seed)
        ml_ps  = [ps / 5 for ps in self.perturbation_sizes]
        return [
            CIFARTrainBaseModel(**shared),
            CIFARStandardEnsemble(n_models=self.n_models, **shared),
            CIFARMCDropout(n_perturbations=self.n_perturbations, **shared),
            CIFARSWAG(n_perturbations=self.n_perturbations, **shared),
            CIFARPJSVD(n_directions=self.n_directions,
                       n_perturbations=self.n_perturbations,
                       perturbation_sizes=self.perturbation_sizes, **shared),
            CIFARMLPJSVDv(n_directions=self.n_directions,
                           n_perturbations=self.n_perturbations,
                           perturbation_sizes=ml_ps, **shared),
        ]
