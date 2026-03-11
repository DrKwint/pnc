import os
import sys
import argparse
import time
import json
import pickle
from pathlib import Path

# Parse --cpu early, BEFORE importing jax, so JAX_PLATFORMS is set before JAX initializes any GPU
if '--cpu' in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Belt-and-suspenders: hide GPU from CUDA too
    sys.argv.remove('--cpu')
else:
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

import luigi
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from util import (
    _get_activation, seed_everything, _evaluate_gym, _evaluate_mnist, _evaluate_cifar,
    _load_gym_data, _ps_str, ACTIVATIONS, _find_pjsvd_directions
)
from pjsvd import (
    find_optimal_perturbation, find_optimal_perturbation_multi_layer,
    find_optimal_perturbation_full, find_optimal_perturbation_multi_layer_full,
    find_pjsvd_directions_randomized_svd
)
from models import (TransitionModel, MCDropoutTransitionModel,
                    ClassificationModel, MCDropoutClassificationModel,
                    RegressionModel, MCDropoutRegressionModel,
                    ProbabilisticRegressionModel,
                    ResNet50, MCDropoutResNet50)
from laplace import compute_kfac_factors
from data import (collect_data, id_policy_random, OODPolicyWrapper,
                  load_mnist, load_uci, load_cifar10, load_cifar100)
from training import (train_model, train_swag_model, train_classification_model,
                      train_swag_classification_model, train_probabilistic_model,
                      train_subspace_model, train_subspace_classification_model,
                      train_resnet_model)
from ensembles import (CompactPJSVDEnsemble, CompactMultiLayerPJSVDEnsemble,
                        LeastSquaresCompactPJSVDEnsemble, LeastSquaresCompactMultiLayerPJSVDEnsemble,
                        StandardEnsemble, MCDropoutEnsemble, SWAGEnsemble, LaplaceEnsemble,
                        EnsemblePJSVDHybrid, SubspaceInferenceEnsemble,
                        BatchNormRefitPJSVDEnsemble, MLBatchNormRefitPJSVDEnsemble)
from metrics import compute_nll, compute_calibration, print_metrics, compute_ood_metrics

# ---------------------------------------------------------------------------
# Task modules  (imported so their task classes register with luigi.build)
# ---------------------------------------------------------------------------
from gym_tasks import (
    CollectGymData, GymStandardEnsemble, GymMCDropout, GymSWAG, GymLaplace,
    GymPJSVD, GymMultiLayerPJSVD, GymSubspaceInference, AllGymExperiments,
)
from mnist_tasks import (
    MNISTTrainBaseModel, MNISTStandardEnsemble, MNISTMCDropout, MNISTSwag,
    MNISTLaplace, MNISTPJSVD, MNISTEnsemblePJSVD, MNISTMultiLayerPJSVD,
    AllMNISTExperiments,
)
from cifar_tasks import (
    CIFARTrainBaseModel, CIFARTrainMCDropoutModel, CIFARTrainSWAGModel,
    CIFARStandardEnsemble, CIFARMCDropout, CIFARSWAG,
    CIFARPJSVD, CIFARMLPJSVDv, AllCIFARExperiments,
)
from uci_tasks import (
    UCIStandardEnsemble, UCIMCDropout, UCISWAG, UCILaplace,
    UCIPJSVD, UCIMultiLayerPJSVD, UCISubspaceInference, AllUCIExperiments,
)

# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run PJSVD experiments via Luigi")
    parser.add_argument("--env",               type=str,   default="HalfCheetah-v5",
                        help="Gym environment name, 'MNIST', 'uci-<dataset>', 'cifar10', 'cifar100', or 'bsuite/<id>'")
    parser.add_argument("--steps",             type=int,   default=10000,
                        help="Training steps / env interactions per policy")
    parser.add_argument("--subset_size",       type=int,   default=4096,
                        help="Data subset size for PJSVD null-space search")
    parser.add_argument("--n_directions",      type=int,   default=40,
                        help="Number of singular directions (K)")
    parser.add_argument("--n_perturbations",   type=int,   default=1000,
                        help="Number of ensemble members to generate")
    parser.add_argument("--n_baseline",        type=int,   default=5,
                        help="Number of models for the deep ensemble baseline")
    parser.add_argument("--perturbation_sizes", nargs="+", type=float,
                        default=[20.0, 40.0, 80.0, 160.0, 320.0],
                        help="List of perturbation norms to sweep")
    parser.add_argument("--laplace_priors",    nargs="+",  type=float,
                        default=[0.1, 0.5, 1.0, 5.0, 10.0],
                        help="List of prior precisions for Laplace Approximation")
    parser.add_argument("--seed",              type=int,   default=0,
                        help="Global random seed for reproducibility")
    parser.add_argument("--workers",           type=int,   default=1,
                        help="Number of Luigi workers (parallel tasks)")
    parser.add_argument("--activation",        type=str,   default="relu",
                        choices=list(ACTIVATIONS),
                        help="Activation function for all models (default: relu)")
    parser.add_argument("--task",              type=str,   default=None,
                        help="Specific Luigi task class name to run (e.g., GymPJSVD)")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.task:
        # Resolve the task class by name from all registered modules
        import gym_tasks, mnist_tasks, cifar_tasks, bsuite_experiments
        all_namespaces = [globals(), vars(gym_tasks), vars(mnist_tasks),
                          vars(cifar_tasks), vars(bsuite_experiments)]
        task_cls = None
        for ns in all_namespaces:
            task_cls = ns.get(args.task)
            if task_cls and isinstance(task_cls, type) and issubclass(task_cls, luigi.Task):
                break
        if task_cls is None:
            print(f"Error: Unknown task '{args.task}'")
            sys.exit(1)

        task_params = task_cls.get_param_names()
        task_kwargs = {}
        args_dict = vars(args)
        if "dataset" in task_params and args.env.lower().startswith("uci-"):
            task_kwargs["dataset"] = args.env.lower()[4:]
        if "bsuite_id" in task_params and args.env.lower().startswith("bsuite/"):
            task_kwargs["bsuite_id"] = args.env[7:]
        for param in task_params:
            if param in args_dict:
                task_kwargs[param] = args_dict[param]

        task = task_cls(**task_kwargs)
        luigi.build([task], local_scheduler=True, workers=args.workers)
        return

    env_lower = args.env.lower()

    if env_lower.startswith("uci-"):
        dataset = env_lower[4:]
        task = AllUCIExperiments(
            dataset=dataset, steps=args.steps,
            n_perturbations=args.n_perturbations,
            n_baseline=args.n_baseline,
            laplace_priors=args.laplace_priors,
            seed=args.seed, activation=args.activation,
        )
    elif env_lower == "mnist":
        task = AllMNISTExperiments(
            steps=args.steps,
            n_perturbations=args.n_perturbations,
            n_baseline=args.n_baseline,
            n_directions=args.n_directions,
            perturbation_sizes=args.perturbation_sizes,
            laplace_priors=args.laplace_priors,
            seed=args.seed, activation=args.activation,
        )
    elif env_lower in ("cifar10", "cifar100"):
        task = AllCIFARExperiments(
            dataset=env_lower,
            n_perturbations=args.n_perturbations,
            n_directions=args.n_directions,
            perturbation_sizes=args.perturbation_sizes,
            seed=args.seed,
        )
    else:
        task = AllGymExperiments(
            env=args.env, steps=args.steps,
            subset_size=args.subset_size,
            n_directions=args.n_directions,
            n_perturbations=args.n_perturbations,
            n_baseline=args.n_baseline,
            perturbation_sizes=args.perturbation_sizes,
            laplace_priors=args.laplace_priors,
            seed=args.seed, activation=args.activation,
        )

    luigi.build([task], local_scheduler=True, workers=args.workers)


if __name__ == "__main__":
    main()
