import os
import sys
import argparse
from typing import Any

# Parse --cpu early, BEFORE importing jax, so JAX_PLATFORMS is set before JAX initializes any GPU
if '--cpu' in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Belt-and-suspenders: hide GPU from CUDA too
    sys.argv.remove('--cpu')
else:
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

import luigi

from util import (
    seed_everything, ACTIVATIONS
)

# ---------------------------------------------------------------------------
# Task modules  (imported so their task classes register with luigi.build)
# ---------------------------------------------------------------------------
from gym_tasks import (
    AllGymExperiments,
)
from mnist_tasks import (
    AllMNISTExperiments,
)
from cifar_tasks import (
    AllCIFARExperiments,
)
from uci_tasks import (
    AllUCIExperiments,
)

# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PJSVD experiments via Luigi")
    parser.add_argument("--env",               type=str,   default="HalfCheetah-v5",
                        help="Gym environment name, 'MNIST', 'uci-<dataset>', 'cifar10', 'cifar100'")
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
                        default=[20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0],
                        help="List of perturbation norms to sweep")
    parser.add_argument("--laplace_priors",    nargs="+",  type=float,
                        default=[1.0, 10.0, 100.0, 1000.0, 10000.0],
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
    args, unknown = parser.parse_known_args()

    # Process unknown arguments into a dictionary (e.g., --param value)
    extra_kwargs = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith("--"):
                val = unknown[i+1]
                # Try to cast to int/float/bool if possible
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                else:
                    try:
                        if "." in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except ValueError:
                        pass
                extra_kwargs[key] = val
                i += 2
            else:
                # Boolean flag (True)
                extra_kwargs[key] = True
                i += 1
        else:
            i += 1

    seed_everything(args.seed)

    if args.task:
        # Resolve the task class by name from all registered modules
        import gym_tasks
        import mnist_tasks
        import cifar_tasks
        import uci_tasks
        all_namespaces = [globals(), vars(gym_tasks), vars(mnist_tasks),
                          vars(cifar_tasks), vars(uci_tasks)]
        task_cls = None
        for ns in all_namespaces:
            task_cls = ns.get(args.task)
            if task_cls and isinstance(task_cls, type) and issubclass(task_cls, luigi.Task):
                break
        if task_cls is None:
            print(f"Error: Unknown task '{args.task}'")
            sys.exit(1)

        task_params = task_cls.get_param_names()
        task_kwargs: dict[str, Any] = {}
        args_dict = vars(args)
        
        # Add standard args if they are requested by the task
        if "dataset" in task_params and args.env.lower().startswith("uci-"):
            task_kwargs["dataset"] = args.env.lower()[4:]
        for param in task_params:
            if param in args_dict:
                task_kwargs[param] = args_dict[param]
            if param in extra_kwargs:
                task_kwargs[param] = extra_kwargs[param]

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
