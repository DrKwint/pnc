import os
import warnings
from typing import Callable, Any
import numpy as np
import gymnasium as gym

def load_neurips_policy(env_name: str, regime: str, strict: bool = True) -> tuple[Callable[[gym.Env, np.ndarray], np.ndarray], dict]:
    """
    Loads a policy for the NeurIPS paper preset.
    Returns:
        policy_function: A callable that takes (env, obs) and returns an action.
        metadata: A dictionary with keys 'policy_id', 'algo_family', 'source'.
    """
    if regime in ("ood_far", "ood"):
        from data import id_policy_random
        return id_policy_random, {
            "policy_id": "pure_random",
            "algo_family": "random",
            "source": "id_policy_random"
        }

    # Map environment and regime to exact farama-minari repository ID
    algo = "SAC"
    if "hopper" in env_name.lower():
        algo = "SAC"
        env_base = "Hopper-v5"
    elif "halfcheetah" in env_name.lower():
        algo = "TQC"
        env_base = "HalfCheetah-v5"
    elif "ant" in env_name.lower():
        algo = "SAC"
        env_base = "Ant-v5"
    elif "humanoid" in env_name.lower():
        algo = "TQC"
        env_base = "Humanoid-v5"
    else:
        if strict:
            raise ValueError(f"Unknown environment for NeurIPS preset: {env_name}")
        env_base = env_name

    # Determine severity suffix
    if regime in ("id", "id_train", "id_eval"):
        level = "expert"
    elif regime == "ood_near":
        level = "medium"
    elif regime == "ood_mid":
        level = "simple"
    else:
        if strict:
            raise ValueError(f"Unknown regime: {regime}")
        level = "expert"

    # Exact Repo ID
    repo_id = f"farama-minari/{env_base}-{algo}-{level}"

    # The farama-minari repos use a per-repo filename convention that mirrors
    # the env / algo / level slug. Casing varies between repos:
    #   hopper-v5-SAC-expert.zip
    #   halfcheetah-v5-TQC-expert.zip
    #   ant-v5-sac-expert.zip   (lowercase sac for Ant expert!)
    #   ant-v5-SAC-medium.zip
    # We list the repo and pick the first top-level .zip file as the artifact.
    print(f"Loading {repo_id} ({algo}) ...")

    # Try Hugging Face download
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list_repo_files(repo_id)
        zip_candidates = [f for f in files if f.endswith(".zip") and "/" not in f]
        if not zip_candidates:
            raise RuntimeError(f"No top-level .zip file found in {repo_id}")
        filename = zip_candidates[0]
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        source = "hf_hub_download"
    except ImportError as e:
        if strict:
            raise RuntimeError("huggingface_hub is required for NeurIPS strict load. Install via `pip install huggingface_hub`.") from e
        else:
            warnings.warn("huggingface_hub not found. Attempting fallback...")
            local_path = None
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to download {repo_id}/{filename}") from e
        else:
            warnings.warn(f"Download failed for {repo_id}. Exception: {e}")
            local_path = None

    # We do not fallback to noisy wrappers in strict mode if local_path is None.
    # In non-strict mode, we can yield a degraded expert wrapper.
    if local_path is None and not strict:
        return _fallback_wrapper(env_name, regime, repo_id, algo)

    # 2. Load the downloaded artifact using SB3 or SB3-Contrib
    # Force CPU device — many older GPUs (e.g. Pascal sm_61) are not supported
    # by the torch versions that ship with current SB3, and we only need
    # SB3 for fast inference here, not training.
    try:
        if algo == "SAC":
            from stable_baselines3 import SAC
            model = SAC.load(local_path, device="cpu", custom_objects={"lr_schedule": lambda _: 0.0})
        elif algo == "TQC":
            from sb3_contrib import TQC
            model = TQC.load(local_path, device="cpu", custom_objects={"lr_schedule": lambda _: 0.0})
        else:
            raise ValueError(f"Unsupported algo family: {algo}")
            
        def policy_fn(env: gym.Env, obs: np.ndarray) -> np.ndarray:
            action, _ = model.predict(obs, deterministic=True)
            return action

        return policy_fn, {
            "policy_id": repo_id,
            "algo_family": algo,
            "source": source
        }
    except ImportError as e:
        if strict:
            raise RuntimeError(f"Missing RL library. SAC requires stable_baselines3, TQC requires sb3_contrib.") from e
        else:
            return _fallback_wrapper(env_name, regime, repo_id, algo)
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to load checkpoint from {local_path} using {algo}.load().") from e
        else:
            return _fallback_wrapper(env_name, regime, repo_id, algo)


def _fallback_wrapper(env_name: str, regime: str, intended_repo: str, algo: str) -> tuple[Callable, dict]:
    """Provides a degraded wrapper fallback if strict=False."""
    print(f"WARNING: Falling back to corrupted expert wrapper for {env_name} {regime}!")
    from data import ExpertPolicyWrapper, GaussianActionNoiseWrapper, ActionDropoutWrapper
    
    if regime in ("id", "id_train", "id_eval"):
        policy = ExpertPolicyWrapper()
    elif regime == "ood_near":
        policy = GaussianActionNoiseWrapper(ExpertPolicyWrapper(), noise_std=0.2)
    elif regime == "ood_mid":
        policy = ActionDropoutWrapper(
            GaussianActionNoiseWrapper(ExpertPolicyWrapper(), noise_std=0.4),
            drop_prob=0.3
        )
    else:
        from data import id_policy_random
        policy = id_policy_random
        
    return policy, {
        "policy_id": intended_repo,
        "algo_family": algo,
        "source": "fallback_wrapper"
    }
