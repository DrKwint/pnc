import os
from collections.abc import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Minari pre-collected dataset loader (NeurIPS preset)
# ---------------------------------------------------------------------------

# Mapping from our env names to Minari dataset slugs.
_MINARI_ENV_SLUG = {
    "Hopper-v5": "hopper",
    "HalfCheetah-v5": "halfcheetah",
    "Ant-v5": "ant",
    "Humanoid-v5": "humanoid",
}

# Mapping from our regime names to Minari level slugs.
_MINARI_REGIME_LEVEL = {
    "id": "expert",
    "id_train": "expert",
    "id_eval": "expert",
    "ood_near": "medium",
    "ood_mid": "simple",
    # ood_far is *not* in Minari — it's pure-random actions, handled separately.
}


def load_minari_transitions(
    env_name: str,
    regime: str,
    n_steps: int,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, dict]:
    """Load `n_steps` (state, action, next_state) transitions from a Minari dataset.

    Returns (inputs, targets, metadata) where:
        inputs[i]  = concat(state_i, action_i)            shape (n_steps, S+A)
        targets[i] = next_state_i                         shape (n_steps, S)

    For `regime=ood_far`, falls back to running `id_policy_random` against the
    live env (Minari has no random-policy dataset). All other regimes are read
    from `mujoco/{env}/{level}-v0` Minari datasets.

    `seed` controls which slice of episodes is sampled (different `seed`s give
    disjoint sub-samples drawn deterministically from the same dataset). This
    matches the legacy `CollectGymData` behaviour where the seed parameter
    affected the rollout RNG.
    """
    import minari

    # ood_far / ood is uniform random actions — no Minari dataset for this.
    if regime in ("ood_far", "ood"):
        print(f"Collecting {n_steps} steps of uniform-random actions for {env_name} ({regime})...")
        env = gym.make(env_name)
        inputs, targets = [], []
        obs, _ = env.reset(seed=seed)
        while len(inputs) < n_steps:
            action = env.action_space.sample()
            next_obs, _, term, trunc, _ = env.step(action)
            inputs.append(np.concatenate([obs, action]))
            targets.append(next_obs)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        inputs_arr = np.stack(inputs[:n_steps]).astype(np.float32)
        targets_arr = np.stack(targets[:n_steps]).astype(np.float32)
        return jnp.array(inputs_arr), jnp.array(targets_arr), {
            "policy_id": "pure_random",
            "algo_family": "random",
            "source": "id_policy_random",
            "n_steps": int(n_steps),
        }

    if env_name not in _MINARI_ENV_SLUG:
        raise ValueError(f"Unknown environment for Minari preset: {env_name}")
    if regime not in _MINARI_REGIME_LEVEL:
        raise ValueError(f"Unknown regime for Minari preset: {regime}")

    env_slug = _MINARI_ENV_SLUG[env_name]
    level = _MINARI_REGIME_LEVEL[regime]
    ds_id = f"mujoco/{env_slug}/{level}-v0"

    print(f"Loading Minari dataset {ds_id} for {regime} ({n_steps} transitions, seed={seed})...")
    ds = minari.load_dataset(ds_id, download=True)

    # Deterministic episode ordering by seed: shuffle episode IDs with `seed`,
    # then read transitions episode-by-episode until we have enough.
    rng = np.random.default_rng(seed)
    episode_ids = list(range(len(ds)))
    rng.shuffle(episode_ids)

    inputs_blocks = []
    targets_blocks = []
    collected = 0
    for ep_id in episode_ids:
        ep = ds[ep_id][0] if isinstance(ds[ep_id], tuple) else ds[ep_id]
        # Some Minari versions index by id rather than positional; fall back.
        if not hasattr(ep, "observations"):
            ep = next(iter(ds.iterate_episodes(episode_indices=[ep_id])))
        obs = np.asarray(ep.observations, dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.float32)
        # `obs` has length T+1; transitions: (obs[i], act[i]) -> obs[i+1].
        n_t = len(act)
        ep_inputs = np.concatenate([obs[:n_t], act], axis=-1)
        ep_targets = obs[1 : n_t + 1]
        take = min(n_t, n_steps - collected)
        inputs_blocks.append(ep_inputs[:take])
        targets_blocks.append(ep_targets[:take])
        collected += take
        if collected >= n_steps:
            break

    if collected < n_steps:
        print(
            f"WARNING: requested {n_steps} transitions but Minari dataset {ds_id} only "
            f"provided {collected} (will return short array)."
        )

    inputs_arr = np.concatenate(inputs_blocks, axis=0).astype(np.float32)
    targets_arr = np.concatenate(targets_blocks, axis=0).astype(np.float32)

    metadata = {
        "policy_id": ds_id,
        "algo_family": "minari",
        "source": "minari.load_dataset",
        "n_steps": int(len(inputs_arr)),
        "n_episodes_used": len(inputs_blocks),
    }
    return jnp.array(inputs_arr), jnp.array(targets_arr), metadata



def positive_policy(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """A placeholder policy that always takes positive actions."""
    action_dim = env.action_space.shape[0]  # type: ignore
    return np.random.uniform(0.5, 1.0, size=(action_dim,)).astype(np.float32)


def negative_policy(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """A placeholder policy that always takes negative actions."""
    action_dim = env.action_space.shape[0]  # type: ignore
    return np.random.uniform(-1.0, -0.5, size=(action_dim,)).astype(np.float32)


def id_policy_random(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """
    Uniform random action policy.

    NOTE: Despite the historical name, this is used as the **Far OOD** policy
    in the current `get_policy_for_regime` mapping. ID is the structured
    expert policy (sine-wave gait), and OOD progresses as
    expert + noise → expert + noise + dropout → pure random (this function).
    """
    return env.action_space.sample()


def ant_expert_policy(env: gym.Env, obs: np.ndarray, step_count: int) -> np.ndarray:
    """
    ID Policy for Ant-v5: A structured, alternating tetrapod gait.
    Mimics 'routine navigation'.
    State coverage: Coordinated 8-joint periodic motion.

    Action space (8D):
    [hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4]
    """
    freq = 2.0  # Hz
    amplitude = 0.8
    t = step_count * 0.05  # Assuming dt is roughly 0.05

    # Legs 1 & 4 move together, Legs 2 & 3 move exactly out of phase
    # We offset the ankle slightly from the hip for a "stepping" motion
    phase_offset = np.array(
        [
            0,
            -0.5,  # Leg 1 (Front Left)
            np.pi,
            np.pi - 0.5,  # Leg 2 (Front Right)
            np.pi,
            np.pi - 0.5,  # Leg 3 (Back Left)
            0,
            -0.5,  # Leg 4 (Back Right)
        ]
    )

    action = amplitude * np.sin(freq * t + phase_offset)
    return action.astype(np.float32)


def hopper_expert_policy(env: gym.Env, obs: np.ndarray, step_count: int) -> np.ndarray:
    """
    ID Policy for Hopper-v5: A structured, 3-joint jumping motion.
    State coverage: Coordinated extension and contraction.

    Action space (3D): [thigh_joint, leg_joint, foot_joint]
    """
    freq = 2.5  # Hz, slightly faster for jumping
    amplitude = 0.7
    t = step_count * 0.02  # Hopper usually has a smaller dt

    # Create a whip-like motion: thigh moves first, then leg, then foot pushes off
    phase_offset = np.array([0, -np.pi / 4, -np.pi / 2])

    action = amplitude * np.sin(freq * t + phase_offset)
    return action.astype(np.float32)


def halfcheetah_expert_policy(
    env: gym.Env, obs: np.ndarray, step_count: int
) -> np.ndarray:
    """
    Structured sine-wave gait used as the **ID policy** in the current
    `get_policy_for_regime` mapping. The model is trained on transitions
    sampled under this policy and evaluated on shifted regimes that
    add noise / dropout / randomize actions on top of (or in place of) it.

    State coverage: High velocity, coordinated periodic motion.
    """
    # A simple trotting gait for HalfCheetah
    # Joints: [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
    # We oscillate legs in anti-phase to create forward motion

    freq = 2.0  # Hz
    phase_offset = np.array([0, -1, -0.5, np.pi, np.pi - 1, np.pi - 0.5])
    amplitude = 0.8

    t = step_count * 0.05  # Assuming dt=0.05 or similar

    # Generate action: A * sin(wt + phi)
    action = amplitude * np.sin(freq * t + phase_offset)

    return action.astype(np.float32)


def collect_data(
    env_name: str,
    steps: int,
    policy_fn: Callable[[gym.Env, np.ndarray], np.ndarray],
    seed: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Collects transitions from a Gym environment using a given policy."""
    print(f"Collecting {steps} steps ({str(policy_fn)})...")
    env = gym.make(env_name)

    inputs, targets = [], []
    obs, _ = env.reset(seed=seed)

    for _ in range(steps):
        action = policy_fn(env, obs)
        # Fix for Gym API quirks
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        # Coerce shape
        if action.ndim == 0:
            action = action[None]

        next_obs, _, term, trunc, _ = env.step(action)

        inputs.append(np.concatenate([obs, action]))  # State + Action
        targets.append(next_obs)  # Predict Next State

        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

    env.close()
    return jnp.array(np.stack(inputs)), jnp.array(np.stack(targets))


class ExpertPolicyWrapper:
    """A stateful wrapper for expert policies that keeps track of the step count."""

    def __init__(self):
        self.step_count = 0

    def __call__(self, env: gym.Env, obs: np.ndarray) -> np.ndarray:
        if "hopper" in env.spec.id.lower():
            action = hopper_expert_policy(env, obs, self.step_count)
        elif "ant" in env.spec.id.lower():
            action = ant_expert_policy(env, obs, self.step_count)
        elif "halfcheetah" in env.spec.id.lower():
            action = halfcheetah_expert_policy(env, obs, self.step_count)
        else:
            raise NotImplementedError(f"Unknown environment: {env.spec.id}")
        self.step_count += 1
        return action


# For backward compatibility with existing configs/scripts
OODPolicyWrapper = ExpertPolicyWrapper


class GaussianActionNoiseWrapper:
    """Wraps a base policy and injects normal noise to actions."""

    def __init__(
        self,
        base_policy: Callable[[gym.Env, np.ndarray], np.ndarray],
        noise_std: float = 0.1,
    ):
        self.base_policy = base_policy
        self.noise_std = noise_std

    def __call__(self, env: gym.Env, obs: np.ndarray) -> np.ndarray:
        action = self.base_policy(env, obs)
        noise = np.random.normal(scale=self.noise_std, size=action.shape)
        # Assuming typical PyBullet/MuJoCo continuous actions in [-1, 1]
        action = np.clip(action + noise, -env.action_space.high[0], env.action_space.high[0])
        return action.astype(np.float32)


class ActionDropoutWrapper:
    """Wraps a base policy and zeros out individual action dimensions randomly."""

    def __init__(
        self,
        base_policy: Callable[[gym.Env, np.ndarray], np.ndarray],
        drop_prob: float = 0.2,
    ):
        self.base_policy = base_policy
        self.drop_prob = drop_prob

    def __call__(self, env: gym.Env, obs: np.ndarray) -> np.ndarray:
        action = self.base_policy(env, obs)
        mask = np.random.uniform(size=action.shape) > self.drop_prob
        action = action * mask
        return action.astype(np.float32)


def get_policy_for_regime(env_name: str, regime: str, preset: str = "", strict: bool = True) -> tuple[Callable[[gym.Env, np.ndarray], np.ndarray], dict]:
    """
    Returns the appropriate policy or policy wrapper for a given environment and regime severity.
    Regimes: "id", "ood_near", "ood_mid", "ood_far"
    Legacy mapping: "ood" -> mapped to "ood_far"
    Returns: (policy_fn, metadata_dict)
    """
    if preset == "neurips_mujoco_ladder":
        from policy_loader import load_neurips_policy
        return load_neurips_policy(env_name, regime, strict=strict)

    if regime == "id" or regime == "id_train":
        policy = ExpertPolicyWrapper()
    elif regime == "ood_near":
        policy = GaussianActionNoiseWrapper(ExpertPolicyWrapper(), noise_std=0.2)
    elif regime == "ood_mid":
        # Combines high noise and dropouts for stronger degradation
        policy = ActionDropoutWrapper(
            GaussianActionNoiseWrapper(ExpertPolicyWrapper(), noise_std=0.4),
            drop_prob=0.3
        )
    elif regime in ("ood_far", "ood"):
        # The ultimate severity: purely random actions
        policy = id_policy_random
    else:
        raise ValueError(f"Unknown regime: {regime}")

    metadata = {
        "policy_id": "legacy_wrapper",
        "algo_family": "wrapper",
        "source": "local_testing"
    }
    return policy, metadata


def _download_idx(url: str, cache_dir: str) -> np.ndarray:
    """Downloads and parses an IDX file (MNIST binary format). Caches locally."""
    import gzip
    import struct
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if not os.path.exists(fname):
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, fname)
    with gzip.open(fname, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data


def load_mnist(
    cache_dir: str = "/tmp/mnist_cache",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Loads MNIST train and test splits.
    Downloads directly from the Google CDN — no tensorflow or keras needed.
    Returns: (x_train, y_train, x_test, y_test)
    """
    MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    print("Loading MNIST dataset...")

    cache = os.path.join(cache_dir, "mnist")
    x_train_raw = _download_idx(f"{MNIST_BASE}train-images-idx3-ubyte.gz", cache)
    y_train_raw = _download_idx(f"{MNIST_BASE}train-labels-idx1-ubyte.gz", cache)
    x_test_raw = _download_idx(f"{MNIST_BASE}t10k-images-idx3-ubyte.gz", cache)
    y_test_raw = _download_idx(f"{MNIST_BASE}t10k-labels-idx1-ubyte.gz", cache)

    def preprocess_x(x):
        return jnp.array(x.reshape(-1, 784).astype(np.float32) / 255.0)

    def preprocess_y(y):
        return jnp.array(y, dtype=jnp.int32)

    return (
        preprocess_x(x_train_raw),
        preprocess_y(y_train_raw),
        preprocess_x(x_test_raw),
        preprocess_y(y_test_raw),
    )


def load_uci(
    name: str, cache_dir: str = "/tmp/uci_cache", seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Loads a UCI regression dataset.
    Downloads, parses, normalizes features and targets, and splits into 90/10 train/test.
    Returns: (x_train, y_train, x_test, y_test)
    """
    import urllib.request
    import zipfile

    import pandas as pd

    name = name.lower()
    os.makedirs(cache_dir, exist_ok=True)
    uci_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

    print(f"Loading UCI dataset: {name}")

    if name == "boston":
        url = uci_base + "housing/housing.data"
        df = pd.read_fwf(url, header=None)
        data = df.values
        X, Y = data[:, :-1], data[:, -1:]
    elif name == "concrete":
        url = uci_base + "concrete/compressive/Concrete_Data.xls"
        df = pd.read_excel(url)
        data = df.values
        X, Y = data[:, :-1], data[:, -1:]
    elif name == "energy":
        url = uci_base + "00242/ENB2012_data.xlsx"
        df = pd.read_excel(url)
        data = df.values
        X, Y = data[:, :-2], data[:, -2:-1]  # The paper uses the first output (Y1)
    elif name == "power":
        url = uci_base + "00294/CCPP.zip"
        fname = os.path.join(cache_dir, "CCPP.zip")
        if not os.path.exists(fname):
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, fname)
        with zipfile.ZipFile(fname, "r") as z:
            with z.open("CCPP/Folds5x2_pp.xlsx") as f:
                df = pd.read_excel(f)
        data = df.values
        X, Y = data[:, :-1], data[:, -1:]
    elif name == "yacht":
        url = uci_base + "00243/yacht_hydrodynamics.data"
        df = pd.read_fwf(url, header=None)
        data = df.values[:-1, :]  # usually drops the last empty row
        X, Y = data[:, :-1], data[:, -1:]
    elif name == "protein":
        url = uci_base + "00265/CASP.csv"
        df = pd.read_csv(url)
        data = df.values
        # First col is RMSD (Target), remaining 9 are features
        X, Y = data[:, 1:], data[:, 0:1]
    else:
        raise ValueError(f"Unknown UCI dataset {name}")

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    N = X.shape[0]
    ind = np.arange(N)

    # We follow the reference repo's exact seeding
    rng = np.random.RandomState(seed)
    rng.shuffle(ind)

    n_train = int(N * 0.9)
    train_idx = ind[:n_train]
    test_idx = ind[n_train:]

    x_train, y_train = X[train_idx], Y[train_idx]
    x_test, y_test = X[test_idx], Y[test_idx]

    # Normalize with train stats
    x_mean = np.mean(x_train, axis=0, keepdims=True)
    x_std = np.std(x_train, axis=0, keepdims=True) + 1e-6
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    y_std = np.std(y_train, axis=0, keepdims=True) + 1e-6

    x_train = (x_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    x_test = (x_test - x_mean) / x_std
    y_test = (y_test - y_mean) / y_std

    return jnp.array(x_train), jnp.array(y_train), jnp.array(x_test), jnp.array(y_test)


# ---------------------------------------------------------------------------
# CIFAR-10 / CIFAR-100
# ---------------------------------------------------------------------------


def _cifar_normalize(x: np.ndarray) -> np.ndarray:
    """Per-channel normalization with CIFAR mean/std (standard values)."""
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    return (x / 255.0 - mean) / std


def load_cifar10(
    cache_dir: str = "/tmp/cifar_cache", normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads CIFAR-10 train and test splits.
    Downloads from the official Toronto CDN if not cached.

    Returns: (x_train, y_train, x_test, y_test)
        x_*: np.ndarray of shape (N, 32, 32, 3), float32, NHWC layout.
        y_*: np.ndarray of shape (N,), int32.
    """
    import tarfile
    import urllib.request

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    dname = "cifar-10-batches-bin"
    os.makedirs(cache_dir, exist_ok=True)
    tar_path = os.path.join(cache_dir, "cifar-10-binary.tar.gz")
    data_dir = os.path.join(cache_dir, dname)

    if not os.path.isdir(data_dir):
        if not os.path.exists(tar_path):
            print(f"Downloading CIFAR-10 from {url} ...")
            urllib.request.urlretrieve(url, tar_path)
        print("Extracting CIFAR-10 ...")
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(cache_dir)

    def _read_bin(filenames):
        xs, ys = [], []
        for fname in filenames:
            path = os.path.join(data_dir, fname)
            with open(path, "rb") as f:
                raw = np.frombuffer(f.read(), dtype=np.uint8)
            # Each record: 1 label byte + 3072 pixel bytes (3 × 32 × 32 CHW)
            raw = raw.reshape(-1, 1 + 3 * 32 * 32)
            ys.append(raw[:, 0].astype(np.int32))
            imgs = raw[:, 1:].reshape(-1, 3, 32, 32)  # NCHW
            imgs = imgs.transpose(0, 2, 3, 1)  # → NHWC
            xs.append(imgs.astype(np.float32))
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    train_files = [f"data_batch_{i}.bin" for i in range(1, 6)]
    x_train, y_train = _read_bin(train_files)
    x_test, y_test = _read_bin(["test_batch.bin"])

    if normalize:
        x_train = _cifar_normalize(x_train)
        x_test = _cifar_normalize(x_test)

    print(f"CIFAR-10 loaded: train={x_train.shape}, test={x_test.shape}")
    return x_train, y_train.astype(np.int32), x_test, y_test.astype(np.int32)


def load_cifar100(
    cache_dir: str = "/tmp/cifar_cache", normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads CIFAR-100 train and test splits.
    Downloads from the official Toronto CDN if not cached.

    Returns: (x_train, y_train, x_test, y_test)
        x_*: np.ndarray of shape (N, 32, 32, 3), float32, NHWC layout.
        y_*: np.ndarray of shape (N,), int32  (fine labels, 100 classes).
    """
    import tarfile
    import urllib.request

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    dname = "cifar-100-binary"
    os.makedirs(cache_dir, exist_ok=True)
    tar_path = os.path.join(cache_dir, "cifar-100-binary.tar.gz")
    data_dir = os.path.join(cache_dir, dname)

    if not os.path.isdir(data_dir):
        if not os.path.exists(tar_path):
            print(f"Downloading CIFAR-100 from {url} ...")
            urllib.request.urlretrieve(url, tar_path)
        print("Extracting CIFAR-100 ...")
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(cache_dir)

    def _read_bin(filename):
        path = os.path.join(data_dir, filename)
        with open(path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        # Each record: 1 coarse label + 1 fine label + 3072 pixel bytes
        raw = raw.reshape(-1, 2 + 3 * 32 * 32)
        ys = raw[:, 1].astype(np.int32)  # fine labels
        imgs = raw[:, 2:].reshape(-1, 3, 32, 32)  # NCHW
        imgs = imgs.transpose(0, 2, 3, 1)  # → NHWC
        return imgs.astype(np.float32), ys

    x_train, y_train = _read_bin("train.bin")
    x_test, y_test = _read_bin("test.bin")

    if normalize:
        x_train = _cifar_normalize(x_train)
        x_test = _cifar_normalize(x_test)

    print(f"CIFAR-100 loaded: train={x_train.shape}, test={x_test.shape}")
    return x_train, y_train.astype(np.int32), x_test, y_test.astype(np.int32)


OPENOOD_CIFAR_BENCHMARKS = {
    "cifar10": {
        "near_ood": {
            "cifar100": ("cifar100", "CIFAR-100"),
            "tiny_imagenet": ("tiny_imagenet", "Tiny ImageNet-200"),
        },
        "far_ood": {
            "mnist": ("mnist", "MNIST"),
            "svhn": ("svhn", "SVHN"),
            "textures": ("textures", "Textures"),
            "places365": ("places365", "Places365"),
        },
    },
    "cifar100": {
        "near_ood": {
            "cifar10": ("cifar10", "CIFAR-10"),
            "tiny_imagenet": ("tiny_imagenet", "Tiny ImageNet-200"),
        },
        "far_ood": {
            "mnist": ("mnist", "MNIST"),
            "svhn": ("svhn", "SVHN"),
            "textures": ("textures", "Textures"),
            "places365": ("places365", "Places365"),
        },
    },
}

_OPENOOD_ALIAS_GROUPS = {
    "cifar10": ["cifar10", "cifar-10", "CIFAR10", "CIFAR-10"],
    "cifar100": ["cifar100", "cifar-100", "CIFAR100", "CIFAR-100"],
    "tiny_imagenet": ["tiny_imagenet", "tiny-imagenet", "tinyimagenet", "TinyImageNet", "tin"],
    "mnist": ["mnist", "MNIST"],
    "svhn": ["svhn", "SVHN"],
    "textures": ["textures", "texture", "Textures", "Texture"],
    "places365": ["places365", "places", "Places365", "Places"],
}


def _openood_npz_array(obj, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in obj:
            return np.asarray(obj[key])
    return None


def _openood_load_npz(path: str, normalize: bool, max_examples: int | None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        images = _openood_npz_array(data, ("images", "x", "inputs"))
        labels = _openood_npz_array(data, ("labels", "y", "targets"))
    if images is None:
        raise ValueError(f"OpenOOD NPZ at {path} must contain images/x/inputs")
    if labels is None:
        labels = np.full((len(images),), -1, dtype=np.int32)
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = np.repeat(images[..., None], 3, axis=-1)
    if images.ndim != 4:
        raise ValueError(f"Expected image batch with 4 dims at {path}, got {images.shape}")
    if images.shape[1:3] != (32, 32):
        raise ValueError(f"Expected 32x32 OpenOOD images at {path}, got {images.shape[1:3]}")
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    if max_examples is not None:
        images = images[:max_examples]
        labels = labels[:max_examples]
    if normalize:
        images = _cifar_normalize(images)
    return images.astype(np.float32), np.asarray(labels, dtype=np.int32)


def _openood_load_from_imglist(
    imglist_path: str,
    root_dir: str,
    normalize: bool,
    max_examples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    images = []
    labels = []
    with open(imglist_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            rel_path = parts[0]
            label = int(parts[1]) if len(parts) > 1 else -1
            candidate_paths = [
                os.path.join(root_dir, rel_path),
                os.path.join(root_dir, "images_classic", rel_path),
            ]
            img_path = None
            for candidate in candidate_paths:
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                raise FileNotFoundError(f"Could not resolve OpenOOD image {rel_path} from {imglist_path}")

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if img.size != (32, 32):
                    img = img.resize((32, 32), Image.BILINEAR)
                images.append(np.asarray(img, dtype=np.float32))
            labels.append(label)
            if max_examples is not None and len(images) >= max_examples:
                break

    if not images:
        raise ValueError(f"No OpenOOD images found via imglist {imglist_path}")
    x = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int32)
    if normalize:
        x = _cifar_normalize(x)
    return x.astype(np.float32), y


def _openood_find_dataset_source(
    root_dir: str,
    id_dataset: str,
    family: str,
    dataset_key: str,
) -> tuple[str, str]:
    aliases = _OPENOOD_ALIAS_GROUPS[dataset_key]
    npz_candidates = []
    imglist_candidates = []
    for alias in aliases:
        npz_candidates.extend(
            [
                os.path.join(root_dir, id_dataset, family, f"{alias}.npz"),
                os.path.join(root_dir, id_dataset, family, alias, "test.npz"),
                os.path.join(root_dir, f"{id_dataset}_{family}_{alias}.npz"),
            ]
        )
        imglist_candidates.extend(
            [
                os.path.join(root_dir, "benchmark_imglist", id_dataset, f"test_{alias}.txt"),
                os.path.join(root_dir, "benchmark_imglist", id_dataset, family, f"{alias}.txt"),
            ]
        )

    for path in npz_candidates:
        if os.path.isfile(path):
            return "npz", path
    for path in imglist_candidates:
        if os.path.isfile(path):
            return "imglist", path
    raise FileNotFoundError(
        f"Could not find OpenOOD source for {id_dataset}/{family}/{dataset_key} under {root_dir}"
    )


def load_openood_cifar_benchmark(
    id_dataset: str,
    root_dir: str = "openood_data",
    normalize: bool = True,
    max_examples_per_dataset: int | None = None,
) -> dict[str, object]:
    """Load OpenOOD v1.5 CIFAR benchmark splits without exposing any OOD validation split."""
    id_key = id_dataset.lower()
    if id_key not in OPENOOD_CIFAR_BENCHMARKS:
        raise ValueError(f"Unsupported OpenOOD CIFAR ID dataset: {id_dataset}")

    if id_key == "cifar10":
        x_train, y_train, x_test, y_test = load_cifar10(normalize=normalize)
    else:
        x_train, y_train, x_test, y_test = load_cifar100(normalize=normalize)

    if max_examples_per_dataset is not None:
        x_test = x_test[:max_examples_per_dataset]
        y_test = y_test[:max_examples_per_dataset]

    benchmark = {
        "id_dataset": id_key,
        "id_train": {"name": id_key, "inputs": x_train, "targets": y_train},
        "id_test": {"name": id_key, "inputs": x_test, "targets": y_test},
        "near_ood": {},
        "far_ood": {},
        "metadata": {
            "benchmark_name": "openood_v1_5",
            "root_dir": root_dir,
            "id_dataset": id_key,
            "normalize": bool(normalize),
            "max_examples_per_dataset": None if max_examples_per_dataset is None else int(max_examples_per_dataset),
            "uses_ood_validation": False,
            "uses_ood_model_selection": False,
        },
    }

    for family in ("near_ood", "far_ood"):
        for dataset_key, (_, display_name) in OPENOOD_CIFAR_BENCHMARKS[id_key][family].items():
            source_type, source_path = _openood_find_dataset_source(root_dir, id_key, family, dataset_key)
            if source_type == "npz":
                inputs, targets = _openood_load_npz(source_path, normalize=normalize, max_examples=max_examples_per_dataset)
            else:
                inputs, targets = _openood_load_from_imglist(
                    source_path,
                    root_dir,
                    normalize=normalize,
                    max_examples=max_examples_per_dataset,
                )
            benchmark[family][dataset_key] = {
                "name": display_name,
                "inputs": inputs,
                "targets": targets,
            }

    return benchmark
