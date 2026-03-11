import os
from collections.abc import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np


def positive_policy(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """A placeholder policy that always takes positive actions."""
    action_dim = env.action_space.shape[0] # type: ignore
    return np.random.uniform(0.5, 1.0, size=(action_dim,)).astype(np.float32)

def negative_policy(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """A placeholder policy that always takes negative actions."""
    action_dim = env.action_space.shape[0] # type: ignore
    return np.random.uniform(-1.0, -0.5, size=(action_dim,)).astype(np.float32)

def id_policy_random(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """
    ID Policy: Pure Random Exploration.
    Mimics 'training data' in offline RL. 
    State coverage: Low velocity, chaotic poses, falling over.
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
    phase_offset = np.array([
        0, -0.5,           # Leg 1 (Front Left)
        np.pi, np.pi-0.5,  # Leg 2 (Front Right)
        np.pi, np.pi-0.5,  # Leg 3 (Back Left)
        0, -0.5            # Leg 4 (Back Right)
    ])

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
    phase_offset = np.array([0, -np.pi/4, -np.pi/2])

    action = amplitude * np.sin(freq * t + phase_offset)
    return action.astype(np.float32)

def halfcheetah_expert_policy(env: gym.Env, obs: np.ndarray, step_count: int) -> np.ndarray:
    """
    OOD Policy: A structured Sine-Wave Gait.
    Mimics an 'expert' or 'deployment' policy.
    State coverage: High velocity, coordinated periodic motion.
    
    This is OOD because the model has never seen 'coordinated running' 
    dynamics, only 'random flailing' dynamics.
    """
    # A simple trotting gait for HalfCheetah
    # Joints: [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
    # We oscillate legs in anti-phase to create forward motion

    freq = 2.0  # Hz
    phase_offset = np.array([0, -1, -0.5, np.pi, np.pi-1, np.pi-0.5])
    amplitude = 0.8

    t = step_count * 0.05 # Assuming dt=0.05 or similar

    # Generate action: A * sin(wt + phi)
    action = amplitude * np.sin(freq * t + phase_offset)

    return action.astype(np.float32)

def collect_data(
    env_name: str,
    steps: int,
    policy_fn: Callable[[gym.Env, np.ndarray], np.ndarray],
    seed: int = 0
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
        if action.ndim == 0:
            action = action[None]

        next_obs, _, term, trunc, _ = env.step(action)

        inputs.append(np.concatenate([obs, action])) # State + Action
        targets.append(next_obs) # Predict Next State

        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

    env.close()
    return jnp.array(np.stack(inputs)), jnp.array(np.stack(targets))

class OODPolicyWrapper:
    """A stateful wrapper for half_cheetah_expert_policy that keeps track of the step count."""
    def __init__(self):
        self.step_count = 0

    def __call__(self, env: gym.Env, obs: np.ndarray) -> np.ndarray:
        if 'hopper' in env.spec.id.lower():
            action = hopper_expert_policy(env, obs, self.step_count)
        elif 'ant' in env.spec.id.lower():
            action = ant_expert_policy(env, obs, self.step_count)
        else:
            action = halfcheetah_expert_policy(env, obs, self.step_count)
        self.step_count += 1
        return action

def _download_idx(url: str, cache_dir: str) -> np.ndarray:
    """Downloads and parses an IDX file (MNIST binary format). Caches locally."""
    import gzip
    import struct
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if not os.path.exists(fname):
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, fname)
    with gzip.open(fname, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        ndim = magic & 0xFF
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def load_mnist(cache_dir: str = '/tmp/mnist_cache') -> tuple:
    """
    Loads MNIST train and test splits.
    Downloads directly from the Google CDN — no tensorflow or keras needed.
    Returns: (x_train, y_train, x_test, y_test)
    """
    MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    print("Loading MNIST dataset...")

    cache = os.path.join(cache_dir, 'mnist')
    x_train_raw = _download_idx(f"{MNIST_BASE}train-images-idx3-ubyte.gz", cache)
    y_train_raw = _download_idx(f"{MNIST_BASE}train-labels-idx1-ubyte.gz", cache)
    x_test_raw  = _download_idx(f"{MNIST_BASE}t10k-images-idx3-ubyte.gz",  cache)
    y_test_raw  = _download_idx(f"{MNIST_BASE}t10k-labels-idx1-ubyte.gz",  cache)

    def preprocess_x(x):
        return jnp.array(x.reshape(-1, 784).astype(np.float32) / 255.0)
    def preprocess_y(y):
        return jnp.array(y, dtype=jnp.int32)

    return preprocess_x(x_train_raw), preprocess_y(y_train_raw), preprocess_x(x_test_raw), preprocess_y(y_test_raw)

def load_uci(name: str, cache_dir: str = '/tmp/uci_cache', seed: int = 0) -> tuple:
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
    uci_base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'

    print(f"Loading UCI dataset: {name}")

    if name == 'boston':
        url = uci_base + 'housing/housing.data'
        df = pd.read_fwf(url, header=None)
        data = df.values
        X, Y = data[:, :-1], data[:, -1:]
    elif name == 'concrete':
        url = uci_base + 'concrete/compressive/Concrete_Data.xls'
        df = pd.read_excel(url)
        data = df.values
        X, Y = data[:, :-1], data[:, -1:]
    elif name == 'energy':
        url = uci_base + '00242/ENB2012_data.xlsx'
        df = pd.read_excel(url)
        data = df.values
        X, Y = data[:, :-2], data[:, -2:-1]  # The paper uses the first output (Y1)
    elif name == 'power':
        url = uci_base + '00294/CCPP.zip'
        fname = os.path.join(cache_dir, 'CCPP.zip')
        if not os.path.exists(fname):
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, fname)
        with zipfile.ZipFile(fname, 'r') as z:
            with z.open('CCPP/Folds5x2_pp.xlsx') as f:
                df = pd.read_excel(f)
        data = df.values
        X, Y = data[:, :-1], data[:, -1:]
    elif name == 'yacht':
        url = uci_base + '00243/yacht_hydrodynamics.data'
        df = pd.read_fwf(url, header=None)
        data = df.values[:-1, :]  # usually drops the last empty row
        X, Y = data[:, :-1], data[:, -1:]
    elif name == 'protein':
         url = uci_base + '00265/CASP.csv'
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
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    return (x / 255.0 - mean) / std


def load_cifar10(cache_dir: str = '/tmp/cifar_cache', normalize: bool = True) -> tuple:
    """
    Loads CIFAR-10 train and test splits.
    Downloads from the official Toronto CDN if not cached.

    Returns: (x_train, y_train, x_test, y_test)
        x_*: np.ndarray of shape (N, 32, 32, 3), float32, NHWC layout.
        y_*: np.ndarray of shape (N,), int32.
    """
    import tarfile
    import urllib.request

    url   = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    dname = 'cifar-10-batches-bin'
    os.makedirs(cache_dir, exist_ok=True)
    tar_path = os.path.join(cache_dir, 'cifar-10-binary.tar.gz')
    data_dir = os.path.join(cache_dir, dname)

    if not os.path.isdir(data_dir):
        if not os.path.exists(tar_path):
            print(f'Downloading CIFAR-10 from {url} ...')
            urllib.request.urlretrieve(url, tar_path)
        print('Extracting CIFAR-10 ...')
        with tarfile.open(tar_path, 'r:gz') as t:
            t.extractall(cache_dir)

    def _read_bin(filenames):
        xs, ys = [], []
        for fname in filenames:
            path = os.path.join(data_dir, fname)
            with open(path, 'rb') as f:
                raw = np.frombuffer(f.read(), dtype=np.uint8)
            # Each record: 1 label byte + 3072 pixel bytes (3 × 32 × 32 CHW)
            raw = raw.reshape(-1, 1 + 3 * 32 * 32)
            ys.append(raw[:, 0].astype(np.int32))
            imgs = raw[:, 1:].reshape(-1, 3, 32, 32)      # NCHW
            imgs = imgs.transpose(0, 2, 3, 1)              # → NHWC
            xs.append(imgs.astype(np.float32))
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    train_files = [f'data_batch_{i}.bin' for i in range(1, 6)]
    x_train, y_train = _read_bin(train_files)
    x_test,  y_test  = _read_bin(['test_batch.bin'])

    if normalize:
        x_train = _cifar_normalize(x_train)
        x_test  = _cifar_normalize(x_test)

    print(f'CIFAR-10 loaded: train={x_train.shape}, test={x_test.shape}')
    return x_train, y_train.astype(np.int32), \
           x_test,  y_test.astype(np.int32)


def load_cifar100(cache_dir: str = '/tmp/cifar_cache', normalize: bool = True) -> tuple:
    """
    Loads CIFAR-100 train and test splits.
    Downloads from the official Toronto CDN if not cached.

    Returns: (x_train, y_train, x_test, y_test)
        x_*: np.ndarray of shape (N, 32, 32, 3), float32, NHWC layout.
        y_*: np.ndarray of shape (N,), int32  (fine labels, 100 classes).
    """
    import tarfile
    import urllib.request

    url      = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
    dname    = 'cifar-100-binary'
    os.makedirs(cache_dir, exist_ok=True)
    tar_path = os.path.join(cache_dir, 'cifar-100-binary.tar.gz')
    data_dir = os.path.join(cache_dir, dname)

    if not os.path.isdir(data_dir):
        if not os.path.exists(tar_path):
            print(f'Downloading CIFAR-100 from {url} ...')
            urllib.request.urlretrieve(url, tar_path)
        print('Extracting CIFAR-100 ...')
        with tarfile.open(tar_path, 'r:gz') as t:
            t.extractall(cache_dir)

    def _read_bin(filename):
        path = os.path.join(data_dir, filename)
        with open(path, 'rb') as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        # Each record: 1 coarse label + 1 fine label + 3072 pixel bytes
        raw  = raw.reshape(-1, 2 + 3 * 32 * 32)
        ys   = raw[:, 1].astype(np.int32)   # fine labels
        imgs = raw[:, 2:].reshape(-1, 3, 32, 32)  # NCHW
        imgs = imgs.transpose(0, 2, 3, 1)          # → NHWC
        return imgs.astype(np.float32), ys

    x_train, y_train = _read_bin('train.bin')
    x_test,  y_test  = _read_bin('test.bin')

    if normalize:
        x_train = _cifar_normalize(x_train)
        x_test  = _cifar_normalize(x_test)

    print(f'CIFAR-100 loaded: train={x_train.shape}, test={x_test.shape}')
    return x_train, y_train.astype(np.int32), \
           x_test,  y_test.astype(np.int32)

