"""Deterministic class-stratified disjoint pools from ImageNet-1k **training** data (spec §4).

The preflight used validation images for feasibility only. Every construction, selection and
calibration operation in the full experiment draws from the training split, leaving the
official 50,000-image validation set completely untouched until final evaluation.

Three pools, pairwise disjoint:

    correction   32,768   fit the corrected W2
    selection     8,192   ID-only hyperparameter search
    temperature   8,192   scalar temperature fit

Stratification is exact rather than approximate: each class contributes
``floor(n/1000)`` rows to a pool, and the remainder is distributed over a seeded
permutation of the class ids so no systematic bias toward low class indices arises.
Within a class the row order is a per-class seeded permutation, and the three pools are
carved from disjoint segments of it.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT_SEED = 20260813
POOLS = {"correction": 32768, "selection": 8192, "temperature": 8192}
N_CLASSES = 1000


def _alloc_per_class(total: int, seed: int) -> np.ndarray:
    """How many rows each class contributes so the pool totals exactly `total`."""
    base, rem = divmod(total, N_CLASSES)
    counts = np.full(N_CLASSES, base, dtype=np.int64)
    if rem:
        extra = np.random.RandomState(seed).permutation(N_CLASSES)[:rem]
        counts[extra] += 1
    return counts


def build_splits(labels: np.ndarray, seed: int = SPLIT_SEED) -> dict[str, np.ndarray]:
    """Return {pool: sorted global row indices}. Pools are disjoint and class-stratified."""
    by_class = {c: np.flatnonzero(labels == c) for c in range(N_CLASSES)}
    short = [c for c, idx in by_class.items() if len(idx) < 50]
    if short:
        raise ValueError(f"{len(short)} classes have <50 training rows: {short[:5]}")

    alloc = {name: _alloc_per_class(n, seed + i)
             for i, (name, n) in enumerate(sorted(POOLS.items()))}
    picked = {name: [] for name in POOLS}
    for c in range(N_CLASSES):
        rows = by_class[c]
        perm = np.random.RandomState((seed * 1000 + c) % (2 ** 32 - 1)).permutation(len(rows))
        cursor = 0
        for name in sorted(POOLS):                       # deterministic pool order
            k = int(alloc[name][c])
            picked[name].append(rows[perm[cursor:cursor + k]])
            cursor += k
    out = {name: np.sort(np.concatenate(v)) for name, v in picked.items()}

    for name, idx in out.items():
        assert len(idx) == POOLS[name], (name, len(idx), POOLS[name])
        assert len(np.unique(idx)) == len(idx), f"{name} has duplicates"
    names = sorted(out)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = np.intersect1d(out[a], out[b])
            assert overlap.size == 0, f"{a} and {b} overlap in {overlap.size} rows"
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: Path, indices: np.ndarray, labels: np.ndarray,
                   paths: np.ndarray, seed: int) -> dict:
    """One CSV row per image: global row, original ImageNet filename, class, seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["global_row", "imagenet_path", "class", "split_seed"])
        for i in indices:
            w.writerow([int(i), str(paths[i]), int(labels[i]), seed])
    cls, counts = np.unique(labels[indices], return_counts=True)
    return {"file": path.name, "n_images": int(len(indices)),
            "n_classes": int(len(cls)),
            "per_class_min": int(counts.min()), "per_class_max": int(counts.max()),
            "sha256": sha256_file(path), "split_seed": seed}


def main():
    import argparse
    from .full_data import ImageNetShards, shard_paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/neurips_2026_rebuttal/imagenet_vit")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = ap.parse_args()
    out = Path(args.out)

    print("locating training shards ...", flush=True)
    tr = ImageNetShards(shard_paths("train"))
    print(f"  {len(tr.paths)} shards, {len(tr):,} training rows")
    idx = tr.build_index(out / "raw" / "train_index.npz")
    labels, paths = idx["labels"], idx["paths"]
    print(f"  index: {len(labels):,} rows, {len(np.unique(labels))} classes")

    splits = build_splits(labels, args.seed)
    summary = {"split_seed": args.seed, "source_repo": "evanarlian/imagenet_1k_resized_256",
               "source_split": "train", "n_train_rows": int(len(labels)), "pools": {}}
    for name, ind in sorted(splits.items()):
        man = write_manifest(out / "splits" / f"{name}_{len(ind)}.csv", ind, labels, paths,
                             args.seed)
        summary["pools"][name] = man
        print(f"  {name:<12} {man['n_images']:>6} images, {man['n_classes']} classes, "
              f"{man['per_class_min']}-{man['per_class_max']}/class, "
              f"sha256 {man['sha256'][:16]}...")
        np.save(out / "splits" / f"{name}_rows.npy", ind)

    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "splits" / "splits_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'splits'}")


if __name__ == "__main__":
    main()
