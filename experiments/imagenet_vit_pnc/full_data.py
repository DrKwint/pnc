"""ImageNet-1k train/val access for the full experiment, with exact row addressing.

The preflight's :class:`data.ImageNetVal` assumed uniformly sized parquet row groups.
That holds for the validation shards (100 rows each) but **not** for the training shards
(24,638 rows over 247 groups), so this module builds an explicit cumulative row-group
offset table per shard and addresses rows exactly.

An index over the training shards (label + original ImageNet filename for all 1.28M rows,
no image bytes) is built once and cached, because the class-stratified split needs to know
which rows belong to which class before any image is decoded.
"""
from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .vit_adapter import WEIGHT_ENUM

# Authentic ILSVRC-2012 JPEGs (gated; requires an accepted licence on the HF account).
#
# The preflight used the public `evanarlian/imagenet_1k_resized_256` mirror, which
# re-encodes every image at ~0.26 bytes/pixel with the short side pre-scaled to 256. That
# costs 1.85 pp of top-1 (79.22% vs the published 81.07%) and — worse for an OOD study —
# leaves ID images visibly more compressed than the OOD sets, which are distributed as
# originals. A detector could then separate ID from OOD partly on compression artefacts.
# The full experiment therefore reads the originals (~0.63 bytes/pixel, median short side
# 375) so ID and OOD images pass through one identical pipeline.
REPO_ID = "ILSVRC/imagenet-1k"
N_CLASSES = 1000
N_TRAIN_SHARDS = 28          # shards are shuffled, so ~28 of 294 cover every class deeply


def shard_paths(split: str, limit: int | None = None) -> list[str]:
    """Local paths of the cached parquet shards for 'train' or 'validation'."""
    from huggingface_hub import hf_hub_download, list_repo_files
    prefix = "data/validation" if split in ("val", "validation") else f"data/{split}-"
    files = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} shards in {REPO_ID}")
    if split == "train":
        files = files[:limit or N_TRAIN_SHARDS]
    return [hf_hub_download(REPO_ID, f, repo_type="dataset") for f in files]


class ImageNetShards:
    """Exact random access over a set of parquet shards, with a thread pool for decode."""

    def __init__(self, paths: list[str], workers: int = 8):
        import pyarrow.parquet as pq
        self.paths = list(paths)
        self.pfs = [pq.ParquetFile(p) for p in self.paths]
        # cumulative row offsets: per shard, per row group
        self.rg_offsets, self.shard_offsets, total = [], [0], 0
        for pf in self.pfs:
            offs, n = [0], 0
            for i in range(pf.metadata.num_row_groups):
                n += pf.metadata.row_group(i).num_rows
                offs.append(n)
            self.rg_offsets.append(np.asarray(offs))
            total += n
            self.shard_offsets.append(total)
        self.shard_offsets = np.asarray(self.shard_offsets)
        self.n_rows = int(total)
        self.transform = WEIGHT_ENUM.transforms()
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache_key = None
        self._cache = None

    def __len__(self) -> int:
        return self.n_rows

    def _locate(self, idx: int) -> tuple[int, int, int]:
        """Global row -> (shard, row group, offset within group), exactly."""
        s = int(np.searchsorted(self.shard_offsets, idx, side="right") - 1)
        local = int(idx) - int(self.shard_offsets[s])
        offs = self.rg_offsets[s]
        rg = int(np.searchsorted(offs, local, side="right") - 1)
        return s, rg, local - int(offs[rg])

    def _row_group(self, shard: int, rg: int, with_path: bool = False):
        key = (shard, rg, with_path)
        if self._cache_key != key:
            tbl = self.pfs[shard].read_row_group(rg, columns=["image", "label"])
            imgs = tbl.column("image").to_pylist()
            self._cache = (imgs, tbl.column("label").to_pylist())
            self._cache_key = key
        return self._cache

    def load(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode + transform -> (N,3,224,224) float32 CPU, (N,) int64 labels."""
        indices = np.asarray(indices)
        raw: list[bytes | None] = [None] * len(indices)
        labels = np.zeros(len(indices), dtype=np.int64)
        for pos in np.argsort(indices, kind="stable"):
            shard, rg, j = self._locate(indices[pos])
            imgs, labs = self._row_group(shard, rg)
            raw[pos] = imgs[j]["bytes"]
            labels[pos] = labs[j]
        dec = list(self._pool.map(self._decode, raw))
        return torch.stack(dec), torch.from_numpy(labels)

    def _decode(self, blob: bytes) -> torch.Tensor:
        return self.transform(Image.open(io.BytesIO(blob)).convert("RGB"))

    def iter_batches(self, indices, batch_size: int):
        indices = np.asarray(indices)
        for s in range(0, len(indices), batch_size):
            chunk = indices[s:s + batch_size]
            imgs, labels = self.load(chunk)
            yield imgs, labels, chunk

    # ---- lightweight index (labels + filenames), no image decode ----
    def build_index(self, cache_path: str | Path) -> dict:
        """(labels, paths) for every row. Cached, because it costs a full metadata pass."""
        cache_path = Path(cache_path)
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            return {"labels": z["labels"], "paths": z["paths"]}
        labels, paths = [], []
        for si, pf in enumerate(self.pfs):
            # nested selection: reads only the filename, never the image bytes
            tbl = pf.read(columns=["image.path", "label"])
            labels.extend(tbl.column("label").to_pylist())
            paths.extend(x["path"] for x in tbl.column("image").to_pylist())
            print(f"  indexed shard {si + 1}/{len(self.pfs)} ({len(labels):,} rows)", flush=True)
        labels = np.asarray(labels, dtype=np.int32)
        paths = np.asarray(paths, dtype=object)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, labels=labels, paths=paths)
        return {"labels": labels, "paths": paths}


@torch.inference_mode()
def base_logits(adapter, dataset: ImageNetShards, indices, batch_size: int = 16,
                progress_every: int = 0):
    """Base-model logits for the given rows -> (N, 1000) float32 CPU, (N,) labels."""
    out, labs, done = [], [], 0
    for imgs, labels, _ in dataset.iter_batches(indices, batch_size):
        out.append(adapter.base_forward(imgs.to(adapter.device, adapter.dtype)).cpu())
        labs.append(labels)
        done += len(labels)
        if progress_every and done % progress_every < batch_size:
            print(f"    {done}/{len(indices)}", flush=True)
    return torch.cat(out), torch.cat(labs)
