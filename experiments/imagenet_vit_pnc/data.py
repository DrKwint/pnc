"""Real ImageNet-1k validation images for the preflight (ID data only -- no OOD).

Source: ``evanarlian/imagenet_1k_resized_256`` on the HuggingFace Hub, which is the
ILSVRC-2012 validation split with the short side resized to 256. Shard 0 holds
25,000 of the 50,000 validation images with labels. The pinned
``ViT_B_16_Weights.IMAGENET1K_V1`` transform (resize 256 -> center-crop 224 ->
normalize) is applied unchanged, so ``accuracy_gate`` reproducing the published
81.07% top-1 is the end-to-end check that the pipeline is faithful.

Splits are a single deterministic permutation of the shard (seed fixed in
:func:`split_indices`), so calibration and held-out ID sets never overlap and are
reproducible across runs. Spec section 21 forbids OOD data here; nothing in this
module can load any.
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .vit_adapter import WEIGHT_ENUM

REPO_ID = "evanarlian/imagenet_1k_resized_256"
SHARDS = ("data/val-00000-of-00002-b5248be478d25e41.parquet",
          "data/val-00001-of-00002-85f3d9c8fa1edb63.parquet")
SPLIT_SEED = 20260812


def shard_paths(n_shards: int = len(SHARDS)) -> list[str]:
    from huggingface_hub import hf_hub_download
    return [hf_hub_download(REPO_ID, s, repo_type="dataset") for s in SHARDS[:n_shards]]


@dataclass
class Splits:
    calib: np.ndarray
    heldout: np.ndarray
    bench: np.ndarray


def split_indices(n_rows: int, n_calib: int, n_heldout: int, n_bench: int = 512) -> Splits:
    """Disjoint deterministic calibration / held-out-ID / benchmark index sets."""
    perm = np.random.RandomState(SPLIT_SEED).permutation(n_rows)
    need = n_calib + n_heldout + n_bench
    if need > n_rows:
        raise ValueError(f"requested {need} images but shard has {n_rows}")
    return Splits(calib=np.sort(perm[:n_calib]),
                  heldout=np.sort(perm[n_calib:n_calib + n_heldout]),
                  bench=np.sort(perm[n_calib + n_heldout:need]))


class ImageNetVal:
    """Random-access decoder over the parquet shard, with a thread pool for JPEG decode."""

    def __init__(self, paths: list[str] | None = None, workers: int = 8):
        import pyarrow.parquet as pq
        self.paths = paths or shard_paths()
        self.pfs = [pq.ParquetFile(p) for p in self.paths]
        self.shard_rows = [pf.metadata.num_rows for pf in self.pfs]
        self.shard_offset = np.cumsum([0] + self.shard_rows)
        self.n_rows = int(self.shard_offset[-1])
        self.rg_size = self.pfs[0].metadata.row_group(0).num_rows
        self.transform = WEIGHT_ENUM.transforms()
        self.workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache_key = None
        self._cache = None

    def __len__(self) -> int:
        return self.n_rows

    def _locate(self, idx: int) -> tuple[int, int, int]:
        """Global row index -> (shard, row group within shard, offset within group)."""
        s = int(np.searchsorted(self.shard_offset, idx, side="right") - 1)
        local = int(idx) - int(self.shard_offset[s])
        return s, local // self.rg_size, local % self.rg_size

    def _row_group(self, shard: int, rg: int):
        if self._cache_key != (shard, rg):
            tbl = self.pfs[shard].read_row_group(rg, columns=["image", "label"])
            self._cache = (tbl.column("image").to_pylist(), tbl.column("label").to_pylist())
            self._cache_key = (shard, rg)
        return self._cache

    def _decode_bytes(self, blob: bytes) -> torch.Tensor:
        return self.transform(Image.open(io.BytesIO(blob)).convert("RGB"))

    def load(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode + transform the given rows -> (N,3,224,224) float32 CPU, (N,) labels.

        Bytes are fetched in row-group order so the single-row-group cache hits, then
        decoded on the thread pool (PIL releases the GIL). Output follows input order.
        """
        indices = np.asarray(indices)
        raw: list[bytes | None] = [None] * len(indices)
        labels = np.zeros(len(indices), dtype=np.int64)
        for pos in np.argsort(indices, kind="stable"):
            shard, rg, j = self._locate(indices[pos])
            imgs, labs = self._row_group(shard, rg)
            raw[pos] = imgs[j]["bytes"]
            labels[pos] = labs[j]
        return torch.stack(list(self._pool.map(self._decode_bytes, raw))), \
            torch.from_numpy(labels)

    def iter_batches(self, indices, batch_size: int):
        """Yield (images, labels, ids) batches; images stay on CPU until the caller moves them."""
        indices = np.asarray(indices)
        for s in range(0, len(indices), batch_size):
            chunk = indices[s:s + batch_size]
            imgs, labels = self.load(chunk)
            yield imgs, labels, chunk


@torch.inference_mode()
def accuracy_gate(adapter, dataset: ImageNetVal, indices, batch_size: int = 16) -> dict:
    """Top-1/top-5 of the base model -- end-to-end check of checkpoint + preprocessing."""
    top1 = top5 = n = 0
    for imgs, labels, _ in dataset.iter_batches(indices, batch_size):
        logits = adapter.base_forward(imgs.to(adapter.device, adapter.dtype))
        labels = labels.to(logits.device)
        top = logits.topk(5, dim=-1).indices
        top1 += int((top[:, 0] == labels).sum())
        top5 += int((top == labels[:, None]).any(-1).sum())
        n += len(labels)
    return {"n": n, "top1": top1 / n, "top5": top5 / n}
