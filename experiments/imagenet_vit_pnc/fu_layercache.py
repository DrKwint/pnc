"""Spec §20 — build the missing training-pool multi-layer CLS cache, efficiently.

The previous attempt stalled because `ImageNetShards.load` sorts indices only *within* a
16-row batch, while `correction_rows.npy` is ordered by class. Consecutive rows therefore
land in different parquet row groups, and the single-entry row-group cache re-read a whole
row group per image: throughput collapsed to ~1% GPU utilisation.

Here every requested row is located once, all rows are grouped by (shard, row group), each
row group is read exactly once, and the ViT runs on contiguous batches. Results are written
back into the original correction-pool order, so the cache is row-for-row comparable with
every other artifact built from `correction_rows.npy`. The pool membership is unchanged.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch

from . import fu_common as F
from .geom_cache import BLOCKS, cls_by_block
from .memprobe import GIB, reset_cuda
from .vit_adapter import ViTPnCAdapter

BATCH = 32


def run(n_rows: int | None = None):
    from .full_data import ImageNetShards, shard_paths

    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    rows = np.load(F.SRC / "splits" / "correction_rows.npy")
    if n_rows:
        rows = rows[:n_rows]
    ds = ImageNetShards(shard_paths("train"))
    print(f"correction pool: {len(rows):,} rows over {len(ds.paths)} shards")

    # ---- locate every row once, then group by physical row group ----
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for pos, r in enumerate(rows):
        shard, rg, j = ds._locate(int(r))
        groups[(shard, rg)].append((j, pos))
    order = sorted(groups)
    print(f"  {len(order):,} distinct row groups "
          f"({len(rows)/len(order):.1f} images per group read)")

    D = 768
    out = {f"block{i}": np.zeros((len(rows), D), np.float32) for i in BLOCKS}
    out["final_ln"] = np.zeros((len(rows), D), np.float32)
    labels = np.zeros(len(rows), np.int64)

    reset_cuda()
    t0, done = time.perf_counter(), 0
    buf_img, buf_pos = [], []

    def flush():
        nonlocal buf_img, buf_pos, done
        if not buf_img:
            return
        imgs = torch.stack(buf_img).to(ad.device, ad.dtype)
        d = cls_by_block(ad, imgs)
        idx = np.asarray(buf_pos)
        for i in BLOCKS:
            out[f"block{i}"][idx] = d[f"cls_block{i}"].numpy()
        out["final_ln"][idx] = d["cls_final_ln"].numpy()
        done += len(idx)
        buf_img, buf_pos = [], []
        del imgs, d

    for gi, key in enumerate(order):
        shard, rg = key
        imgs_raw, labs = ds._row_group(shard, rg)          # one read per row group
        members = groups[key]
        blobs = [imgs_raw[j]["bytes"] for j, _ in members]
        decoded = list(ds._pool.map(ds._decode, blobs))
        for (j, pos), img in zip(members, decoded):
            labels[pos] = labs[j]
            buf_img.append(img)
            buf_pos.append(pos)
            if len(buf_img) == BATCH:
                flush()
        del imgs_raw, labs, blobs, decoded
        if gi % 50 == 0 and done:
            el = time.perf_counter() - t0
            print(f"    group {gi:>5,}/{len(order):,}  {done:>6,} images  "
                  f"{done/el:.0f} img/s  eta {(len(rows)-done)/(done/el)/60:.1f} min",
                  flush=True)
    flush()
    wall = time.perf_counter() - t0

    ref = np.load(F.SRC / "raw" / "cache_correction.npz")["labels"][:len(rows)]
    assert np.array_equal(labels, ref), "row ordering does not match the frozen pool"
    with torch.inference_mode():
        phi = torch.as_tensor(out["final_ln"], device=ad.device, dtype=ad.dtype)
        pred = torch.cat([ad.head(phi[s:s + 8192]).argmax(-1).cpu()
                          for s in range(0, len(phi), 8192)]).numpy()
    del phi
    reset_cuda()
    print(f"  label check OK; base top-1 on pool {100*(pred == labels).mean():.4f}%")

    path = F.OUT / "raw" / "cls_layers_correction.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out, labels=labels, predicted_labels=pred.astype(np.int64))
    st = {"n": int(len(rows)), "seconds": wall, "images_per_s": len(rows) / wall,
          "n_row_groups": len(order), "batch": BATCH,
          "disk_mib": path.stat().st_size / 1024 ** 2,
          "peak_gpu_gib": torch.cuda.max_memory_allocated() / GIB,
          "base_top1_on_pool": float((pred == labels).mean())}
    F.write_json(F.OUT / "metrics" / "correction_layer_cache.json", st)
    print(f"\n  {len(rows):,} images in {wall/60:.1f} min ({len(rows)/wall:.0f} img/s), "
          f"{st['disk_mib']:.0f} MiB\nwrote {path}")
    return st


if __name__ == "__main__":
    run()
