"""Multi-layer CLS representation cache for the geometry diagnostic (spec A1, A8).

One backbone pass per image set records the CLS token after encoder blocks 8, 9, 10 and 11
and after the final `encoder.ln`. Everything else the diagnostic needs is an exact function
of the already-cached block-11 post-attention residual and costs no forward pass:

    h        = ln_2(x_resid)                       input to W1
    y        = gelu(h @ W1 + b1)                   post-GELU, input to W2
    z        = y @ W2 + b2                         FFN output
    blockout = x_resid + z                         final block output
    phi      = encoder.ln(blockout)                the 768-d feature Mahalanobis uses

so the expensive pass is needed only for the earlier blocks (A8).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import full_cache as fc
from . import full_ood as fo
from .memprobe import GIB, reset_cuda
from .vit_adapter import ViTPnCAdapter

SRC = Path("results/neurips_2026_rebuttal/imagenet_vit")
OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_geometry_scod_llla")
BLOCKS = (8, 9, 10, 11)


@torch.inference_mode()
def cls_by_block(ad, images: torch.Tensor) -> dict:
    """CLS token after each requested encoder block, and after the final LayerNorm."""
    m = ad.model
    x = m._process_input(images)
    x = torch.cat([m.class_token.expand(x.shape[0], -1, -1), x], dim=1)
    x = ad.enc.dropout(x + ad.enc.pos_embedding)
    out = {}
    for i, layer in enumerate(ad.enc.layers):
        x = layer(x)
        if i in BLOCKS:
            out[f"cls_block{i}"] = x[:, 0].cpu()
    out["cls_final_ln"] = ad.enc.ln(x)[:, 0].cpu()
    return out


def build(ad, iter_batches, n_total: int, tag: str, path: Path) -> dict:
    if path.exists():
        print(f"  {tag}: cached")
        return {"tag": tag, "cached": True}
    reset_cuda()
    t0, acc, n = time.perf_counter(), {}, 0
    for imgs, *_ in iter_batches():
        d = cls_by_block(ad, imgs.to(ad.device, ad.dtype))
        for k, v in d.items():
            acc.setdefault(k, []).append(v)
        n += len(imgs)
    wall = time.perf_counter() - t0
    payload = {k: torch.cat(v).numpy().astype(np.float32) for k, v in acc.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    st = {"tag": tag, "n": n, "seconds": wall, "images_per_s": n / wall,
          "disk_mib": path.stat().st_size / 1024**2,
          "peak_gpu_gib": torch.cuda.max_memory_allocated() / GIB, "cached": False}
    print(f"  {tag}: {n:,} images in {wall/60:.1f} min ({n/wall:.0f} img/s), "
          f"{st['disk_mib']:.0f} MiB")
    del acc, payload
    reset_cuda()
    return st


def derived_from_residual(ad, x_cls: torch.Tensor, chunk: int = 8192) -> dict:
    """h, post-GELU y, FFN output z, block output and phi — all exact, no forward pass."""
    hs, ys, zs, bo, ph = [], [], [], [], []
    with torch.inference_mode():
        for s in range(0, x_cls.shape[0], chunk):
            x = x_cls[s:s + chunk]
            h = ad.block.ln_2(x[:, None, :])[:, 0]
            y = torch.nn.functional.gelu(h @ ad.W1 + ad.b1)
            z = y @ ad.W2 + ad.b2
            b = x + z
            p = ad.enc.ln(b[:, None, :])[:, 0]
            hs.append(h.cpu()); ys.append(y.cpu()); zs.append(z.cpu())
            bo.append(b.cpu()); ph.append(p.cpu())
    return {"h_w1_input": torch.cat(hs), "y_post_gelu": torch.cat(ys),
            "z_ffn_out": torch.cat(zs), "block_out": torch.cat(bo),
            "phi_final_ln": torch.cat(ph)}


def run():
    from .full_data import ImageNetShards, shard_paths
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    stats = {}
    val = ImageNetShards(shard_paths("val"))
    idx = np.arange(len(val))
    stats["val50k"] = build(ad, lambda: val.iter_batches(idx, 16), len(idx), "val50k",
                            OUT / "raw" / "cls_layers_val50k.npz")
    # A8 fits its layerwise Gaussians on the ID *training* calibration pool; fitting them
    # on the evaluation split would leak. The layer cache therefore needs the train pool
    # as well as val + OOD.
    train = ImageNetShards(shard_paths("train"))
    rows = np.load(Path("results/neurips_2026_rebuttal/imagenet_vit")
                   / "splits" / "correction_rows.npy")
    stats["correction"] = build(ad, lambda: train.iter_batches(rows, 16), len(rows),
                                "correction", OUT / "raw" / "cls_layers_correction.npz")
    for key in fo.DATASETS:
        d = fo.OODZip(key)
        stats[key] = build(ad, lambda d=d: d.iter_batches(16), len(d), key,
                           OUT / "raw" / f"cls_layers_{key}.npz")
    fc.write_json(OUT / "metrics" / "cls_layer_cache.json", stats)
    print(f"\nwrote {OUT/'raw'}")


if __name__ == "__main__":
    run()
