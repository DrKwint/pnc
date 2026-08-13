"""Base-activation caches for the full experiment (spec §6).

Everything downstream of the final block's FFN is token-wise followed by a CLS slice, so a
single cached tensor per image set — the CLS row of the target block's post-attention
residual, ``x_resid[:, 0]`` (768 floats/image) — is sufficient to reproduce **exactly**:

    base logits            adapter.tail(x)
    any member's logits    adapter.tail(x, W1=W1v, W2=W2c, b2=b2c)
    penultimate features   adapter.features(x)          (what ReAct clips)
    the correction design  h  = ln_2(x)      -> y_v = gelu(h @ W1v + b1)
    the correction target  z0 = gelu(h @ W1 + b1) @ W2 + b2

This is the preflight's `cls_only_parity` result applied at scale: the ViT prefix is paid
once per image set instead of once per configuration, which is what makes an 18-point
hyperparameter search and 5×2 final ensembles affordable on this card.

For the correction pool, ``h`` and ``z0`` are additionally materialised and stored because
§6 names them explicitly; both are exact functions of the cached residual.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .memprobe import GIB, cpu_peak_rss_gib, cpu_rss_gib, reset_cuda


@torch.inference_mode()
def build_cls_cache(adapter, dataset, indices, out_path: str | Path, batch: int = 16,
                    tag: str = "", store_hz: bool = False, progress_every: int = 8192):
    """Cache CLS residual (+ labels) for `indices`. Returns a stats dict."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reset_cuda()
    rss0 = cpu_rss_gib()
    t0 = time.perf_counter()

    xs, labs, n = [], [], 0
    for imgs, labels, _ in dataset.iter_batches(indices, batch):
        x = adapter.prefix(imgs.to(adapter.device, adapter.dtype))[:, 0]   # (B, 768)
        xs.append(x.cpu())
        labs.append(labels)
        n += len(labels)
        if progress_every and n % progress_every < batch:
            print(f"    {tag} {n}/{len(indices)} "
                  f"({n / (time.perf_counter() - t0):.1f} img/s)", flush=True)
    X = torch.cat(xs)
    Y = torch.cat(labs)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    payload = {"x_resid_cls": X.numpy().astype(np.float32),
               "labels": Y.numpy().astype(np.int64),
               "rows": np.asarray(indices, dtype=np.int64)}
    if store_hz:
        Xg = X.to(adapter.device, adapter.dtype)
        h = adapter.block.ln_2(Xg)
        z0 = torch.nn.functional.gelu(h @ adapter.W1 + adapter.b1) @ adapter.W2 + adapter.b2
        payload["h"] = h.cpu().numpy().astype(np.float32)
        payload["z0"] = z0.cpu().numpy().astype(np.float32)
        del Xg, h, z0
    np.savez(out_path, **payload)

    stats = {
        "tag": tag, "n_images": int(len(indices)),
        "cache_seconds": wall, "images_per_s": len(indices) / wall,
        "x_resid_cls_mib": X.numel() * 4 / 1024**2,
        "stored_h_z0": bool(store_hz),
        "disk_mib": out_path.stat().st_size / 1024**2,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / GIB,
        "cpu_rss_before_gib": rss0, "cpu_peak_rss_gib": cpu_peak_rss_gib(),
        "path": str(out_path),
    }
    del xs, labs, X, Y, payload
    reset_cuda()
    return stats


def load_cache(path: str | Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def cache_to_gpu(cache: dict, adapter, key: str = "x_resid_cls") -> torch.Tensor:
    return torch.as_tensor(cache[key], device=adapter.device, dtype=adapter.dtype)


@torch.inference_mode()
def logits_from_cache(adapter, X: torch.Tensor, W1=None, W2=None, b2=None,
                      chunk: int = 8192) -> torch.Tensor:
    """Logits for a cached CLS residual (N, 768) -> (N, 1000) on CPU, in chunks."""
    out = []
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk][:, None, :]                     # (n, 1, 768)
        out.append(adapter.tail(xb, W1=W1, W2=W2, b2=b2, cls_only=True).cpu())
    return torch.cat(out)


@torch.inference_mode()
def features_from_cache(adapter, X: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Penultimate CLS features (N, 768) on CPU — the representation ReAct clips."""
    out = []
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk][:, None, :]
        out.append(adapter.features(xb, cls_only=True).cpu())
    return torch.cat(out)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
