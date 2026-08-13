"""Driver for the full ImageNet ViT-B/16 P&C experiment.

Stages share state through ``raw/state.json``. Run them in order; each is idempotent and
skips work whose artefacts already exist.

    parity   §5  base checkpoint on the untouched 50k validation set
    cache    §6  CLS-residual caches for every image set (the prefix, paid once)
    stage_a  §8-11 coarse + adaptive perturbation-scale sweep, ID-only
    stage_b  §12-14 refined joint search, frozen selection
    final    §15-18 M=20 ensembles over seeds, temperature, ID eval, uncorrected ablation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.banking77_pnc.pnc_metrics import clf_metrics  # noqa: E402  REUSED evaluator

from . import full_cache as fc  # noqa: E402
from .full_data import ImageNetShards, shard_paths  # noqa: E402
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda  # noqa: E402
from .vit_adapter import ViTPnCAdapter, checkpoint_info  # noqa: E402

OUT = Path("results/neurips_2026_rebuttal/imagenet_vit")
BATCH = 16


def state_path() -> Path:
    return OUT / "raw" / "state.json"


def load_state() -> dict:
    p = state_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(st, indent=2, default=str))
    return st


def load_split(name: str) -> np.ndarray:
    return np.load(OUT / "splits" / f"{name}_rows.npy")


def topk_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """top-1/top-5 plus the shared clf_metrics (accuracy, NLL, Brier, 15-bin ECE)."""
    probs = torch.softmax(logits.double(), -1).numpy()
    m = clf_metrics(probs, labels.numpy())
    top5 = logits.topk(5, dim=-1).indices
    m["top1"] = m.pop("accuracy")
    m["top5"] = float((top5 == labels[:, None]).any(-1).double().mean())
    return m


# --------------------------------------------------------------------- §5
def stage_parity(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    info = checkpoint_info(ad.model)
    print(f"checkpoint {info.weight_enum}  sha256 {info.sha256[:16]}...  "
          f"{info.n_params:,} params")
    val = ImageNetShards(shard_paths("val"))
    print(f"official validation set: {len(val):,} images")
    assert len(val) == 50000, f"expected 50,000 validation images, got {len(val)}"

    idx = np.arange(len(val))
    st = fc.build_cls_cache(ad, val, idx, OUT / "raw" / "cache_val50k.npz", BATCH,
                            tag="val50k")
    print(f"  cached in {st['cache_seconds']/60:.1f} min at {st['images_per_s']:.1f} img/s "
          f"({st['disk_mib']:.0f} MiB, peak GPU {st['peak_gpu_gib']:.3f} GiB)")

    cache = fc.load_cache(OUT / "raw" / "cache_val50k.npz")
    X = fc.cache_to_gpu(cache, ad)
    logits = fc.logits_from_cache(ad, X)
    labels = torch.from_numpy(cache["labels"])
    m = topk_metrics(logits, labels)
    print(f"\n  BASE  top-1 {m['top1']*100:.3f}%  top-5 {m['top5']*100:.3f}%  "
          f"NLL {m['nll']:.4f}  ECE {m['ece']:.4f}  Brier {m['brier']:.4f}")
    print(f"  published torchvision reference: top-1 81.072%  top-5 95.318%")

    d1 = abs(m["top1"] * 100 - 81.072)
    ok = d1 < 0.5
    print(f"  |delta top-1| = {d1:.3f} pp -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("base accuracy inconsistent with the checkpoint - debug "
                         "preprocessing/labels before continuing (spec §5)")

    np.save(OUT / "raw" / "base_val_logits.npy", logits.numpy().astype(np.float32))
    fc.write_json(OUT / "metrics" / "base_id_metrics.json",
                  {"split": "imagenet_val_50k", **m,
                   "reference_top1": 0.81072, "reference_top5": 0.95318,
                   "delta_top1_pp": d1, "checkpoint": info.__dict__})
    fc.write_json(OUT / "timing" / "cache_val50k.json", st)
    save_state(parity_done=True, base_top1=m["top1"], base_top5=m["top5"],
               base_nll=m["nll"], base_ece=m["ece"])
    del X, cache
    reset_cuda()


# --------------------------------------------------------------------- §6
def stage_cache(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    train = ImageNetShards(shard_paths("train"))
    stats = {}
    for name, store_hz in (("correction", True), ("selection", False),
                           ("temperature", False)):
        idx = load_split(name)
        path = OUT / "raw" / f"cache_{name}.npz"
        if path.exists() and not args.force:
            print(f"  {name}: cache exists, skipping")
            continue
        print(f"  caching {name} ({len(idx):,} train images) ...", flush=True)
        st = fc.build_cls_cache(ad, train, idx, path, BATCH, tag=name, store_hz=store_hz)
        stats[name] = st
        print(f"    {st['cache_seconds']/60:.1f} min at {st['images_per_s']:.1f} img/s, "
              f"{st['disk_mib']:.0f} MiB on disk, peak GPU {st['peak_gpu_gib']:.3f} GiB, "
              f"peak CPU {st['cpu_peak_rss_gib']:.2f} GiB")
    if stats:
        fc.write_json(OUT / "timing" / "cache_train_pools.json", stats)
    save_state(cache_done=True)


STAGES = {"parity": stage_parity, "cache": stage_cache}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=5)
    args, extra = ap.parse_known_args()
    if args.stage in STAGES:
        STAGES[args.stage](args)
    elif args.stage in ("stage_a", "stage_b", "robustness"):
        from . import full_search
        full_search.dispatch(args.stage, args, extra, OUT, BATCH)
    elif args.stage == "final":
        from .full_final import run_final
        run_final(args, OUT)
    elif args.stage == "ood":
        from .full_oodeval import run_ood
        run_ood(args, OUT)
    else:
        raise SystemExit(f"unknown stage {args.stage!r}")


if __name__ == "__main__":
    main()
