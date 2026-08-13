"""Correctness gates for the full experiment.

The load-bearing assumption is that a cached CLS residual reproduces the full model
exactly; every ID number, every OOD score and every baseline is computed through it. Gate 3
checks that against real images rather than trusting the preflight's synthetic-input result.

    1 split_integrity      pools disjoint, class-stratified, sizes exact
    2 ood_lists            all five OOD sets resolve 100% of OpenOOD's canonical list
    3 cache_forward_parity cached-residual logits == full model logits on real images
    4 ablation_identity    uncorrected ablation uses a bit-identical W1 mutation
    5 ood_metric_parity    metrics match pnc_core/openood_eval (needs jax; see below)

Gate 5 needs the main venv:
    JAX_PLATFORMS=cpu .venv/bin/python -m experiments.imagenet_vit_pnc.full_validate \
        --only ood_metric_parity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS = []
OUT = Path("results/neurips_2026_rebuttal/imagenet_vit")


def gate(name, ok, detail):
    RESULTS.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    return ok


def split_integrity():
    import csv
    pools = {}
    for name in ("correction", "selection", "temperature"):
        rows = list(csv.DictReader((OUT / "splits" / f"{name}_{'32768' if name=='correction' else '8192'}.csv").open()))
        pools[name] = {int(r["global_row"]) for r in rows}
        cls = np.array([int(r["class"]) for r in rows])
        _, counts = np.unique(cls, return_counts=True)
        gate(f"split_{name}", len(cls) == (32768 if name == "correction" else 8192)
             and len(np.unique(cls)) == 1000 and counts.max() - counts.min() <= 1,
             f"{len(cls)} images, {len(np.unique(cls))} classes, "
             f"{counts.min()}-{counts.max()} per class")
    names = sorted(pools)
    overlaps = {f"{a}&{b}": len(pools[a] & pools[b])
                for i, a in enumerate(names) for b in names[i + 1:]}
    gate("split_disjoint", all(v == 0 for v in overlaps.values()), f"{overlaps}")


def ood_lists():
    from .full_ood import CANONICAL_N, OODZip
    ok, detail = True, []
    for key in CANONICAL_N:
        d = OODZip(key)
        ok &= len(d) == CANONICAL_N[key]
        detail.append(f"{key}={len(d)}")
    gate("ood_lists", ok, ", ".join(detail))


def cache_forward_parity(n: int = 256, batch: int = 16):
    """The whole experiment reads logits out of a cache; prove it equals the real model."""
    import torch

    from . import full_cache as fc
    from .full_data import ImageNetShards, shard_paths
    from .vit_adapter import ViTPnCAdapter

    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    val = ImageNetShards(shard_paths("val"))
    cache = fc.load_cache(OUT / "raw" / "cache_val50k.npz")
    idx = np.linspace(0, len(val) - 1, n).astype(int)

    direct = []
    with torch.inference_mode():
        for imgs, _, _ in val.iter_batches(idx, batch):
            direct.append(ad.model(imgs.to(ad.device, ad.dtype)).cpu())
    direct = torch.cat(direct)

    X = torch.as_tensor(cache["x_resid_cls"][idx], device=ad.device, dtype=ad.dtype)
    cached = fc.logits_from_cache(ad, X)
    d = float((direct - cached).abs().max())
    gate("cache_forward_parity", d < 1e-4,
         f"max|full model - cached tail| = {d:.3e} over {n} real validation images "
         f"(tol 1e-4)")

    feats = fc.features_from_cache(ad, X)
    dh = float((ad.head(feats.to(ad.device)).cpu() - cached).abs().max())
    gate("features_head_parity", dh < 1e-4,
         f"max|head(features) - logits| = {dh:.3e} (ReAct path consistency)")

    labels = torch.from_numpy(cache["labels"][idx])
    gate("cache_labels", bool((direct.argmax(-1) == cached.argmax(-1)).all()),
         f"top-1 agreement between paths = "
         f"{float((direct.argmax(-1) == cached.argmax(-1)).float().mean()):.4f}; "
         f"cached-label accuracy {float((cached.argmax(-1) == labels).float().mean()):.4f}")


def ablation_identity():
    """The uncorrected ablation must differ from P&C only in W2/b2."""
    import torch

    from . import full_pnc as fp
    from . import pnc_core as pc
    from .vit_adapter import ViTPnCAdapter

    cfg = json.loads((OUT / "selection" / "selected_config.json").read_text())
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    z = np.load(OUT / "construction" / "members_seed0.npz")
    U = pc.perturbation_basis(0, cfg["K"])
    dW1 = pc.member_dW1(U, z["coefficients"][0], float(z["scale"]))
    W1v = ad.W1 + torch.as_tensor(dW1, device=ad.device, dtype=ad.dtype)

    mem = {"W1v": W1v,
           "W2": torch.as_tensor(z["W2"][0], device=ad.device, dtype=ad.dtype),
           "b2": torch.as_tensor(z["b2"][0], device=ad.device, dtype=ad.dtype)}
    X = torch.randn(64, 768, device=ad.device, dtype=ad.dtype)
    lc = fp.member_logits(ad, X, mem, corrected=True)
    lu = fp.member_logits(ad, X, mem, corrected=False)
    lb = fp.member_logits(ad, X, None)
    gate("ablation_identity",
         not torch.equal(lc, lu) and not torch.equal(lu, lb),
         f"corrected != uncorrected (max diff {float((lc-lu).abs().max()):.3e}) and "
         f"uncorrected != base (max diff {float((lu-lb).abs().max()):.3e}); "
         "both share one W1 mutation")

    r = float(np.linalg.norm(dW1) / float(ad.W1.norm()))
    gate("realized_scale", abs(r - cfg["r_target"]) / cfg["r_target"] < 0.25,
         f"member 0 realized r = {r:.4f} (target {cfg['r_target']}, "
         f"median over members {cfg['r_realized_median']:.4f})")


def ood_metric_parity(seed: int = 0):
    """Gate the reproduced metrics against pnc_core/openood_eval (needs jax)."""
    from . import full_oodmetrics as fm
    try:
        from pnc_core.openood_eval import (_aggregate_family_metrics,
                                           _binary_ood_metrics)
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] ood_metric_parity: pnc_core.openood_eval unavailable "
              f"({type(exc).__name__}: {str(exc)[:70]})")
        return
    rng = np.random.RandomState(seed)
    idd = rng.normal(size=2000)
    fam = {"a": rng.normal(0.7, 1.0, 900), "b": rng.normal(1.3, 1.2, 1500)}
    mine = fm.binary_ood_metrics(idd, fam["a"])
    ref = _binary_ood_metrics(idd, fam["a"])
    ok1 = all(abs(mine[k] - ref[k]) < 1e-12 for k in ("auroc", "aupr", "fpr95"))
    gate("ood_metric_parity", ok1,
         f"auroc/aupr/fpr95 match to <1e-12 (auroc {mine['auroc']:.10f})")

    mine_a = fm.aggregate_family_metrics(idd, fam)
    ref_a = _aggregate_family_metrics({"s": idd}, {k: {"s": v} for k, v in fam.items()})["s"]
    ok2 = all(abs(mine_a[k] - ref_a[k]) < 1e-12
              for k in ("mean_auroc", "mean_aupr", "mean_fpr95", "concat_auroc"))
    gate("ood_aggregate_parity", ok2,
         f"Near/Far macro aggregation matches to <1e-12 "
         f"(mean_auroc {mine_a['mean_auroc']:.10f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.only == "ood_metric_parity":
        ood_metric_parity()
    else:
        split_integrity()
        ood_lists()
        cache_forward_parity()
        ablation_identity()
        ood_metric_parity()
    n_fail = sum(1 for r in RESULTS if not r["pass"])
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} gates passed")
    path = Path(args.out) if args.out else OUT / "raw" / "validate_full_gates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() and args.only else []
    path.write_text(json.dumps(existing + RESULTS, indent=2))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
