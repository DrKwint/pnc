"""OpenOOD ImageNet-1k evaluation for P&C and the post-hoc baselines (spec §19-24).

Runs **only** after ``selection/selected_config.json`` exists. Nothing in this module feeds
back into configuration choice; the headline score is fixed in advance as the predictive
entropy of the temperature-scaled mean member softmax (§20).

Per-dataset metrics and Near/Far aggregation follow the repo's existing OpenOOD convention
(``full_oodmetrics``, gated against ``pnc_core/openood_eval.py``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import full_baselines as fb
from . import full_cache as fc
from . import full_ood as fo
from . import full_oodmetrics as fm
from . import full_pnc as fp
from . import pnc_core as pc
from .memprobe import GIB, reset_cuda
from .vit_adapter import ViTPnCAdapter


def cache_ood_sets(ad, out: Path, batch: int = 16, force: bool = False) -> dict:
    """Cache the CLS residual for every canonical OOD test image."""
    stats = {}
    for key in fo.DATASETS:
        path = out / "raw" / f"cache_ood_{key}.npz"
        if path.exists() and not force:
            print(f"  {key}: cached")
            continue
        ds = fo.OODZip(key)
        print(f"  caching {key} ({len(ds):,} images) ...", flush=True)
        reset_cuda()
        t0 = time.perf_counter()
        xs = []
        with torch.inference_mode():
            for imgs, _, _ in ds.iter_batches(batch):
                xs.append(ad.prefix(imgs.to(ad.device, ad.dtype))[:, 0].cpu())
        X = torch.cat(xs)
        wall = time.perf_counter() - t0
        np.savez(path, x_resid_cls=X.numpy().astype(np.float32),
                 ids=np.asarray(ds.ids(), dtype=object))
        stats[key] = {"n": len(ds), "seconds": wall, "images_per_s": len(ds) / wall,
                      "group": ds.group, "disk_mib": path.stat().st_size / 1024**2,
                      "peak_gpu_gib": torch.cuda.max_memory_allocated() / GIB}
        print(f"    {wall/60:.1f} min at {len(ds)/wall:.1f} img/s")
        del xs, X
        reset_cuda()
    return stats


def load_members(ad, out: Path, cfg: dict, seed: int) -> list[dict]:
    """Rebuild a seed's members from compact state (basis is regenerated from the seed)."""
    z = np.load(out / "construction" / f"members_seed{seed}.npz")
    U = pc.perturbation_basis(seed, cfg["K"])
    coeffs, scale = z["coefficients"], float(z["scale"])
    members = []
    for m in range(len(coeffs)):
        dW1 = torch.as_tensor(pc.member_dW1(U, coeffs[m], scale), device=ad.device,
                              dtype=ad.dtype)
        members.append({"W1v": ad.W1 + dW1,
                        "W2": torch.as_tensor(z["W2"][m], device=ad.device, dtype=ad.dtype),
                        "b2": torch.as_tensor(z["b2"][m], device=ad.device, dtype=ad.dtype)})
        del dW1
    return members


@torch.inference_mode()
def pnc_entropy(ad, X: torch.Tensor, members: list, T: float, corrected: bool) -> np.ndarray:
    acc = None
    for mem in members:
        lg = fp.member_logits(ad, X, mem, corrected=corrected)
        p = torch.softmax(lg.double() / T, dim=-1)
        acc = p if acc is None else acc + p
        del lg, p
    pbar = (acc / len(members)).numpy()
    return fp.predictive_entropy(pbar)


def run_ood(args, out: Path):
    cfg = json.loads((out / "selection" / "selected_config.json").read_text())
    temp = json.loads((out / "metrics" / "temperature.json").read_text())
    T = temp["temperature"]
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)

    print("== caching OOD activations ==")
    cache_stats = cache_ood_sets(ad, out, force=args.force)
    if cache_stats:
        fc.write_json(out / "timing" / "cache_ood.json", cache_stats)

    # ---- ID side ----
    val = fc.load_cache(out / "raw" / "cache_val50k.npz")
    Xval = fc.cache_to_gpu(val, ad)
    tempc = fc.load_cache(out / "raw" / "cache_temperature.npz")
    Xtemp = fc.cache_to_gpu(tempc, ad)
    react_c = fb.react_threshold(fc.features_from_cache(ad, Xtemp), percentile=90.0)
    print(f"\nReAct clipping threshold (ID temperature pool, p90): {react_c:.4f}")
    del Xtemp
    reset_cuda()

    id_scores = {}
    b = fb.base_scores(ad, Xval, react_c=react_c)
    id_scores["MSP"] = b["msp"]
    id_scores["Energy"] = b["energy"]
    id_scores["ReAct+Energy"] = b["react_energy"]

    seeds = [int(s) for s in json.loads(
        (out / "metrics" / "id_final.json").read_text())["seeds"]]
    for seed in seeds:
        pred = np.load(out / "predictions" / f"id_val_seed{seed}.npz")
        id_scores[f"P&C(seed{seed})"] = pred["pnc_entropy"].astype(np.float64)
        id_scores[f"Uncorrected(seed{seed})"] = pred["unc_entropy"].astype(np.float64)

    # ---- OOD side ----
    results = {"config": cfg, "temperature": T, "react_percentile": 90.0,
               "react_threshold": react_c, "seeds": seeds, "per_dataset": {}}
    ood_scores = {m: {} for m in id_scores}
    groups = {}
    for key, (_, _, group) in fo.DATASETS.items():
        cache = fc.load_cache(out / "raw" / f"cache_ood_{key}.npz")
        X = fc.cache_to_gpu(cache, ad)
        groups[key] = group
        print(f"\n-- {key} ({X.shape[0]:,} images, {group}) --", flush=True)
        ob = fb.base_scores(ad, X, react_c=react_c)
        ood_scores["MSP"][key] = ob["msp"]
        ood_scores["Energy"][key] = ob["energy"]
        ood_scores["ReAct+Energy"][key] = ob["react_energy"]
        for seed in seeds:
            members = load_members(ad, out, cfg, seed)
            ood_scores[f"P&C(seed{seed})"][key] = pnc_entropy(ad, X, members, T, True)
            ood_scores[f"Uncorrected(seed{seed})"][key] = pnc_entropy(ad, X, members, T,
                                                                     False)
            for mem in members:
                del mem["W1v"], mem["W2"], mem["b2"]
            del members
            reset_cuda()
        results["per_dataset"][key] = {"group": group, "n_ood": int(X.shape[0])}
        for method in id_scores:
            met = fm.binary_ood_metrics(id_scores[method], ood_scores[method][key])
            results["per_dataset"][key][method] = met
        np.savez_compressed(
            out / "predictions" / f"ood_{key}.npz",
            ids=cache["ids"],
            **{m.replace("&", "and"): ood_scores[m][key].astype(np.float32)
               for m in ood_scores})
        for m in ("MSP", "Energy", "ReAct+Energy", f"P&C(seed{seeds[0]})"):
            r = results["per_dataset"][key][m]
            print(f"   {m:<22} AUROC {r['auroc']*100:6.2f}  FPR95 {r['fpr95']*100:6.2f}")
        del X, cache
        reset_cuda()

    # ---- Near / Far aggregation (repo convention: macro mean over datasets) ----
    results["aggregate"] = {}
    for method in id_scores:
        agg = {}
        for group in ("near", "far"):
            fam = {k: v for k, v in ood_scores[method].items() if groups[k] == group}
            agg[group] = fm.aggregate_family_metrics(id_scores[method], fam)
        results["aggregate"][method] = agg
    _summarise_over_seeds(results, seeds)
    fc.write_json(out / "metrics" / "ood_results.json", results)
    print(f"\nwrote {out/'metrics'/'ood_results.json'}")
    return results


def _summarise_over_seeds(results: dict, seeds: list[int]):
    """Mean +/- std over construction seeds for the stochastic methods."""
    out = {}
    for label, prefix in (("P&C", "P&C(seed"), ("Uncorrected", "Uncorrected(seed")):
        rows = [results["aggregate"][f"{prefix}{s})"] for s in seeds]
        out[label] = {}
        for group in ("near", "far"):
            for stat in ("mean_auroc", "mean_fpr95", "mean_aupr", "mean_aupr_in"):
                v = [r[group][stat] for r in rows]
                out[label][f"{group}_{stat}"] = {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    for label in ("MSP", "Energy", "ReAct+Energy"):
        out[label] = {f"{g}_{s}": {"mean": results["aggregate"][label][g][s], "std": 0.0}
                      for g in ("near", "far")
                      for s in ("mean_auroc", "mean_fpr95", "mean_aupr", "mean_aupr_in")}
    results["summary"] = out
    print("\n=== Near / Far summary (macro mean over datasets) ===")
    print(f"  {'method':<16}{'Near AUROC':>12}{'Near FPR95':>12}{'Far AUROC':>12}"
          f"{'Far FPR95':>12}")
    for label, v in out.items():
        print(f"  {label:<16}{v['near_mean_auroc']['mean']*100:>11.2f} "
              f"{v['near_mean_fpr95']['mean']*100:>11.2f} "
              f"{v['far_mean_auroc']['mean']*100:>11.2f} "
              f"{v['far_mean_fpr95']['mean']*100:>11.2f}")
