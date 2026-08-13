"""Final ensembles, temperature, ImageNet ID evaluation and the uncorrected ablation.

Spec §15-18. Runs only after ``selection/selected_config.json`` is frozen.

Temperature follows the existing classification protocol exactly (see
``banking77_pnc/evaluate.py``): **one shared scalar temperature**, fitted by grid search on
the *base model's* logits over an ID pool, then applied to every member's logits before the
softmax. It is not refitted per seed and not fitted on the ensemble.

The uncorrected ablation (§18) reuses the *identical* W1 mutation — same basis, same
coefficients, same scale, same seed — and differs only in whether the corrected (W2, b2)
or the original ones are used. Nothing else changes.

Member logits over 50,000 images are 200 MiB each, so per-member statistics are accumulated
incrementally and only the ensemble mean probability is retained.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.banking77_pnc.pnc_metrics import clf_metrics, fit_temperature  # REUSED

from . import full_cache as fc
from . import full_pnc as fp
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter

SEEDS_PREFERRED = [0, 10, 42, 123, 2026]


class RunningMSE:
    """Streams per-(member, image) logit MSE quantiles without holding every logit."""

    def __init__(self):
        self.vals = []

    def add(self, member_logits: torch.Tensor, base_logits: torch.Tensor):
        d = ((member_logits.double() - base_logits.double()) ** 2).mean(-1)
        self.vals.append(d.float().numpy())

    def summary(self) -> dict:
        v = np.concatenate(self.vals)
        q = np.percentile(v, [50, 90, 95, 99])
        return {"mean": float(v.mean()), "median": float(q[0]), "p90": float(q[1]),
                "p95": float(q[2]), "p99": float(q[3]), "max": float(v.max()),
                "n_pairs": int(v.size)}


def fit_shared_temperature(adapter, out: Path) -> dict:
    """One shared temperature from base logits on the ID temperature pool (§16)."""
    path = out / "metrics" / "temperature.json"
    if path.exists():
        return json.loads(path.read_text())
    cache = fc.load_cache(out / "raw" / "cache_temperature.npz")
    X = fc.cache_to_gpu(cache, adapter)
    logits = fc.logits_from_cache(adapter, X).numpy().astype(np.float64)
    labels = cache["labels"]
    T = fit_temperature(logits, labels)
    info = {"temperature": float(T), "protocol": "shared scalar T, grid search on base "
            "model logits over the ID temperature pool; applied to every member's logits "
            "before softmax (matches banking77_pnc/evaluate.py)",
            "pool": "temperature_8192 (ImageNet train)", "n_images": int(len(labels))}
    fc.write_json(path, info)
    del X
    reset_cuda()
    return info


@torch.inference_mode()
def evaluate_ensemble(adapter, X: torch.Tensor, labels: np.ndarray, base_logits,
                      members: list, T: float, corrected: bool, chunk: int = 8192):
    """Ensemble metrics + per-member preservation over a cached image set."""
    acc = None
    mse = RunningMSE()
    for mem in members:
        lg = fp.member_logits(adapter, X, mem, corrected=corrected, chunk=chunk)
        mse.add(lg, base_logits)
        p = torch.softmax(lg.double() / T, dim=-1)
        acc = p if acc is None else acc + p
        del lg, p
    pbar = (acc / len(members)).numpy()
    m = clf_metrics(pbar, labels)
    pred = pbar.argmax(-1)
    top5 = torch.from_numpy(pbar).topk(5, -1).indices.numpy()
    ent = -np.sum(pbar * np.log(pbar + 1e-12), -1)
    base_pred = base_logits.argmax(-1).numpy()
    return {
        "top1": m["accuracy"],
        "top5": float((top5 == labels[:, None]).any(-1).mean()),
        "nll": m["nll"], "ece": m["ece"], "brier": m["brier"],
        "base_agreement": float((pred == base_pred).mean()),
        "mean_pred_entropy": float(ent.mean()),
        "logit_mse": mse.summary(),
        "n_images": int(len(labels)),
    }, pbar, ent


def build_members_for_seed(adapter, h, z0, cfg: dict, seed: int) -> dict:
    factory = fp.MemberFactory(adapter, h, z0, seed, K=cfg["K"], M=cfg["M_final"])
    t0 = time.perf_counter()
    built = factory.build(cfg["r_target"], cfg["lambda"], cfg["n_cal"])
    built["wall_s"] = time.perf_counter() - t0
    return built


def run_final(args, out: Path):
    cfg = json.loads((out / "selection" / "selected_config.json").read_text())
    print(f"selected config: r={cfg['r_target']} lambda={cfg['lambda']} "
          f"n_cal={cfg['n_cal']} K={cfg['K']} M={cfg['M_final']}")

    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    corr = fc.load_cache(out / "raw" / "cache_correction.npz")
    h = torch.as_tensor(corr["h"], device=ad.device, dtype=ad.dtype)
    z0 = torch.as_tensor(corr["z0"], device=ad.device, dtype=ad.dtype)

    temp = fit_shared_temperature(ad, out)
    T = temp["temperature"]
    print(f"shared temperature T = {T:.4f} (fit on base logits, ID temperature pool)")

    val = fc.load_cache(out / "raw" / "cache_val50k.npz")
    Xval = fc.cache_to_gpu(val, ad)
    yval = val["labels"]
    base_logits = torch.from_numpy(np.load(out / "raw" / "base_val_logits.npy"))
    base_m = clf_metrics(torch.softmax(base_logits.double() / T, -1).numpy(), yval)
    print(f"base (T-scaled): top-1 {base_m['accuracy']*100:.3f}%  NLL {base_m['nll']:.4f}  "
          f"ECE {base_m['ece']:.4f}")

    seeds = SEEDS_PREFERRED[:args.n_seeds]
    results = {"config": cfg, "temperature": temp, "seeds": {},
               "base": {**base_m, "top1": base_m.pop("accuracy")}}
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        reset_cuda()
        built = build_members_for_seed(ad, h, z0, cfg, seed)
        print(f"  built M={len(built['members'])} in {built['wall_s']:.1f}s "
              f"(realized r median {built['realized_r']['median']:.4f}, "
              f"calib residual {built['calib_residual_median']:.4f})")
        peak_build = torch.cuda.max_memory_allocated() / GIB

        t0 = time.perf_counter()
        pc_m, pc_probs, pc_ent = evaluate_ensemble(
            ad, Xval, yval, base_logits, built["members"], T, corrected=True)
        un_m, un_probs, un_ent = evaluate_ensemble(
            ad, Xval, yval, base_logits, built["members"], T, corrected=False)
        t_eval = time.perf_counter() - t0

        print(f"  P&C          top-1 {pc_m['top1']*100:.3f}%  top-5 {pc_m['top5']*100:.3f}%  "
              f"NLL {pc_m['nll']:.4f}  ECE {pc_m['ece']:.4f}  "
              f"agree {pc_m['base_agreement']:.4f}")
        print(f"  uncorrected  top-1 {un_m['top1']*100:.3f}%  top-5 {un_m['top5']*100:.3f}%  "
              f"NLL {un_m['nll']:.4f}  ECE {un_m['ece']:.4f}  "
              f"agree {un_m['base_agreement']:.4f}")
        print(f"  logit MSE  P&C median {pc_m['logit_mse']['median']:.3e} "
              f"p99 {pc_m['logit_mse']['p99']:.3e} | uncorrected median "
              f"{un_m['logit_mse']['median']:.3e} p99 {un_m['logit_mse']['p99']:.3e}")

        np.savez_compressed(
            out / "predictions" / f"id_val_seed{seed}.npz",
            pnc_entropy=pc_ent.astype(np.float32),
            unc_entropy=un_ent.astype(np.float32),
            pnc_pred=pc_probs.argmax(-1).astype(np.int16),
            unc_pred=un_probs.argmax(-1).astype(np.int16),
            pnc_conf=pc_probs.max(-1).astype(np.float32),
            unc_conf=un_probs.max(-1).astype(np.float32),
            labels=yval.astype(np.int16))
        # compact member state (never 20 model copies)
        np.savez_compressed(
            out / "construction" / f"members_seed{seed}.npz",
            coefficients=np.stack([m["coeff"] for m in built["members"]]),
            W2=np.stack([m["W2"].cpu().numpy() for m in built["members"]]).astype(np.float32),
            b2=np.stack([m["b2"].cpu().numpy() for m in built["members"]]).astype(np.float32),
            scale=built["scale"], r_target=cfg["r_target"], seed=seed)

        results["seeds"][str(seed)] = {
            "pnc": pc_m, "uncorrected": un_m,
            "construct_s": built["wall_s"], "eval_s": t_eval,
            "peak_gpu_build_gib": peak_build,
            "peak_gpu_eval_gib": torch.cuda.max_memory_allocated() / GIB,
            "realized_r": built["realized_r"]["median"],
            "calib_residual": built["calib_residual_median"],
            "all_finite": built["all_finite"],
        }
        for mem in built["members"]:
            del mem["W1v"], mem["W2"], mem["b2"]
        del built, pc_probs, un_probs
        reset_cuda()

    _aggregate(results, seeds)
    fc.write_json(out / "metrics" / "id_final.json", results)
    print(f"\nwrote {out/'metrics'/'id_final.json'}")
    return results


def _aggregate(results: dict, seeds: list[int]):
    agg = {}
    for method in ("pnc", "uncorrected"):
        agg[method] = {}
        for k in ("top1", "top5", "nll", "ece", "brier", "base_agreement",
                  "mean_pred_entropy"):
            v = [results["seeds"][str(s)][method][k] for s in seeds]
            agg[method][k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))
                              if len(v) > 1 else 0.0, "per_seed": v}
        for q in ("mean", "median", "p90", "p95", "p99"):
            v = [results["seeds"][str(s)][method]["logit_mse"][q] for s in seeds]
            agg[method][f"logit_mse_{q}"] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    results["aggregate"] = agg
    print("\n=== aggregate over seeds ===")
    for method in ("pnc", "uncorrected"):
        a = agg[method]
        print(f"  {method:<12} top-1 {a['top1']['mean']*100:.3f}±{a['top1']['std']*100:.3f}  "
              f"NLL {a['nll']['mean']:.4f}±{a['nll']['std']:.4f}  "
              f"ECE {a['ece']['mean']:.4f}  agree {a['base_agreement']['mean']:.4f}")
