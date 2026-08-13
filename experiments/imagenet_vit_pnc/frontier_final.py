"""Final ensembles, uncorrected ablation and OOD for the three frozen operating points.

Spec §16-18. Runs only after `selection/frozen_configs.json` exists.

Everything reusable is reused rather than recomputed: the ID/OOD activation caches, the
base validation logits, the shared temperature T = 0.700, and the deterministic
MSP / Energy / ReAct+Energy baselines from the original experiment (identical checkpoint,
images, preprocessing and scoring code — §18 explicitly says not to rerun them merely to
generate different numbers).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.banking77_pnc.pnc_metrics import clf_metrics  # REUSED evaluator

from . import full_cache as fc
from . import full_ood as fo
from . import full_oodmetrics as fm
from . import full_pnc as fp
from . import pnc_core as pc
from .frontier import BUDGETS, OUT, SRC, paired_bootstrap_lcb
from .memprobe import GIB, reset_cuda
from .vit_adapter import ViTPnCAdapter

SEEDS_FINAL = [0, 10, 42, 123, 2026]
M_FINAL = 20
K = 20


def _build(ad, h, z0, Theta0, seed, r, lam, M=M_FINAL):
    """M members at one (r, lam); directions drawn once per seed, scaled to r."""
    U = pc.perturbation_basis(seed, K)
    co = pc.member_coefficients(seed, M, K)
    W1n = float(ad.W1.norm())
    scale = pc.base_scale(U, co, W1n, target_rel=r)
    members, rs, calib = [], [], []
    for m in range(M):
        dW1 = torch.as_tensor(pc.member_dW1(U, co[m], scale), device=ad.device,
                              dtype=ad.dtype)
        rs.append(float(torch.linalg.norm(dW1.double()) / W1n))
        W1v = ad.W1 + dW1
        y = torch.nn.functional.gelu(h @ W1v + ad.b1)
        X = pc.SufficientStats.augment(y)
        G = (X.T @ X).double().cpu().numpy()
        C = (X.T @ z0).double().cpu().numpy()
        yty = float((z0.double() ** 2).sum())
        Theta, _, _ = pc.cho_solve_shared(G, C, lam, w_prior=Theta0)
        calib.append(pc.relative_residual({"G": G, "C": C, "yty": yty}, Theta))
        members.append({"coeff": co[m], "W1v": W1v,
                        "W2": torch.as_tensor(Theta[1:], device=ad.device, dtype=ad.dtype),
                        "b2": torch.as_tensor(Theta[0], device=ad.device, dtype=ad.dtype),
                        "finite": bool(np.isfinite(Theta).all())})
        del y, X, dW1
    return {"members": members, "scale": scale,
            "realized_r_median": float(np.median(rs)),
            "calib_residual_median": float(np.median(calib)),
            "all_finite": all(m["finite"] for m in members)}


@torch.inference_mode()
def _evaluate(ad, X, labels, base_logits, members, T, corrected, chunk=8192):
    acc, mses = None, []
    for mem in members:
        lg = fp.member_logits(ad, X, mem, corrected=corrected, chunk=chunk)
        mses.append(((lg.double() - base_logits.double()) ** 2).mean(-1).numpy())
        p = torch.softmax(lg.double() / T, -1)
        acc = p if acc is None else acc + p
        del lg, p
    pbar = (acc / len(members)).numpy()
    m = clf_metrics(pbar, labels)
    pred = pbar.argmax(-1)
    top5 = torch.from_numpy(pbar).topk(5, -1).indices.numpy()
    v = np.concatenate(mses)
    q = np.percentile(v, [50, 90, 95, 99])
    ent = -np.sum(pbar * np.log(pbar + 1e-12), -1)
    return {"top1": m["accuracy"],
            "top5": float((top5 == labels[:, None]).any(-1).mean()),
            "nll": m["nll"], "ece": m["ece"], "brier": m["brier"],
            "base_agreement": float((pred == base_logits.argmax(-1).numpy()).mean()),
            "mean_pred_entropy": float(ent.mean()),
            "logit_mse_mean": float(v.mean()), "logit_mse_median": float(q[0]),
            "logit_mse_p90": float(q[1]), "logit_mse_p95": float(q[2]),
            "logit_mse_p99": float(q[3])}, pbar, ent


def run_final(args):
    frozen = json.loads((OUT / "selection" / "frozen_configs.json").read_text())
    T = frozen["temperature"]
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    corr = fc.load_cache(SRC / "raw" / "cache_correction.npz")
    val = fc.load_cache(SRC / "raw" / "cache_val50k.npz")
    Xval = fc.cache_to_gpu(val, ad)
    yval = val["labels"]
    base_logits = torch.from_numpy(np.load(SRC / "raw" / "base_val_logits.npy"))
    base_correct = (base_logits.argmax(-1).numpy() == yval).astype(np.float64)
    base_m = clf_metrics(torch.softmax(base_logits.double() / T, -1).numpy(), yval)
    print(f"base on 50k val (T={T}): top-1 {base_m['accuracy']*100:.3f}%  "
          f"NLL {base_m['nll']:.4f}  ECE {base_m['ece']:.4f}")

    Theta0 = ad.theta().detach().double().cpu().numpy()
    seeds = SEEDS_FINAL[:args.n_seeds]
    results = {"temperature": T, "seeds": seeds, "M": M_FINAL,
               "base": {**base_m, "top1": base_m.pop("accuracy")}, "configs": {}}
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    for name in ("strict", "primary", "relaxed"):
        cfg = frozen["configs"].get(name)
        if cfg is None:
            print(f"\n=== {name.upper()}: no passing scale, skipping ===")
            continue
        n_cal = cfg["n_cal"]
        h = torch.as_tensor(corr["h"][:n_cal], device=ad.device, dtype=ad.dtype)
        z0 = torch.as_tensor(corr["z0"][:n_cal], device=ad.device, dtype=ad.dtype)
        print(f"\n=== {name.upper()}  r={cfg['r_target']} lam={cfg['lambda']} "
              f"n_cal={n_cal}  budget {cfg['max_top1_loss_pp']} pp ===", flush=True)
        per_seed, corr_stack = {}, []
        for seed in seeds:
            reset_cuda()
            t0 = time.perf_counter()
            built = _build(ad, h, z0, Theta0, seed, cfg["r_target"], cfg["lambda"])
            t_build = time.perf_counter() - t0
            peak_build = torch.cuda.max_memory_allocated() / GIB
            pc_m, pc_p, pc_e = _evaluate(ad, Xval, yval, base_logits, built["members"],
                                         T, True)
            un_m, un_p, un_e = _evaluate(ad, Xval, yval, base_logits, built["members"],
                                         T, False)
            corr_stack.append((pc_p.argmax(-1) == yval).astype(np.float64))
            per_seed[str(seed)] = {
                "pnc": pc_m, "uncorrected": un_m, "build_s": t_build,
                "peak_gpu_build_gib": peak_build,
                "peak_gpu_eval_gib": torch.cuda.max_memory_allocated() / GIB,
                "realized_r_median": built["realized_r_median"],
                "calib_residual": built["calib_residual_median"],
                "all_finite": built["all_finite"]}
            np.savez_compressed(
                OUT / "predictions" / f"id_val_{name}_seed{seed}.npz",
                pnc_entropy=pc_e.astype(np.float32), unc_entropy=un_e.astype(np.float32),
                pnc_pred=pc_p.argmax(-1).astype(np.int16),
                unc_pred=un_p.argmax(-1).astype(np.int16),
                pnc_conf=pc_p.max(-1).astype(np.float32),
                unc_conf=un_p.max(-1).astype(np.float32),
                labels=yval.astype(np.int16))
            np.savez_compressed(
                OUT / "raw" / f"members_{name}_seed{seed}.npz",
                coefficients=np.stack([m["coeff"] for m in built["members"]]),
                W2=np.stack([m["W2"].cpu().numpy() for m in built["members"]]).astype(np.float32),
                b2=np.stack([m["b2"].cpu().numpy() for m in built["members"]]).astype(np.float32),
                scale=built["scale"], r_target=cfg["r_target"], lam=cfg["lambda"], seed=seed)
            print(f"  seed {seed:<5} build {t_build:.0f}s  P&C top-1 {pc_m['top1']*100:.3f}% "
                  f"(Δ{(pc_m['top1']-results['base']['top1'])*100:+.3f}pp)  "
                  f"unc {un_m['top1']*100:.3f}%  NLL {pc_m['nll']:.4f}  "
                  f"agree {pc_m['base_agreement']:.4f}", flush=True)
            for mem in built["members"]:
                del mem["W1v"], mem["W2"], mem["b2"]
            del built
            reset_cuda()

        # §16 validation: does the M=20, 5-seed ensemble still meet its budget?
        d = np.stack(corr_stack).mean(0) - base_correct
        boot = paired_bootstrap_lcb(d)
        eps = BUDGETS[name]
        ok = boot["lcb"] >= -eps
        agg = _agg(per_seed, seeds)
        results["configs"][name] = {
            "config": cfg, "per_seed": per_seed, "aggregate": agg,
            "budget_validation": {
                "budget_pp": eps * 100, "delta_top1_pp": boot["point"] * 100,
                "lcb_pp": boot["lcb"] * 100, "passes": bool(ok),
                "n_examples": boot["n_examples"], "n_replicates": boot["n_replicates"],
                "split": "official 50k validation, M=20, seed-averaged"},
        }
        print(f"  ---- budget check on 50k val: Δ{boot['point']*100:+.3f}pp  "
              f"LCB {boot['lcb']*100:+.3f}pp  vs budget -{eps*100:.2f}pp  "
              f"-> {'PASS' if ok else 'VIOLATION'}")
        del h, z0
        reset_cuda()

    fc.write_json(OUT / "metrics" / "id_final.json", results)
    print(f"\nwrote {OUT/'metrics'/'id_final.json'}")
    return results


def _agg(per_seed: dict, seeds) -> dict:
    out = {}
    for method in ("pnc", "uncorrected"):
        out[method] = {}
        keys = per_seed[str(seeds[0])][method]
        for k in keys:
            v = [per_seed[str(s)][method][k] for s in seeds]
            out[method][k] = {"mean": float(np.mean(v)),
                              "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                              "per_seed": v}
    return out


# ------------------------------------------------------------------ OOD
@torch.inference_mode()
def _entropy(ad, X, members, T, corrected):
    acc = None
    for mem in members:
        lg = fp.member_logits(ad, X, mem, corrected=corrected)
        p = torch.softmax(lg.double() / T, -1)
        acc = p if acc is None else acc + p
        del lg, p
    return fp.predictive_entropy((acc / len(members)).numpy())


def _load_members(ad, name, seed):
    z = np.load(OUT / "raw" / f"members_{name}_seed{seed}.npz")
    U = pc.perturbation_basis(int(seed), K)
    out = []
    for m in range(len(z["coefficients"])):
        dW1 = torch.as_tensor(pc.member_dW1(U, z["coefficients"][m], float(z["scale"])),
                              device=ad.device, dtype=ad.dtype)
        out.append({"W1v": ad.W1 + dW1,
                    "W2": torch.as_tensor(z["W2"][m], device=ad.device, dtype=ad.dtype),
                    "b2": torch.as_tensor(z["b2"][m], device=ad.device, dtype=ad.dtype)})
        del dW1
    return out


def run_ood(args):
    frozen = json.loads((OUT / "selection" / "frozen_configs.json").read_text())
    idf = json.loads((OUT / "metrics" / "id_final.json").read_text())
    T = frozen["temperature"]
    seeds = idf["seeds"]
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)

    val = fc.load_cache(SRC / "raw" / "cache_val50k.npz")
    Xval = fc.cache_to_gpu(val, ad)
    groups = {k: v[2] for k, v in fo.DATASETS.items()}

    # §18: reuse the deterministic baselines verbatim from the original run
    orig = json.loads((SRC / "metrics" / "ood_results.json").read_text())
    results = {"temperature": T, "seeds": seeds, "groups": groups,
               "reused_baselines_from": str(SRC / "metrics" / "ood_results.json"),
               "baselines": {m: orig["aggregate"][m] for m in
                             ("MSP", "Energy", "ReAct+Energy")},
               "baselines_per_dataset": {
                   ds: {m: orig["per_dataset"][ds][m]
                        for m in ("MSP", "Energy", "ReAct+Energy")}
                   for ds in orig["per_dataset"]},
               "per_dataset": {}, "aggregate": {}}

    id_scores, ood_scores = {}, {}
    for name in ("strict", "primary", "relaxed"):
        if frozen["configs"].get(name) is None:
            continue
        for seed in seeds:
            mem = _load_members(ad, name, seed)
            for tag, cor in ((f"P&C[{name}]", True), (f"Uncorrected[{name}]", False)):
                key = f"{tag}(seed{seed})"
                id_scores[key] = _entropy(ad, Xval, mem, T, cor)
            for mm in mem:
                del mm["W1v"], mm["W2"], mm["b2"]
            del mem
            reset_cuda()
        print(f"  {name}: ID scores done ({len(seeds)} seeds)", flush=True)

    for ds in fo.DATASETS:
        X = fc.cache_to_gpu(fc.load_cache(SRC / "raw" / f"cache_ood_{ds}.npz"), ad)
        print(f"\n-- {ds} ({X.shape[0]:,}, {groups[ds]}) --", flush=True)
        results["per_dataset"][ds] = {"group": groups[ds], "n_ood": int(X.shape[0])}
        for name in ("strict", "primary", "relaxed"):
            if frozen["configs"].get(name) is None:
                continue
            for seed in seeds:
                mem = _load_members(ad, name, seed)
                for tag, cor in ((f"P&C[{name}]", True), (f"Uncorrected[{name}]", False)):
                    key = f"{tag}(seed{seed})"
                    s = _entropy(ad, X, mem, T, cor)
                    ood_scores.setdefault(key, {})[ds] = s
                    results["per_dataset"][ds][key] = fm.binary_ood_metrics(
                        id_scores[key], s)
                for mm in mem:
                    del mm["W1v"], mm["W2"], mm["b2"]
                del mem
                reset_cuda()
            a = results["per_dataset"][ds][f"P&C[{name}](seed{seeds[0]})"]
            print(f"   P&C[{name}] seed{seeds[0]}  AUROC {a['auroc']*100:6.2f}  "
                  f"FPR95 {a['fpr95']*100:6.2f}", flush=True)
        del X
        reset_cuda()

    for key, per in ood_scores.items():
        agg = {}
        for g in ("near", "far"):
            fam = {k: v for k, v in per.items() if groups[k] == g}
            agg[g] = fm.aggregate_family_metrics(id_scores[key], fam)
        results["aggregate"][key] = agg
    _summarise(results, seeds, frozen)
    fc.write_json(OUT / "metrics" / "ood_results.json", results)
    for key, per in ood_scores.items():
        pass
    np.savez_compressed(OUT / "predictions" / "ood_scores.npz",
                        **{k.replace("&", "and").replace("[", "_").replace("]", ""):
                           np.concatenate([per[d] for d in sorted(per)]).astype(np.float32)
                           for k, per in ood_scores.items()},
                        dataset_order=np.array(sorted(fo.DATASETS), dtype=object))
    print(f"\nwrote {OUT/'metrics'/'ood_results.json'}")
    return results


def _summarise(results, seeds, frozen):
    summ = {}
    for name in ("strict", "primary", "relaxed"):
        if frozen["configs"].get(name) is None:
            continue
        for tag in (f"P&C[{name}]", f"Uncorrected[{name}]"):
            rows = [results["aggregate"][f"{tag}(seed{s})"] for s in seeds]
            summ[tag] = {f"{g}_{st}": {
                "mean": float(np.mean([r[g][st] for r in rows])),
                "std": float(np.std([r[g][st] for r in rows], ddof=1))
                if len(rows) > 1 else 0.0}
                for g in ("near", "far")
                for st in ("mean_auroc", "mean_fpr95", "mean_aupr", "mean_aupr_in")}
    for m in ("MSP", "Energy", "ReAct+Energy"):
        summ[m] = {f"{g}_{st}": {"mean": results["baselines"][m][g][st], "std": 0.0}
                   for g in ("near", "far")
                   for st in ("mean_auroc", "mean_fpr95", "mean_aupr", "mean_aupr_in")}
    results["summary"] = summ
    print("\n=== Near / Far summary ===")
    print(f"  {'method':<24}{'Near AUROC':>12}{'Near FPR95':>12}{'Far AUROC':>12}{'Far FPR95':>12}")
    for k, v in summ.items():
        print(f"  {k:<24}{v['near_mean_auroc']['mean']*100:>11.2f} "
              f"{v['near_mean_fpr95']['mean']*100:>11.2f} "
              f"{v['far_mean_auroc']['mean']*100:>11.2f} "
              f"{v['far_mean_fpr95']['mean']*100:>11.2f}")
