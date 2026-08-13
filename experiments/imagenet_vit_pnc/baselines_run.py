"""Runner for the matched baseline comparison (spec §7, §11-13, §22).

    parity     §7 common-harness gate against the completed ViT experiment
    fit        Mahalanobis + Laplace-KFAC: ID-only selection, then full evaluation
    scod       §13 measured SCOD feasibility preflight
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.banking77_pnc.pnc_metrics import clf_metrics

from . import baselines_vit as bl
from . import full_baselines as fb
from . import full_cache as fc
from . import full_ood as fo
from . import full_oodmetrics as fm
from .baselines_vit import FRONTIER, OUT, SRC
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter

DATASETS = list(fo.DATASETS)
GROUPS = {k: v[2] for k, v in fo.DATASETS.items()}


def _load(adapter, name: str):
    return fc.cache_to_gpu(fc.load_cache(SRC / "raw" / name), adapter)


# ------------------------------------------------------------------ §7 parity
def stage_parity(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    ref = json.loads((SRC / "metrics" / "base_id_metrics.json").read_text())
    ref_ood = json.loads((SRC / "metrics" / "ood_results.json").read_text())
    T = json.loads((SRC / "metrics" / "temperature.json").read_text())["temperature"]

    val = fc.load_cache(SRC / "raw" / "cache_val50k.npz")
    X = fc.cache_to_gpu(val, ad)
    y = val["labels"]
    logits = fc.logits_from_cache(ad, X)
    probs = torch.softmax(logits.double(), -1).numpy()
    m = clf_metrics(probs, y)
    top5 = logits.topk(5, -1).indices.numpy()
    got = {"top1": m["accuracy"], "top5": float((top5 == y[:, None]).any(-1).mean()),
           "nll": m["nll"], "ece": m["ece"]}
    checks = [(k, got[k], ref[k], abs(got[k] - ref[k]) < 1e-6) for k in
              ("top1", "top5", "nll", "ece")]

    tempc = _load(ad, "cache_temperature.npz")
    react_c = fb.react_threshold(fc.features_from_cache(ad, tempc), percentile=90.0)
    del tempc
    reset_cuda()
    ok_react = abs(react_c - ref_ood["react_threshold"]) < 1e-6
    checks.append(("react_threshold", react_c, ref_ood["react_threshold"], ok_react))

    b = fb.base_scores(ad, X, react_c=react_c)
    for meth, key in (("MSP", "msp"), ("Energy", "energy")):
        fam = {g: {} for g in ("near", "far")}
        for ds in DATASETS:
            Xo = _load(ad, f"cache_ood_{ds}.npz")
            fam[GROUPS[ds]][ds] = fb.base_scores(ad, Xo, react_c=None)[key]
            del Xo
            reset_cuda()
        for g in ("near", "far"):
            got_a = fm.aggregate_family_metrics(b[key], fam[g])["mean_auroc"]
            ref_a = ref_ood["aggregate"][meth][g]["mean_auroc"]
            checks.append((f"{meth}_{g}_auroc", got_a, ref_a, abs(got_a - ref_a) < 1e-9))

    print(f"{'check':<26}{'harness':>14}{'reference':>14}   ok")
    for name, g, r, ok in checks:
        print(f"{name:<26}{g:>14.8f}{r:>14.8f}   {'OK' if ok else 'MISMATCH'}")
    n_bad = sum(1 for *_, ok in checks if not ok)
    bl.write_json(OUT / "provenance" / "parity_gate.json",
                  {"checks": [{"name": n, "harness": g, "reference": r, "ok": bool(o)}
                              for n, g, r, o in checks], "n_failed": n_bad})
    print(f"\n{len(checks)-n_bad}/{len(checks)} parity checks passed")
    if n_bad:
        raise SystemExit("common harness does not reproduce the completed experiment (§7)")


# ------------------------------------------------------- §11-12 fit + evaluate
def stage_fit(args):
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    T = json.loads((SRC / "metrics" / "temperature.json").read_text())["temperature"]

    corr = fc.load_cache(SRC / "raw" / "cache_correction.npz")
    Xc = fc.cache_to_gpu(corr, ad)
    y_cal = corr["labels"]
    t0 = time.perf_counter()
    feats_cal = fc.features_from_cache(ad, Xc)
    t_feat = time.perf_counter() - t0
    print(f"calibration features: {tuple(feats_cal.shape)} in {t_feat:.1f}s")
    del Xc
    reset_cuda()

    sel = fc.load_cache(SRC / "raw" / "cache_selection.npz")
    Xs = fc.cache_to_gpu(sel, ad)
    feats_sel = fc.features_from_cache(ad, Xs).to(ad.device)
    y_sel = sel["labels"]
    base_sel = ad.head(feats_sel).cpu()
    base_sel_pred = base_sel.argmax(-1).numpy()
    del Xs
    reset_cuda()

    results = {"temperature": T, "M_samples": bl.M_SAMPLES, "n_cal": int(len(y_cal))}

    # ---- Mahalanobis: no hyperparameters, ID-only, fit once ----
    print("\n== Mahalanobis (single-layer class-conditional Gaussian) ==")
    reset_cuda()
    t0 = time.perf_counter()
    means, prec = bl.fit_mahalanobis(feats_cal.numpy(), y_cal, 1000)
    t_maha = time.perf_counter() - t0
    print(f"  fitted in {t_maha:.1f}s  (means {means.shape}, precision {prec.shape})")
    np.savez_compressed(OUT / "raw" / "mahalanobis_fit.npz",
                        class_means=means.astype(np.float32),
                        precision=prec.astype(np.float32))
    results["mahalanobis"] = {
        "fit_seconds": t_maha, "feature_dim": int(means.shape[1]),
        "storage_mib": (means.nbytes + prec.nbytes) / 1024**2 / 2,   # float32 on disk
        "hyperparameters": {"covariance_ridge": 1e-6},
        "selection": "none — no tunable hyperparameter",
        "peak_cpu_gib": cpu_peak_rss_gib()}

    # ---- Laplace-KFAC: prior precision chosen by ID-selection-pool NLL ----
    print("\n== Laplace (KFAC last layer) ==")
    t0 = time.perf_counter()
    A, S, N = bl.kfac_factors(ad, feats_cal.to(ad.device))
    t_kfac = time.perf_counter() - t0
    print(f"  KFAC factors A{A.shape} S{S.shape} from N={N} in {t_kfac:.1f}s")

    rows = []
    for lam in bl.PRIOR_PRECISION_GRID:
        h = bl.LaplaceKFACHead(ad, A, S, lam, N, n_models=bl.M_SAMPLES, seed=0)
        p = h.predict_probs(feats_sel, T=T)
        m = bl.id_metrics(p, y_sel, base_sel_pred)
        rows.append({"prior_precision": lam, **m})
        print(f"  lam={lam:<9g} top1 {m['top1']*100:7.3f}  NLL {m['nll']:.4f}  "
              f"ECE {m['ece']:.4f}  agree {m['base_agreement']:.4f}")
        del h, p
    best = min(rows, key=lambda r: r["nll"])
    print(f"  -> selected prior_precision={best['prior_precision']:g} "
          f"(ID-selection-pool NLL {best['nll']:.4f})")

    (OUT / "id_selection").mkdir(parents=True, exist_ok=True)
    with (OUT / "id_selection" / "laplace_kfac_selection.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bl.write_json(OUT / "id_selection" / "laplace_kfac_selected.json", {
        "method": "Laplace (KFAC, last layer)",
        "source": "pnc_core/ensembles.py::LaplaceEnsemble + pnc_core/laplace.py::"
                  "compute_kfac_factors(is_classification=True)",
        "prior_precision": best["prior_precision"], "data_size": N,
        "n_samples": bl.M_SAMPLES, "selection_criterion": "ID selection-pool NLL",
        "selection_pool": "8,192 ImageNet train images (disjoint from calibration)",
        "grid": bl.PRIOR_PRECISION_GRID,
        "temperature": T, "temperature_fit_data": "ID temperature pool (reused)",
        "OOD data accessed before selection": "NO",
        "kfac_seconds": t_kfac})
    np.savez_compressed(OUT / "raw" / "laplace_kfac_factors.npz",
                        A=A.astype(np.float32), S=S.astype(np.float32), N=N)
    results["laplace_kfac"] = {
        "fit_seconds": t_kfac, "prior_precision": best["prior_precision"],
        "data_size": N, "n_samples": bl.M_SAMPLES,
        "storage_mib": (A.astype(np.float32).nbytes + S.astype(np.float32).nbytes)
        / 1024**2,
        "selection_rows": rows}
    del feats_cal, feats_sel
    reset_cuda()

    # ---- full evaluation on ID + all OOD sets ----
    print("\n== evaluating on 50k ID + 85,908 OOD ==")
    val = fc.load_cache(SRC / "raw" / "cache_val50k.npz")
    Xv = fc.cache_to_gpu(val, ad)
    yv = val["labels"]
    fv = fc.features_from_cache(ad, Xv).to(ad.device)
    base_logits = ad.head(fv).cpu()
    base_pred = base_logits.argmax(-1).numpy()

    head = bl.LaplaceKFACHead(ad, A, S, best["prior_precision"], N,
                              n_models=bl.M_SAMPLES, seed=0)
    t0 = time.perf_counter()
    p_lap = head.predict_probs(fv, T=T)
    t_lap_id = time.perf_counter() - t0
    id_lap = bl.id_metrics(p_lap, yv, base_pred)
    print(f"  Laplace-KFAC  top1 {id_lap['top1']*100:.3f}%  NLL {id_lap['nll']:.4f}  "
          f"ECE {id_lap['ece']:.4f}  agree {id_lap['base_agreement']:.4f}  "
          f"({t_lap_id:.0f}s for 50k x M={bl.M_SAMPLES})")

    fnp = fv.cpu().numpy()
    t0 = time.perf_counter()
    id_maha = bl.mahalanobis_scores(fnp, means, prec)
    t_maha_id = time.perf_counter() - t0
    id_scores = {"Mahalanobis": id_maha,
                 "Laplace-KFAC": bl.predictive_entropy(p_lap)}
    np.savez_compressed(OUT / "predictions" / "id_val_baselines.npz",
                        mahalanobis=id_maha.astype(np.float32),
                        laplace_entropy=id_scores["Laplace-KFAC"].astype(np.float32),
                        laplace_pred=p_lap.argmax(-1).astype(np.int16),
                        laplace_conf=p_lap.max(-1).astype(np.float32),
                        labels=yv.astype(np.int16))
    del Xv, p_lap
    reset_cuda()

    ood_scores = {k: {} for k in id_scores}
    per_ds = {}
    for ds in DATASETS:
        Xo = _load(ad, f"cache_ood_{ds}.npz")
        fo_ = fc.features_from_cache(ad, Xo).to(ad.device)
        ood_scores["Mahalanobis"][ds] = bl.mahalanobis_scores(fo_.cpu().numpy(),
                                                              means, prec)
        po = head.predict_probs(fo_, T=T)
        ood_scores["Laplace-KFAC"][ds] = bl.predictive_entropy(po)
        per_ds[ds] = {"group": GROUPS[ds], "n_ood": int(fo_.shape[0])}
        for meth in id_scores:
            per_ds[ds][meth] = fm.binary_ood_metrics(id_scores[meth],
                                                     ood_scores[meth][ds])
        print(f"  {ds:<13} Maha AUROC {per_ds[ds]['Mahalanobis']['auroc']*100:6.2f}  "
              f"Laplace AUROC {per_ds[ds]['Laplace-KFAC']['auroc']*100:6.2f}")
        np.savez_compressed(OUT / "predictions" / f"ood_{ds}_baselines.npz",
                            mahalanobis=ood_scores["Mahalanobis"][ds].astype(np.float32),
                            laplace_entropy=ood_scores["Laplace-KFAC"][ds].astype(np.float32))
        del Xo, fo_, po
        reset_cuda()

    agg = {}
    for meth in id_scores:
        agg[meth] = {g: fm.aggregate_family_metrics(
            id_scores[meth], {k: v for k, v in ood_scores[meth].items()
                              if GROUPS[k] == g}) for g in ("near", "far")}
    results["id_metrics"] = {"Laplace-KFAC": id_lap,
                             "Mahalanobis": {"note": "pure OOD score; ID predictions are "
                                             "the base model's", **{k: None for k in ()}}}
    results["per_dataset"] = per_ds
    results["aggregate"] = agg
    results["timing"] = {"features_seconds": t_feat, "mahalanobis_fit_seconds": t_maha,
                         "kfac_seconds": t_kfac,
                         "mahalanobis_id_score_seconds": t_maha_id,
                         "laplace_id_predict_seconds": t_lap_id}
    bl.write_json(OUT / "metrics" / "baselines_results.json", results)
    print("\n=== Near / Far ===")
    for meth, a in agg.items():
        print(f"  {meth:<14} Near AUROC {a['near']['mean_auroc']*100:6.2f}  "
              f"FPR95 {a['near']['mean_fpr95']*100:6.2f}   "
              f"Far AUROC {a['far']['mean_auroc']*100:6.2f}  "
              f"FPR95 {a['far']['mean_fpr95']*100:6.2f}")
    print(f"\nwrote {OUT/'metrics'/'baselines_results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["parity", "fit", "scod"])
    args = ap.parse_args()
    if args.stage == "parity":
        stage_parity(args)
    elif args.stage == "fit":
        stage_fit(args)
    else:
        from .scod_preflight import run
        run()


if __name__ == "__main__":
    main()
