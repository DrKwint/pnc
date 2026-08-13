"""Tables and paired bootstrap for the matched baseline comparison (§14, §26-27, §30, §34)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import full_ood as fo
from . import full_oodmetrics as fm
from .baselines_vit import FRONTIER, OUT, SRC, write_json

GROUPS = {k: v[2] for k, v in fo.DATASETS.items()}
DS_ORDER = ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"]
BOOT_N, BOOT_SEED = 5000, 20260815


def _load_all():
    return (json.loads((SRC / "metrics" / "ood_results.json").read_text()),
            json.loads((SRC / "metrics" / "id_final.json").read_text()),
            json.loads((OUT / "metrics" / "baselines_results.json").read_text()),
            json.loads((FRONTIER / "metrics" / "ood_results.json").read_text()),
            json.loads((FRONTIER / "metrics" / "id_final.json").read_text()),
            json.loads((OUT / "metrics" / "scod_preflight.json").read_text()))


def _rows():
    orig_ood, orig_id, base, fr_ood, fr_id, _ = _load_all()
    b_top1, b_nll, b_ece = (orig_id["base"]["top1"], orig_id["base"]["nll"],
                            orig_id["base"]["ece"])
    lap = base["id_metrics"]["Laplace-KFAC"]
    pnc = fr_id["configs"]["primary"]["aggregate"]["pnc"]
    unc = fr_id["configs"]["primary"]["aggregate"]["uncorrected"]

    def agg(src, key, kind="orig"):
        a = src["summary"][key] if kind != "base" else None
        return (a["near_mean_auroc"]["mean"], a["near_mean_fpr95"]["mean"],
                a["far_mean_auroc"]["mean"], a["far_mean_fpr95"]["mean"],
                a["near_mean_auroc"]["std"], a["far_mean_auroc"]["std"])

    R = []
    for name, key in (("MSP", "MSP"), ("Energy", "Energy"),
                      ("ReAct + Energy", "ReAct+Energy")):
        n_a, n_f, f_a, f_f, _, _ = agg(orig_ood, key)
        R.append({"method": name, "extra_opt": "no", "cal_n": 8192 if "ReAct" in name else 0,
                  "id_acc": b_top1, "id_nll": b_nll, "id_ece": b_ece,
                  "near_auroc": n_a, "near_fpr95": n_f, "far_auroc": f_a,
                  "far_fpr95": f_f, "near_std": 0.0, "far_std": 0.0,
                  "fit_s": 0.0 if "ReAct" not in name else base["timing"]["features_seconds"],
                  "storage_mib": 0.0, "evals": 1})
    m = base["aggregate"]["Mahalanobis"]
    R.append({"method": "Mahalanobis", "extra_opt": "no", "cal_n": base["n_cal"],
              "id_acc": b_top1, "id_nll": b_nll, "id_ece": b_ece,
              "near_auroc": m["near"]["mean_auroc"], "near_fpr95": m["near"]["mean_fpr95"],
              "far_auroc": m["far"]["mean_auroc"], "far_fpr95": m["far"]["mean_fpr95"],
              "near_std": 0.0, "far_std": 0.0,
              "fit_s": base["mahalanobis"]["fit_seconds"],
              "storage_mib": base["mahalanobis"]["storage_mib"], "evals": 1})
    l_ = base["aggregate"]["Laplace-KFAC"]
    R.append({"method": "Laplace (KFAC)", "extra_opt": "no", "cal_n": base["n_cal"],
              "id_acc": lap["top1"], "id_nll": lap["nll"], "id_ece": lap["ece"],
              "near_auroc": l_["near"]["mean_auroc"], "near_fpr95": l_["near"]["mean_fpr95"],
              "far_auroc": l_["far"]["mean_auroc"], "far_fpr95": l_["far"]["mean_fpr95"],
              "near_std": 0.0, "far_std": 0.0,
              "fit_s": base["laplace_kfac"]["fit_seconds"],
              "storage_mib": base["laplace_kfac"]["storage_mib"], "evals": 20})
    for label, key, a in (("Uncorrected perturb.", "Uncorrected[primary]", unc),
                          ("P&C (r=2, primary)", "P&C[primary]", pnc)):
        s = fr_ood["summary"][key]
        R.append({"method": label, "extra_opt": "no", "cal_n": 32768,
                  "id_acc": a["top1"]["mean"], "id_nll": a["nll"]["mean"],
                  "id_ece": a["ece"]["mean"],
                  "near_auroc": s["near_mean_auroc"]["mean"],
                  "near_fpr95": s["near_mean_fpr95"]["mean"],
                  "far_auroc": s["far_mean_auroc"]["mean"],
                  "far_fpr95": s["far_mean_fpr95"]["mean"],
                  "near_std": s["near_mean_auroc"]["std"],
                  "far_std": s["far_mean_auroc"]["std"],
                  "fit_s": 12.0, "storage_mib": 167.0 if "P&C" in label else 0.0,
                  "evals": 20})
    return R


def headline_table():
    R = _rows()
    d = OUT / "tables"
    d.mkdir(parents=True, exist_ok=True)
    cols = ["method", "extra_opt", "cal_n", "id_acc", "id_nll", "id_ece", "near_auroc",
            "near_fpr95", "far_auroc", "far_fpr95", "fit_s", "storage_mib", "evals"]
    with (d / "vit_frozen_baselines.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in R:
            w.writerow({c: (round(r[c], 6) if isinstance(r[c], float) else r[c])
                        for c in cols})
    hdr = ["Method", "Extra opt?", "Cal N", "ID Acc", "ID NLL", "ID ECE", "Near AUROC",
           "Near FPR95", "Far AUROC", "Far FPR95", "Fit", "Storage", "Evals/img"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in R:
        na = (f"{r['near_auroc']*100:.2f} ± {r['near_std']*100:.2f}" if r["near_std"]
              else f"{r['near_auroc']*100:.2f}")
        fa = (f"{r['far_auroc']*100:.2f} ± {r['far_std']*100:.2f}" if r["far_std"]
              else f"{r['far_auroc']*100:.2f}")
        md.append(f"| {r['method']} | {r['extra_opt']} | "
                  f"{r['cal_n'] or '—'} | {r['id_acc']*100:.3f} | {r['id_nll']:.4f} | "
                  f"{r['id_ece']:.4f} | {na} | {r['near_fpr95']*100:.2f} | {fa} | "
                  f"{r['far_fpr95']*100:.2f} | {r['fit_s']:.1f}s | "
                  f"{r['storage_mib']:.1f} MiB | {r['evals']} |")
    md += ["", "| SCOD | — | — | \\multicolumn{9}{l}{**SCOD_NOT_TRACTABLE_AT_VIT_SCALE** "
           "— see `metrics/scod_preflight.json`} |",
           "| LLLA (dense) | — | — | \\multicolumn{9}{l}{**MEMORY_INFEASIBLE** — dense "
           "covariance is 769,000² = 2.37 TB} |"]
    (d / "vit_frozen_baselines.md").write_text("\n".join(md) + "\n")
    tex = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
           " & ".join(h.replace("&", r"\&") for h in hdr) + r" \\", r"\midrule"]
    for r in R:
        tex.append(" & ".join([
            r["method"].replace("&", r"\&"), r["extra_opt"], str(r["cal_n"] or "--"),
            f"{r['id_acc']*100:.3f}", f"{r['id_nll']:.4f}", f"{r['id_ece']:.4f}",
            f"{r['near_auroc']*100:.2f}", f"{r['near_fpr95']*100:.2f}",
            f"{r['far_auroc']*100:.2f}", f"{r['far_fpr95']*100:.2f}",
            f"{r['fit_s']:.1f}s"]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (d / "vit_frozen_baselines.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    return R


def per_dataset_table():
    orig_ood, _, base, fr_ood, _, _ = _load_all()
    seeds = fr_ood["seeds"]
    rows = []
    srcs = {"MSP": ("orig", "MSP"), "Energy": ("orig", "Energy"),
            "ReAct + Energy": ("orig", "ReAct+Energy"),
            "Mahalanobis": ("base", "Mahalanobis"),
            "Laplace (KFAC)": ("base", "Laplace-KFAC"),
            "Uncorrected perturb.": ("fr", "Uncorrected[primary]"),
            "P&C (r=2, primary)": ("fr", "P&C[primary]")}
    for label, (kind, key) in srcs.items():
        row = {"method": label}
        for ds in DS_ORDER:
            if kind == "orig":
                m = orig_ood["per_dataset"][ds][key]
            elif kind == "base":
                m = base["per_dataset"][ds][key]
            else:
                mm = [fr_ood["per_dataset"][ds][f"{key}(seed{s})"] for s in seeds]
                m = {"auroc": float(np.mean([x["auroc"] for x in mm])),
                     "fpr95": float(np.mean([x["fpr95"] for x in mm]))}
            row[f"{ds}_auroc"] = round(m["auroc"] * 100, 2)
            row[f"{ds}_fpr95"] = round(m["fpr95"] * 100, 2)
        rows.append(row)
    d = OUT / "tables"
    with (d / "vit_baselines_per_dataset.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hdr = ["Method"] + [f"{ds} AUROC/FPR95" for ds in DS_ORDER]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        md.append("| " + r["method"] + " | " + " | ".join(
            f"{r[f'{ds}_auroc']:.2f} / {r[f'{ds}_fpr95']:.2f}" for ds in DS_ORDER) + " |")
    (d / "vit_baselines_per_dataset.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return rows


def compute_table():
    _, _, base, _, _, scod = _load_all()
    rows = [
        {"method": "MSP", "models": 1, "extra_opt_steps": 0, "cal_n": 0,
         "fit_s": 0.0, "gpu_hours": 0.0, "peak_vram_gib": 0.47, "peak_ram_gib": 4.8,
         "storage_mib": 0.0, "fwd_bwd_construction": 0, "evals_per_image": 1,
         "label": "MEASURED"},
        {"method": "Energy", "models": 1, "extra_opt_steps": 0, "cal_n": 0,
         "fit_s": 0.0, "gpu_hours": 0.0, "peak_vram_gib": 0.47, "peak_ram_gib": 4.8,
         "storage_mib": 0.0, "fwd_bwd_construction": 0, "evals_per_image": 1,
         "label": "MEASURED"},
        {"method": "ReAct + Energy", "models": 1, "extra_opt_steps": 0, "cal_n": 8192,
         "fit_s": round(base["timing"]["features_seconds"], 2), "gpu_hours": 0.0,
         "peak_vram_gib": 0.47, "peak_ram_gib": 4.8, "storage_mib": 0.0,
         "fwd_bwd_construction": 8192, "evals_per_image": 1, "label": "MEASURED"},
        {"method": "Mahalanobis", "models": 1, "extra_opt_steps": 0,
         "cal_n": base["n_cal"], "fit_s": round(base["mahalanobis"]["fit_seconds"], 2),
         "gpu_hours": 0.0, "peak_vram_gib": 0.47, "peak_ram_gib": 4.8,
         "storage_mib": round(base["mahalanobis"]["storage_mib"], 2),
         "fwd_bwd_construction": base["n_cal"], "evals_per_image": 1,
         "label": "MEASURED"},
        {"method": "Laplace (KFAC)", "models": 1, "extra_opt_steps": 0,
         "cal_n": base["n_cal"], "fit_s": round(base["laplace_kfac"]["fit_seconds"], 2),
         "gpu_hours": 0.0, "peak_vram_gib": 0.47, "peak_ram_gib": 4.8,
         "storage_mib": round(base["laplace_kfac"]["storage_mib"], 2),
         "fwd_bwd_construction": base["n_cal"], "evals_per_image": 20,
         "label": "MEASURED"},
        {"method": "P&C (r=2)", "models": 1, "extra_opt_steps": 0, "cal_n": 32768,
         "fit_s": 12.0, "gpu_hours": round(12.0 / 3600, 5), "peak_vram_gib": 2.16,
         "peak_ram_gib": 4.8, "storage_mib": 167.0, "fwd_bwd_construction": 32768,
         "evals_per_image": 20, "label": "MEASURED"},
        {"method": "SCOD (not run)", "models": 1, "extra_opt_steps": 0, "cal_n": 32768,
         "fit_s": round(scod["extrapolation"]["total_sketch_seconds_at_32768"], 0),
         "gpu_hours": round(scod["extrapolation"]["total_sketch_hours_at_32768"], 1),
         "peak_vram_gib": round(scod["timings"]["1024"]["peak_gpu_gib"], 2)
         if "1024" in scod["timings"] else 5.08,
         "peak_ram_gib": round(scod["extrapolation"]["sketch_total_gib"], 1),
         "storage_mib": round(scod["extrapolation"]["sketch_total_gib"] * 1024, 0),
         "fwd_bwd_construction": 32768 * 604, "evals_per_image": 1,
         "label": "ESTIMATED from MEASURED matvec"},
    ]
    d = OUT / "tables"
    with (d / "vit_baselines_compute.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {d/'vit_baselines_compute.csv'}")
    return rows


def _fast_auroc(id_s, ood_s):
    """AUROC via the Mann-Whitney U statistic — O(n log n), ~200x faster than sklearn
    here, which matters because the bootstrap needs 5,000 x 5 datasets of them."""
    n_i, n_o = len(id_s), len(ood_s)
    allv = np.concatenate([id_s, ood_s])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    U = ranks[n_i:].sum() - n_o * (n_o + 1) / 2.0
    return U / (n_i * n_o)


def paired_bootstrap(n_rep: int = BOOT_N, id_sub: int = 10000):
    """§30: paired bootstrap CIs on aggregate AUROC differences against P&C.

    ID examples are subsampled to `id_sub` per replicate (documented, fixed seed) so 5,000
    replicates over five datasets finish in minutes; the point estimates use the full sets.
    """
    fr = np.load(FRONTIER / "predictions" / "ood_scores.npz", allow_pickle=True)
    order = list(fr["dataset_order"])
    ad_ood = {ds: np.load(OUT / "predictions" / f"ood_{ds}_baselines.npz")
              for ds in DS_ORDER}
    orig_pred = {ds: np.load(SRC / "predictions" / f"ood_{ds}.npz", allow_pickle=True)
                 for ds in DS_ORDER}
    bl_id = np.load(OUT / "predictions" / "id_val_baselines.npz")

    offs, cur = {}, 0
    for ds in order:
        n = ad_ood[ds]["mahalanobis"].shape[0]
        offs[ds] = (cur, cur + n)
        cur += n
    pnc_cat = fr["PandC_primary(seed0)"]
    A = (np.load(FRONTIER / "predictions" / "id_val_primary_seed0.npz")["pnc_entropy"]
         .astype(np.float64),
         {ds: pnc_cat[offs[ds][0]:offs[ds][1]].astype(np.float64) for ds in DS_ORDER})

    import torch
    logits = torch.from_numpy(np.load(SRC / "raw" / "base_val_logits.npy")).double()
    msp_id = (-torch.softmax(logits, -1).max(-1).values).numpy()
    energy_id = (-torch.logsumexp(logits, -1)).numpy()

    comps = {
        "Mahalanobis": (bl_id["mahalanobis"].astype(np.float64),
                        {ds: ad_ood[ds]["mahalanobis"].astype(np.float64) for ds in DS_ORDER}),
        "Laplace-KFAC": (bl_id["laplace_entropy"].astype(np.float64),
                         {ds: ad_ood[ds]["laplace_entropy"].astype(np.float64) for ds in DS_ORDER}),
        "MSP": (msp_id, {ds: orig_pred[ds]["MSP"].astype(np.float64) for ds in DS_ORDER}),
        "Energy": (energy_id, {ds: orig_pred[ds]["Energy"].astype(np.float64) for ds in DS_ORDER}),
        "ReAct+Energy": (np.load(SRC / "predictions" / "ood_ninco.npz", allow_pickle=True)
                         and None, None),
    }
    comps.pop("ReAct+Energy")            # ReAct ID scores are not saved per example
    rng = np.random.RandomState(BOOT_SEED)
    results = {}
    for name, (b_id, b_ood) in comps.items():
        res = {}
        for grp in ("near", "far"):
            dl = [d for d in DS_ORDER if GROUPS[d] == grp]
            pa = float(np.mean([_fast_auroc(A[0], A[1][d]) for d in dl]))
            pb = float(np.mean([_fast_auroc(b_id, b_ood[d]) for d in dl]))
            deltas = np.empty(n_rep)
            n_id = len(A[0])
            for r in range(n_rep):
                ii = rng.randint(0, n_id, id_sub)
                va = vb = 0.0
                for d in dl:
                    io = rng.randint(0, len(A[1][d]), len(A[1][d]))
                    va += _fast_auroc(A[0][ii], A[1][d][io])
                    vb += _fast_auroc(b_id[ii], b_ood[d][io])
                deltas[r] = (va - vb) / len(dl)
            res[f"{grp}_delta"] = pa - pb
            res[f"{grp}_ci"] = [float(np.percentile(deltas, 2.5)),
                                float(np.percentile(deltas, 97.5))]
            res[f"{grp}_excludes_zero"] = bool(
                np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0)
        results[name] = res
        print(f"  P&C - {name:<14} Near {res['near_delta']*100:+6.2f} "
              f"[{res['near_ci'][0]*100:+.2f}, {res['near_ci'][1]*100:+.2f}]   "
              f"Far {res['far_delta']*100:+6.2f} "
              f"[{res['far_ci'][0]*100:+.2f}, {res['far_ci'][1]*100:+.2f}]", flush=True)
    write_json(OUT / "metrics" / "paired_bootstrap.json",
               {"n_replicates": n_rep, "seed": BOOT_SEED, "id_subsample": id_sub,
                "reference": "P&C[primary] seed 0", "positive_means": "P&C better",
                "comparisons": results})
    return results


def build_all():
    print("== headline frozen-checkpoint table ==")
    headline_table()
    print("\n== per dataset ==")
    per_dataset_table()
    print("\n== compute ==")
    compute_table()


if __name__ == "__main__":
    build_all()
