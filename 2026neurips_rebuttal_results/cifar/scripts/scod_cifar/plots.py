"""Stage 5 -- figures (Section 19). Generated from saved artifacts (parquet + metrics + spectra).

Design: CVD-safe Okabe-Ito categorical palette in fixed order, thin marks, recessive grid, direct
labels/legends (identity never color-alone), one axis per chart. Plots that need data not produced
by this pipeline (e.g. P&C per-example scores, sketch-seed sensitivity when only the primary seed
was built) are skipped with a printed note rather than fabricated.
"""
from __future__ import annotations

import argparse, json, glob
from pathlib import Path
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 11, "legend.frameon": False})


def _cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


def _load_pred(root, seed, sketch_seed, name):
    import pandas as pd
    f = root / "predictions" / f"seed_{seed}" / f"sketch_{sketch_seed}" / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    args = ap.parse_args(); cfg = _cfg(args.config)
    root = REPO / cfg["root"]; pdir = root / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    seeds = cfg["model_seeds"]; base = int(cfg["sketch_seed_base_primary"])
    s0 = seeds[0]; ss0 = base + s0
    near, far = list(cfg["near_ood"]), list(cfg["far_ood"])
    allds = ["cifar10"] + near + far
    made = []

    # 1 & 2: score distributions (seed0, k10 tempered)
    dfs = {d: _load_pred(root, s0, ss0, d) for d in allds}
    if all(v is not None for v in dfs.values()):
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, d in enumerate(allds):
            s = dfs[d]["scod_score_k10"].to_numpy()
            ax.hist(s, bins=60, histtype="step", lw=1.6, color=OKABE[i % len(OKABE)],
                    density=True, label=d)
        ax.set_xlabel("SCOD score (k=10)"); ax.set_ylabel("density")
        ax.set_title("SCOD score distribution by dataset (seed 0)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(pdir / "score_distribution_by_dataset.png"); plt.close(fig)
        made.append("score_distribution_by_dataset.png")

        # by regime
        import pandas as pd
        reg = {"ID": dfs["cifar10"]["scod_score_k10"].to_numpy(),
               "Near": np.concatenate([dfs[d]["scod_score_k10"].to_numpy() for d in near]),
               "Far": np.concatenate([dfs[d]["scod_score_k10"].to_numpy() for d in far])}
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (r, s) in enumerate(reg.items()):
            ax.hist(s, bins=60, histtype="step", lw=1.8, density=True, color=OKABE[i], label=r)
        ax.set_xlabel("SCOD score (k=10)"); ax.set_ylabel("density")
        ax.set_title("SCOD score by regime (seed 0)"); ax.legend()
        fig.tight_layout(); fig.savefig(pdir / "score_distribution_by_regime.png"); plt.close(fig)
        made.append("score_distribution_by_regime.png")

        # 10: SCOD vs energy scatter (Far pooled)
        fig, ax = plt.subplots(figsize=(5, 5))
        idd = dfs["cifar10"]; farpool = pd.concat([dfs[d] for d in far])
        ax.scatter(idd["energy_score"], idd["scod_score_k10"], s=4, alpha=0.3, color=OKABE[0], label="ID")
        ax.scatter(farpool["energy_score"], farpool["scod_score_k10"], s=4, alpha=0.3,
                   color=OKABE[3], label="Far-OOD")
        ax.set_xlabel("energy score"); ax.set_ylabel("SCOD score (k=10)")
        ax.set_title("SCOD vs Energy (seed 0)"); ax.legend()
        fig.tight_layout(); fig.savefig(pdir / "scod_vs_energy_scatter.png"); plt.close(fig)
        made.append("scod_vs_energy_scatter.png")

    # 3 & 4: Near/Far AUROC & FPR95 by method (from aggregate + submitted comparators)
    aggf = root / "tables" / "scod_aggregate.json"
    if aggf.exists():
        from experiments.scod_cifar.aggregate import SUBMITTED
        agg = json.load(open(aggf))
        methods = [m[0] for m in SUBMITTED] + ["SCOD-1024"]
        near_au = [m[2] for m in SUBMITTED] + [agg["near"]["auroc"][0]]
        far_au = [m[4] for m in SUBMITTED] + [agg["far"]["auroc"][0]]
        near_fp = [m[3] for m in SUBMITTED] + [agg["near"]["fpr95"][0]]
        far_fp = [m[5] for m in SUBMITTED] + [agg["far"]["fpr95"][0]]
        for metric, nn, ff, fname, lo in [("AUROC", near_au, far_au, "near_far_auroc_by_method.png", 80),
                                          ("FPR95", near_fp, far_fp, "near_far_fpr95_by_method.png", 0)]:
            fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(methods)); w = 0.4
            ax.bar(x - w/2, nn, w, color=OKABE[0], label="Near")
            ax.bar(x + w/2, ff, w, color=OKABE[1], label="Far")
            ax.set_xticks(x); ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(metric); ax.set_title(f"Near/Far {metric} by method"); ax.legend()
            if metric == "AUROC": ax.set_ylim(lo, 100)
            for xi, (a, b) in enumerate(zip(nn, ff)):
                if methods[xi] == "SCOD-1024":
                    ax.text(xi - w/2, a + 0.3, "SCOD", ha="center", fontsize=7, color=OKABE[7])
            fig.tight_layout(); fig.savefig(pdir / fname); plt.close(fig); made.append(fname)

    # 5 & 6: rank and Meps sensitivity (3-seed mean)
    metrics = {}
    for s in seeds:
        mf = root / "metrics" / f"seed{s}_metrics.json"
        if mf.exists(): metrics[s] = json.load(open(mf))
    if metrics:
        def mean_over_seeds(key, fam, m):
            vals = [metrics[s]["variants"][key][fam][m] for s in metrics if key in metrics[s]["variants"]]
            return float(np.mean(vals)) if vals else np.nan
        ks = cfg["sensitivity"]["num_eigs"]; meps = cfg["sensitivity"]["Meps"]; mprim = int(float(cfg["Meps"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, fam in enumerate(["near", "far"]):
            y = [mean_over_seeds(f"tempered_k{k}_Meps{mprim}", fam, "auroc") for k in ks]
            ax.plot(ks, y, "-o", lw=2, ms=7, color=OKABE[i], label=f"{fam.title()} AUROC")
        ax.set_xlabel("rank k"); ax.set_ylabel("AUROC"); ax.set_xticks(ks)
        ax.set_title(f"Rank sensitivity (Meps={mprim}, 3-seed mean)"); ax.legend()
        fig.tight_layout(); fig.savefig(pdir / "rank_sensitivity.png"); plt.close(fig); made.append("rank_sensitivity.png")

        kprim = int(cfg["reported_num_eigs"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, fam in enumerate(["near", "far"]):
            y = [mean_over_seeds(f"tempered_k{kprim}_Meps{int(m)}", fam, "auroc") for m in meps]
            ax.plot(meps, y, "-o", lw=2, ms=7, color=OKABE[i], label=f"{fam.title()} AUROC")
        ax.set_xscale("log"); ax.set_xlabel("Meps"); ax.set_ylabel("AUROC")
        ax.set_title(f"Meps sensitivity (k={kprim}, 3-seed mean)"); ax.legend()
        fig.tight_layout(); fig.savefig(pdir / "Meps_sensitivity.png"); plt.close(fig); made.append("Meps_sensitivity.png")

    # 8: Fisher spectrum by seed
    specs = sorted(glob.glob(str(root / "spectra" / "scod1024_seed*_tempered.json")))
    if specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, sp in enumerate(specs):
            d = json.load(open(sp)); ev = np.array(d["eigvals"])
            ax.semilogy(np.arange(1, len(ev)+1), ev, "-o", ms=5, lw=1.5,
                        color=OKABE[i % len(OKABE)], label=f"seed {d['seed']}")
        ax.set_xlabel("eigenvalue index"); ax.set_ylabel("Fisher eigenvalue")
        ax.set_title("Fisher spectrum by seed (tempered)"); ax.legend()
        fig.tight_layout(); fig.savefig(pdir / "fisher_spectrum_by_seed.png"); plt.close(fig)
        made.append("fisher_spectrum_by_seed.png")

    # 11: cost-quality frontier (inference passes vs Near AUROC)
    if aggf.exists():
        from experiments.scod_cifar.aggregate import SUBMITTED
        agg = json.load(open(aggf))
        # (label, inference_passes, near_auroc)
        pts = [("MSP", 1, 87.70), ("Mahalanobis", 1, 87.98), ("MC-Dropout", 32, 87.25),
               ("LLLA", 50, 88.97), ("SWAG", 50, 90.03), ("P&C", 50, 91.55),
               ("Deep Ens", 5, 91.10), ("SCOD-1024", 1, agg["near"]["auroc"][0])]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, (lab, passes, au) in enumerate(pts):
            ax.scatter(passes, au, s=70, color=OKABE[i % len(OKABE)], zorder=3)
            ax.annotate(lab, (passes, au), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("inference passes (log)"); ax.set_ylabel("Near AUROC")
        ax.set_title("Cost-quality frontier (Near-OOD)")
        fig.tight_layout(); fig.savefig(pdir / "cost_quality_frontier.png"); plt.close(fig)
        made.append("cost_quality_frontier.png")

    # sketch-seed sensitivity: only if >1 sketch seed built per model seed
    ssfiles = glob.glob(str(root / "sketches" / f"scod1024_seed{s0}_sketchseed*_tempered.npz"))
    if len(ssfiles) <= 1:
        print("[plots] sketch_seed_sensitivity.png SKIPPED (only primary sketch seed built)")
    print(f"[plots] wrote {len(made)} figures to {pdir}: {made}")
    skipped = {"scod_vs_pnc_scatter.png": "needs P&C per-example scores (not produced here)",
               "sketch_seed_sensitivity.png": "needs multiple sketch seeds per model seed"}
    json.dump(dict(made=made, skipped=skipped), open(pdir / "_plots_manifest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
