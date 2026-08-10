"""Assemble CIFAR_FULL_GRID_REPORT.md from selection + interaction + OOD-eval artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "pnc_full_grid"
SEEDS = [0, 1, 2]


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def fmt(t, dp=2):
    return "n/a" if t is None else (f"{t[0]:.{dp}f}±{t[1]:.{dp}f}" if t[1] else f"{t[0]:.{dp}f}")


def ood_agg(tag):
    rows = []
    for s in SEEDS:
        f = ROOT / "final_metrics" / tag / f"seed_{s}" / "summary.json"
        if f.exists(): rows.append(json.load(open(f)))
    if not rows: return None
    keys = ["id_acc", "id_nll", "id_ece", "id_brier", "near_auroc", "near_fpr95", "far_auroc", "far_fpr95"]
    return {k: ms([r.get(k) for r in rows]) for k in keys}


def main():
    sel = json.load(open(ROOT / "selected" / "global_config.json"))
    g = sel["global_config"]; cd = sel["coordinate_descent_config"]
    gg = ood_agg("global_grid"); cdo = ood_agg("coordinate_descent")
    same_config = (g["stage_idx"], g["block_idx"], g["scale"], g["bootstrap_frac"]) == \
                  (cd["stage_idx"], cd["block_idx"], cd["scale"], cd["bootstrap_frac"])
    if same_config and cdo is None:   # identical config -> OOD numbers are the same
        cdo = gg
    interaction = (ROOT / "tables" / "interaction_summary.md").read_text() if (ROOT / "tables" / "interaction_summary.md").exists() else ""
    cdpaths = json.load(open(ROOT / "tables" / "cd_paths.json")) if (ROOT / "tables" / "cd_paths.json").exists() else {}
    regret = sel["val_nll_regret_CD_minus_global"]

    L = []
    L.append("# CIFAR-10 Full P&C Grid Search — Report\n")
    L.append("Exhaustive 6×3×3 = 54 configs/checkpoint × 3 checkpoints = **162 runs**, replacing the "
             "submitted coordinate-wise selection over {target block, perturbation scale, bootstrap "
             "fraction}. K=20, M=50, λ=1e-3, calib=1024 fixed. Selection by **mean ID-val NLL** "
             "(temperature-scaled, ID-only). This closes the *search-procedure* gap; the one-factor "
             "sweeps remain a separate *robustness* claim.\n")

    L.append("## Reviewer statement\n")
    def rv(k, o): return "n/a" if not o or o.get(k) is None else fmt(o[k])
    verdict = "did not miss" if (not sel["regret_exceeds_seed_variability"] and same_config) else "missed"
    L.append(f"> We clarify that CIFAR-10 originally used coordinate-wise ID-validation selection over "
             f"target block, perturbation scale, and bootstrap fraction. We therefore evaluated the "
             f"complete 6×3×3 = 54 cross-product on each of the same three checkpoints. The global "
             f"optimum was **{g['label']} ps{g['scale']:g} bf{g['bootstrap_frac']:g}**, compared with "
             f"the coordinate-descent choice **{cd['label']} ps{cd['scale']:g} bf{cd['bootstrap_frac']:g}** "
             f"(rank {cd['rank_among_54']}/54). Its mean ID-validation NLL changed by **{regret:+.4f}**, "
             f"and Near/Far AUROC changed from **{rv('near_auroc', cdo)}/{rv('far_auroc', cdo)}** to "
             f"**{rv('near_auroc', gg)}/{rv('far_auroc', gg)}**. Thus coordinate descent **{verdict}** a "
             f"material interaction among the tuned variables.\n")

    L.append("## Objective answers\n")
    L.append(f"1. **Global optimum vs coordinate-descent:** {'SAME config' if same_config else 'DIFFERENT'} "
             f"— global {g['label']}/ps{g['scale']:g}/bf{g['bootstrap_frac']:g} (val NLL "
             f"{g['mean_val_nll']:.4f}); CD {cd['label']}/ps{cd['scale']:g}/bf{cd['bootstrap_frac']:g} "
             f"rank {cd['rank_among_54']}/54 (val NLL {cd['mean_val_nll']:.4f}).")
    L.append(f"2. **Val-NLL lost to coordinate descent:** {regret:.4f} (median seed std "
             f"{sel['median_seed_std']:.4f}; exceeds seed variability: {sel['regret_exceeds_seed_variability']}).")
    if gg and cdo:
        L.append(f"3. **OOD/ID change:** Near AUROC {rv('near_auroc', cdo)} → {rv('near_auroc', gg)}, "
                 f"Far AUROC {rv('far_auroc', cdo)} → {rv('far_auroc', gg)}, ID acc "
                 f"{rv('id_acc', cdo)} → {rv('id_acc', gg)}.")
    else:
        L.append("3. **OOD/ID change:** (OOD eval pending)")
    L.append(f"4. **Interactions:** see interaction summary; CD-path forward→"
             f"{cdpaths.get('forward',{}).get('block','?')} vs reverse→{cdpaths.get('reverse',{}).get('block','?')} "
             f"vs global {g['label']}.\n")

    L.append("## Final OOD comparison (3-seed mean ± std)\n")
    L.append("| Config | Acc | NLL | Near AUROC | Near FPR95 | Far AUROC | Far FPR95 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, o in [(f"Coordinate-descent ({cd['label']}/ps{cd['scale']:g}/bf{cd['bootstrap_frac']:g})", cdo),
                    (f"Global grid ({g['label']}/ps{g['scale']:g}/bf{g['bootstrap_frac']:g})", gg)]:
        if o:
            L.append(f"| {name} | {rv('id_acc', o)} | {fmt(o['id_nll'],3)} | {rv('near_auroc', o)} | "
                     f"{rv('near_fpr95', o)} | {rv('far_auroc', o)} | {rv('far_fpr95', o)} |")
    L.append("")
    L.append(interaction)
    (ROOT / "CIFAR_FULL_GRID_REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote CIFAR_FULL_GRID_REPORT.md")
    print(f"global={g['label']}/ps{g['scale']:g}/bf{g['bootstrap_frac']:g} same_as_CD={same_config} regret={regret:.4f}")


if __name__ == "__main__":
    main()
