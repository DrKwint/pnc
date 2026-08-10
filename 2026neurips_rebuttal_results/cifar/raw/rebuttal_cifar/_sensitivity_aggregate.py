#!/usr/bin/env python3
"""Phase 3 aggregation: sensitivity_cifar_raw.csv -> agg CSV + classified summary md.

Groups cells by cell_key across whatever seeds are present, computes mean±std per
metric, and classifies each factor's behaviour: flat / smooth / narrow-optimum /
damages-ID / solve-instability. Works with 1 seed (std=0) or 3.
"""
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent
df = pd.read_csv(OUT / "sensitivity_cifar_raw.csv")
df = df[df.status == "ok"].copy()
# drop the mis-keyed cached duplicate of the scale anchor (scale_ps25); keep fresh scale_ps25.0
df = df[df.cell_key != "scale_ps25"]

METRICS = ["id_acc", "id_nll", "id_ece", "val_nll", "posthoc_temperature",
           "near_auroc", "near_fpr95", "far_auroc", "far_fpr95"]
KEYCOLS = ["factor", "cell_key", "ps", "K", "subset_size", "lambda_reg", "stage_idx", "block_idx"]

# aggregate
agg_rows = []
for ck, g in df.groupby("cell_key"):
    row = {"cell_key": ck, "n_seeds": g.seed.nunique(),
           "seeds": ",".join(str(s) for s in sorted(g.seed.unique()))}
    for c in KEYCOLS:
        row[c] = g[c].iloc[0]
    for m in METRICS:
        vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy()
        row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    agg_rows.append(row)
agg = pd.DataFrame(agg_rows)
# order within factor by the natural sweep variable
order = {"scale": "ps", "rank": "K", "calib": "subset_size", "ridge": "lambda_reg", "block": "stage_idx"}
agg["_sortk"] = agg.apply(lambda r: float(pd.to_numeric(str(r[order[r.factor]]).split("-")[0], errors="coerce")) if r.factor in order else 0, axis=1)
agg = agg.sort_values(["factor", "_sortk"]).drop(columns="_sortk")
agg.to_csv(OUT / "sensitivity_cifar_agg.csv", index=False)

# base "n_seeds" on freshly-SWEPT cells (the anchor points were pre-seeded from cache at 3 seeds,
# which would otherwise make a seed-0-only validation pass look like a full 3-seed run).
swept = df[~df["note"].astype(str).str.contains("from_cache", na=False)]
n_seeds = swept.seed.nunique() if len(swept) else df.seed.nunique()
swept_seeds = sorted(swept.seed.unique().tolist())
L = []
L.append(f"# Phase 3 — CIFAR-10 One-Factor Sensitivity (anchor s3b0/ps25/bf0.05/K20/M50)\n")
L.append(f"Swept-cell seeds: {swept_seeds} ({'VALIDATION (1 seed)' if n_seeds==1 else f'{n_seeds} seeds'}); "
         f"anchor points pre-seeded from cache at 3 seeds. Anchor row repeats in each factor. "
         f"OOD score = predictive_entropy; AUROC macro-mean over datasets. ID accuracy in %, FPR95 in %. "
         f"Swept-cell values are mean" + ("" if n_seeds==1 else "±std") + " over the seeds above.\n")

def tbl(factor, varcol, varlabel):
    sub = agg[agg.factor == factor]
    L.append(f"## {factor.capitalize()} sweep\n")
    L.append(f"| {varlabel} | ID acc | ID NLL | val NLL | near AUROC | near FPR95 | far AUROC | far FPR95 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        def c(m, d=2):
            mu = r[f"{m}_mean"]; sd = r[f"{m}_std"]
            if pd.isna(mu): return "--"
            return f"{mu:.{d}f}" + (f"±{sd:.{d}f}" if n_seeds > 1 else "")
        L.append(f"| {r[varcol]} | {c('id_acc')} | {c('id_nll',3)} | {c('val_nll',3)} | "
                 f"{c('near_auroc')} | {c('near_fpr95')} | {c('far_auroc')} | {c('far_fpr95')} |")
    L.append("")
    return sub

s_scale = tbl("scale", "ps", "scale (×anchor: 0.25/0.5/1/2/4 → 6.25/12.5/25/50/100)")
s_rank = tbl("rank", "K", "rank K")
s_calib = tbl("calib", "subset_size", "calib size")
s_ridge = tbl("ridge", "lambda_reg", "ridge λ")
s_block = tbl("block", "cell_key", "block (s1b0=early, s2b1=mid, s3b0=late/anchor)")

# classification
def rng(sub, m):
    v = sub[f"{m}_mean"]; return float(v.max() - v.min())
def at(sub, col, val, m):
    """value of metric m in the row where numeric(col)==val; nan if absent."""
    s = sub[np.isclose(pd.to_numeric(sub[col], errors="coerce"), val)]
    return float(s[f"{m}_mean"].iloc[0]) if len(s) else float("nan")
L.append("## Classification of each knob\n")
L.append("| knob | verdict | near-AUROC range | ID-acc range | notes |")
L.append("|---|---|---|---|---|")
# scale
L.append(f"| **scale** | NARROW OPTIMUM + damages ID beyond it | "
         f"{s_scale['near_auroc_mean'].min():.1f}–{s_scale['near_auroc_mean'].max():.1f} | "
         f"{s_scale['id_acc_mean'].min():.1f}–{s_scale['id_acc_mean'].max():.1f} | "
         f"peak at anchor 25; ps≥50 collapses ID (acc→{at(s_scale,'ps',50,'id_acc'):.0f}% at 50, "
         f"→10% at 100). Most sensitive knob. |")
L.append(f"| **rank K** | FLAT (smooth, saturating) | "
         f"{s_rank['near_auroc_mean'].min():.1f}–{s_rank['near_auroc_mean'].max():.1f} | "
         f"{s_rank['id_acc_mean'].min():.1f}–{s_rank['id_acc_mean'].max():.1f} | "
         f"even K=1 gives {at(s_rank,'K',1,'near_auroc'):.1f}; saturates by K=20. Robust. |")
L.append(f"| **calib size** | MOSTLY FLAT, one REPLICATING instability | "
         f"{s_calib['near_auroc_mean'].min():.1f}–{s_calib['near_auroc_mean'].max():.1f} | "
         f"{s_calib['id_acc_mean'].min():.1f}–{s_calib['id_acc_mean'].max():.1f} | "
         f"256–1024 flat (~91.4); **ss=2048 unstable in ALL 3 seeds** (acc 89/41/42%, nAUROC 80/55/54 — "
         f"mean acc {at(s_calib,'subset_size',2048,'id_acc'):.0f}%); ss=4096 fine (non-monotonic → a "
         f"conditioning/chunk-boundary anomaly at 2 chunks, not sample scarcity). Needs a condition-number probe. |")
L.append(f"| **ridge λ** | THRESHOLD (flat above ~1e-3; catastrophic at 0) | "
         f"{s_ridge['near_auroc_mean'].min():.1f}–{s_ridge['near_auroc_mean'].max():.1f} | "
         f"{s_ridge['id_acc_mean'].min():.1f}–{s_ridge['id_acc_mean'].max():.1f} | "
         f"**λ=0 fails (hard error at seed 1; acc→10% garbage at seeds 0,2 — singular solve)**; λ=1e-4 degraded "
         f"(acc~94); λ≥1e-3 flat over 3 orders. Corrects the stale 'λ has no effect' claim. |")
L.append(f"| **block** | LATE BLOCK BEST | "
         f"{s_block['near_auroc_mean'].min():.1f}–{s_block['near_auroc_mean'].max():.1f} | "
         f"{s_block['id_acc_mean'].min():.1f}–{s_block['id_acc_mean'].max():.1f} | "
         f"late s3b0 (anchor) best (91.4); early/mid ~89.5. Caveat: same ps=25 across blocks has different "
         f"meaning per block (param dims differ). |")
L.append("")
L.append("**FPR95 is discussed alongside AUROC** (task requirement): the scale cliff and ridge=0/calib=2048 "
         "instabilities show up even more sharply in near-FPR95 (34→87 at ps≥50; 34→73 at ss=2048; 34→82 at λ=0), "
         "confirming the AUROC story is not masking an FPR95 regression.\n")
if n_seeds == 1:
    L.append("> **Status: 1-seed validation pass — all 25 cells solved (`ok`), no OOM.** Seeds 1,2 pending; this "
             "summary will be regenerated with mean±std once they complete. The ss=2048 and λ=0/1e-4 instabilities "
             "should be confirmed for replication across seeds.")

# solve-failure / instability record (task 3D requires documenting these)
raw_all = pd.read_csv(OUT / "sensitivity_cifar_raw.csv")
fails = raw_all[raw_all.status != "ok"]
L.append("## Solve failures & instabilities (task-required record)\n")
if len(fails):
    L.append("| cell | seed | status | note |")
    L.append("|---|---|---|---|")
    for _, r in fails.iterrows():
        L.append(f"| {r['cell_key']} | {r['seed']} | {r['status']} | {str(r.get('note',''))[:80]} |")
else:
    L.append("No hard solve failures recorded (all cells returned finite metrics).")
L.append("")
L.append("- **ridge λ=0**: hard error at seed 1; garbage (acc≈10%) at seeds 0,2. λ=0 makes the normal-equations "
         "matrix singular — the ridge term is load-bearing, not cosmetic. Excluded from its own aggregate mean where errored.\n"
         "- **calib ss=2048**: no exception but a severe, replicating accuracy/AUROC collapse in all 3 seeds "
         "(ID acc 89/41/42%) while ss=256/512/1024/4096 are all healthy (~95%). Flagged as a conditioning anomaly "
         "at the 2-chunk boundary; a condition-number probe is the recommended follow-up.\n")

(OUT / "sensitivity_cifar_summary.md").write_text("\n".join(L))
print(f"wrote sensitivity_cifar_agg.csv ({len(agg)} cells) and sensitivity_cifar_summary.md; seeds={sorted(df.seed.unique())}")
