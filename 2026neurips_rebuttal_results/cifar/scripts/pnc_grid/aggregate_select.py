"""Aggregate the 162 candidates, select the global config by mean ID-val NLL, compare to the
coordinate-descent config, and analyze interactions + coordinate-descent path reconstruction.
Aggregates ONLY when all 54 configs exist for all 3 seeds (per spec).
"""
from __future__ import annotations
import json, hashlib, itertools
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "pnc_full_grid"
SEEDS = [0, 1, 2]; BLOCKS = [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
SCALES = [25.0, 50.0, 100.0]; FRACS = [0.05, 0.10, 0.20]
CD = dict(label="s3b0", stage_idx=3, block_idx=0, scale=25.0, bootstrap_frac=0.05)  # artifact-backed


def load_all():
    rows = []
    for seed in SEEDS:
        for (s, b), scale, frac in itertools.product(BLOCKS, SCALES, FRACS):
            d = ROOT / "candidates" / f"seed_{seed}" / f"s{s}b{b}_ps{scale:g}_bf{frac:g}"
            mf = d / "metrics.json"
            if not mf.exists():
                rows.append(dict(seed=seed, stage_idx=s, block_idx=b, label=f"s{s}b{b}", scale=scale,
                                 bootstrap_frac=frac, status="missing")); continue
            m = json.load(open(mf)); m.setdefault("label", f"s{s}b{b}"); rows.append(m)
    return pd.DataFrame(rows)


def config_key(r): return (r["stage_idx"], r["block_idx"], r["scale"], r["bootstrap_frac"])


def main():
    df = load_all()
    ok = df[df["status"] == "ok"].copy() if "status" in df else df.copy()
    n_missing = int((df["status"] != "ok").sum()) if "status" in df else 0
    (ROOT / "tables").mkdir(exist_ok=True)
    df.to_csv(ROOT / "tables" / "full_grid_per_seed.csv", index=False)
    if n_missing:
        print(f"[select] {n_missing} candidates not ok; cannot finalize selection yet.");
        if len(ok) < 162:
            print(f"[select] {len(ok)}/162 complete."); return

    # aggregate 54 configs across seeds
    grp = ok.groupby(["stage_idx", "block_idx", "label", "scale", "bootstrap_frac"])
    agg = grp["val_nll_calibrated"].agg(["mean", "std", "count"]).reset_index()
    agg = agg.rename(columns={"mean": "mean_val_nll", "std": "std_val_nll"})
    agg["val_acc_mean"] = grp["val_accuracy"].mean().values
    agg = agg.sort_values(["mean_val_nll", "std_val_nll", "stage_idx", "block_idx", "scale", "bootstrap_frac"])
    agg.to_csv(ROOT / "tables" / "full_grid_aggregate.csv", index=False)

    # global selection (tie-break already encoded by sort)
    best = agg.iloc[0]
    global_cfg = dict(label=best["label"], stage_idx=int(best["stage_idx"]), block_idx=int(best["block_idx"]),
                      scale=float(best["scale"]), bootstrap_frac=float(best["bootstrap_frac"]),
                      mean_val_nll=float(best["mean_val_nll"]), std_val_nll=float(best["std_val_nll"]))
    # CD config stats + rank
    agg = agg.reset_index(drop=True); agg["rank"] = np.arange(1, len(agg) + 1)
    cdrow = agg[(agg.stage_idx == CD["stage_idx"]) & (agg.block_idx == CD["block_idx"]) &
                (agg.scale == CD["scale"]) & (agg.bootstrap_frac == CD["bootstrap_frac"])]
    cd = cdrow.iloc[0]
    regret = float(cd["mean_val_nll"] - best["mean_val_nll"])
    seed_var = float(agg["std_val_nll"].median())
    sel = dict(global_config=global_cfg, coordinate_descent_config={**CD,
               "mean_val_nll": float(cd["mean_val_nll"]), "std_val_nll": float(cd["std_val_nll"]),
               "rank_among_54": int(cd["rank"])},
               val_nll_regret_CD_minus_global=regret,
               regret_exceeds_seed_variability=bool(regret > seed_var),
               median_seed_std=seed_var, n_configs=len(agg))
    h = hashlib.sha256(json.dumps(global_cfg, sort_keys=True).encode()).hexdigest()
    sel["global_config_sha256"] = h
    (ROOT / "selected").mkdir(exist_ok=True)
    json.dump(sel, open(ROOT / "selected" / "global_config.json", "w"), indent=2)

    # coordinate_vs_full_grid.md
    md = ["# Coordinate-descent vs full-grid selection\n",
          f"- **Global optimum:** {global_cfg['label']} ps{global_cfg['scale']:g} bf{global_cfg['bootstrap_frac']:g} "
          f"(mean val NLL {global_cfg['mean_val_nll']:.4f} ± {global_cfg['std_val_nll']:.4f})",
          f"- **Coordinate-descent config:** {CD['label']} ps{CD['scale']:g} bf{CD['bootstrap_frac']:g} "
          f"(mean val NLL {float(cd['mean_val_nll']):.4f}, **rank {int(cd['rank'])}/54**)",
          f"- **Val-NLL regret (CD − global):** {regret:.4f}  (median seed std {seed_var:.4f}; "
          f"exceeds seed variability: {sel['regret_exceeds_seed_variability']})\n",
          "## Top 10 configs by mean val NLL\n",
          "| rank | config | mean val NLL | std | val acc |", "|---|---|---|---|---|"]
    for _, r in agg.head(10).iterrows():
        md.append(f"| {int(r['rank'])} | {r['label']} ps{r['scale']:g} bf{r['bootstrap_frac']:g} | "
                  f"{r['mean_val_nll']:.4f} | {r['std_val_nll']:.4f} | {r['val_acc_mean']:.2f} |")
    (ROOT / "tables" / "coordinate_vs_full_grid.md").write_text("\n".join(md) + "\n")

    # per-block 3x3 tables + interaction summary
    lines = ["# Validation NLL by block (3x3 scale × bootstrap, mean over 3 seeds)\n"]
    for (s, b) in BLOCKS:
        sub = agg[(agg.stage_idx == s) & (agg.block_idx == b)]
        lines.append(f"## s{s}b{b}\n")
        lines.append("| scale＼bf | " + " | ".join(f"bf={f:g}" for f in FRACS) + " |")
        lines.append("|---|" + "---|" * len(FRACS))
        for sc in SCALES:
            cells = []
            for f in FRACS:
                v = sub[(sub.scale == sc) & (sub.bootstrap_frac == f)]["mean_val_nll"]
                cells.append(f"{v.iloc[0]:.4f}" if len(v) else "n/a")
            lines.append(f"| ps={sc:g} | " + " | ".join(cells) + " |")
        lines.append("")
    (ROOT / "tables" / "validation_nll_by_block.md").write_text("\n".join(lines) + "\n")

    # interaction summary: best scale/frac per block, best frac per scale
    isum = ["# Interaction summary\n", "## Best scale & bootstrap within each block (mean val NLL)\n",
            "| block | best scale | best bf | best NLL | NLL range across other factors |",
            "|---|---|---|---|---|"]
    for (s, b) in BLOCKS:
        sub = agg[(agg.stage_idx == s) & (agg.block_idx == b)]
        r = sub.loc[sub["mean_val_nll"].idxmin()]
        rng = float(sub["mean_val_nll"].max() - sub["mean_val_nll"].min())
        isum.append(f"| s{s}b{b} | ps{r['scale']:g} | bf{r['bootstrap_frac']:g} | {r['mean_val_nll']:.4f} | {rng:.4f} |")
    isum += ["\n## Best bootstrap fraction at each scale (pooled over blocks)\n",
             "| scale | best bf | mean NLL |", "|---|---|---|"]
    for sc in SCALES:
        sub = agg[agg.scale == sc].groupby("bootstrap_frac")["mean_val_nll"].mean()
        isum.append(f"| ps{sc:g} | bf{sub.idxmin():g} | {sub.min():.4f} |")

    # coordinate-descent path reconstruction (forward block->scale->frac; reverse frac->scale->block)
    A = agg.set_index(["stage_idx", "block_idx", "scale", "bootstrap_frac"])["mean_val_nll"]
    def cell(s, b, sc, f): return float(A.loc[(s, b, sc, f)])
    # forward: init (scale=25,frac=0.05); pick best block, then best scale, then best frac
    blocks_at = {(s, b): cell(s, b, 25.0, 0.05) for (s, b) in BLOCKS}
    fb = min(blocks_at, key=blocks_at.get)
    scales_at = {sc: cell(*fb, sc, 0.05) for sc in SCALES}; fsc = min(scales_at, key=scales_at.get)
    fracs_at = {f: cell(*fb, fsc, f) for f in FRACS}; ff = min(fracs_at, key=fracs_at.get)
    fwd = dict(block=f"s{fb[0]}b{fb[1]}", scale=fsc, frac=ff, nll=cell(*fb, fsc, ff))
    # reverse: init (block=s3b0, scale=25); pick best frac, then best scale, then best block
    fr0 = min(FRACS, key=lambda f: cell(3, 0, 25.0, f))
    sc0 = min(SCALES, key=lambda sc: cell(3, 0, sc, fr0))
    blk0 = min(BLOCKS, key=lambda sb: cell(sb[0], sb[1], sc0, fr0))
    rev = dict(block=f"s{blk0[0]}b{blk0[1]}", scale=sc0, frac=fr0, nll=cell(blk0[0], blk0[1], sc0, fr0))
    isum += ["\n## Coordinate-descent path reconstruction (from the completed grid)\n",
             f"- Forward (block→scale→bootstrap, init ps25/bf0.05): → **{fwd['block']} ps{fwd['scale']:g} "
             f"bf{fwd['frac']:g}** (NLL {fwd['nll']:.4f})",
             f"- Reverse (bootstrap→scale→block, init s3b0/ps25): → **{rev['block']} ps{rev['scale']:g} "
             f"bf{rev['frac']:g}** (NLL {rev['nll']:.4f})",
             f"- Global optimum: **{global_cfg['label']} ps{global_cfg['scale']:g} bf{global_cfg['bootstrap_frac']:g}** "
             f"(NLL {global_cfg['mean_val_nll']:.4f})",
             f"- Paths converge to same config: {fwd==rev and fwd['block']==global_cfg['label']}"]
    (ROOT / "tables" / "interaction_summary.md").write_text("\n".join(isum) + "\n")
    json.dump(dict(forward=fwd, reverse=rev, global_opt=global_cfg), open(ROOT / "tables" / "cd_paths.json", "w"), indent=2, default=float)

    print(f"[select] GLOBAL: {global_cfg['label']} ps{global_cfg['scale']:g} bf{global_cfg['bootstrap_frac']:g} "
          f"mean val NLL {global_cfg['mean_val_nll']:.4f}")
    print(f"[select] CD s3b0/ps25/bf0.05 rank {int(cd['rank'])}/54, NLL {float(cd['mean_val_nll']):.4f}, "
          f"regret {regret:.4f} (seed std {seed_var:.4f})")
    print(f"[select] CD-path forward→{fwd['block']}/ps{fwd['scale']:g}/bf{fwd['frac']:g}, "
          f"reverse→{rev['block']}/ps{rev['scale']:g}/bf{rev['frac']:g}")


if __name__ == "__main__":
    main()
