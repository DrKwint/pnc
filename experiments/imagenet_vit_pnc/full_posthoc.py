"""Post-hoc OOD sensitivity across perturbation scale (spec §30).

**This analysis is post hoc and does not revise the frozen configuration** (§29). It exists
to answer two questions the headline cannot:

  * how strongly does ID-only selection identify the best OOD operating point?
  * does the affine correction separate from the matched uncorrected ensemble on OOD at
    larger perturbations, where its ID benefit is known to be much bigger?

The scale grid deliberately runs past the ID-stability gate (r = 1.0 and 2.0 both fail it),
because the interesting comparison is exactly where the correction starts to matter.
Configurations that fail the gate are marked and can never be promoted to the headline.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import full_cache as fc
from . import full_ood as fo
from . import full_oodmetrics as fm
from . import full_pnc as fp
from .memprobe import reset_cuda
from .vit_adapter import ViTPnCAdapter

SCALES = [0.125, 0.25, 0.375, 0.5, 1.0, 2.0]
M_DIAG = 10


@torch.inference_mode()
def _entropy(ad, X, members, T, corrected):
    acc = None
    for mem in members:
        lg = fp.member_logits(ad, X, mem, corrected=corrected)
        p = torch.softmax(lg.double() / T, dim=-1)
        acc = p if acc is None else acc + p
        del lg, p
    return fp.predictive_entropy((acc / len(members)).numpy())


def run(out: Path, seed: int = 0):
    cfg = json.loads((out / "selection" / "selected_config.json").read_text())
    T = json.loads((out / "metrics" / "temperature.json").read_text())["temperature"]
    gate_pass = {float(r["r_target"]): r["passes_gate"] == "True"
                 for r in csv.DictReader((out / "selection" / "all_stage_a.csv").open())}

    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    corr = fc.load_cache(out / "raw" / "cache_correction.npz")
    h = torch.as_tensor(corr["h"], device=ad.device, dtype=ad.dtype)
    z0 = torch.as_tensor(corr["z0"], device=ad.device, dtype=ad.dtype)
    val = fc.load_cache(out / "raw" / "cache_val50k.npz")
    Xval = fc.cache_to_gpu(val, ad)

    ood_X, groups = {}, {}
    for key, (_, _, group) in fo.DATASETS.items():
        ood_X[key] = fc.cache_to_gpu(fc.load_cache(out / "raw" / f"cache_ood_{key}.npz"), ad)
        groups[key] = group

    rows = []
    for r in SCALES:
        factory = fp.MemberFactory(ad, h, z0, seed, cfg["K"], M_DIAG)
        built = factory.build(r, cfg["lambda"], cfg["n_cal"])
        for corrected in (True, False):
            id_s = _entropy(ad, Xval, built["members"], T, corrected)
            fam = {g: {} for g in ("near", "far")}
            per_ds = {}
            for key, X in ood_X.items():
                s = _entropy(ad, X, built["members"], T, corrected)
                fam[groups[key]][key] = s
                per_ds[key] = fm.binary_ood_metrics(id_s, s)
            agg = {g: fm.aggregate_family_metrics(id_s, fam[g]) for g in fam}
            row = {"r_target": r, "variant": "P&C" if corrected else "uncorrected",
                   "lambda": cfg["lambda"], "n_cal": cfg["n_cal"], "M": M_DIAG,
                   "seed": seed,
                   "passes_id_gate": gate_pass.get(r, "not_in_stage_a"),
                   "is_selected": bool(corrected and abs(r - cfg["r_target"]) < 1e-9),
                   "near_auroc": agg["near"]["mean_auroc"],
                   "near_fpr95": agg["near"]["mean_fpr95"],
                   "far_auroc": agg["far"]["mean_auroc"],
                   "far_fpr95": agg["far"]["mean_fpr95"],
                   **{f"{k}_auroc": per_ds[k]["auroc"] for k in per_ds}}
            rows.append(row)
            print(f"  r={r:<6g} {row['variant']:<12} gate={str(row['passes_id_gate']):<5} "
                  f"Near {row['near_auroc']*100:6.2f}  Far {row['far_auroc']*100:6.2f}",
                  flush=True)
        for mem in built["members"]:
            del mem["W1v"], mem["W2"], mem["b2"]
        del built
        reset_cuda()

    d = out / "metrics" / "posthoc_ood_sensitivity"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "scale_sensitivity.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    stable = [x for x in rows if x["passes_id_gate"] is True and x["variant"] == "P&C"]
    spread = {
        "near_auroc_range_over_id_stable": (
            max(x["near_auroc"] for x in stable) - min(x["near_auroc"] for x in stable))
        if stable else None,
        "far_auroc_range_over_id_stable": (
            max(x["far_auroc"] for x in stable) - min(x["far_auroc"] for x in stable))
        if stable else None,
        "best_id_stable_r_by_near_auroc": max(stable, key=lambda x: x["near_auroc"])["r_target"]
        if stable else None,
        "selected_r": cfg["r_target"],
        "best_overall_r_by_near_auroc": max(
            (x for x in rows if x["variant"] == "P&C"), key=lambda x: x["near_auroc"])["r_target"],
        "note": "post hoc only; the frozen configuration is not revised (spec §29)",
    }
    fc.write_json(d / "summary.json", {"spread": spread, "rows": rows})
    print(f"\n  ID-stable Near-AUROC spread: "
          f"{(spread['near_auroc_range_over_id_stable'] or 0)*100:.2f} pp; "
          f"best ID-stable r = {spread['best_id_stable_r_by_near_auroc']}, "
          f"selected r = {spread['selected_r']}, "
          f"best overall r = {spread['best_overall_r_by_near_auroc']}")
    return rows


if __name__ == "__main__":
    run(Path("results/neurips_2026_rebuttal/imagenet_vit"))
