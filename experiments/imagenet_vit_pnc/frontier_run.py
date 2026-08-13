"""Driver for the preservation-frontier search (spec §9-15).

    search   coarse grid + adaptive extension + bisection refinement, freeze 3 configs
    final    M=20 x 5-seed ensembles on the 50k validation set + matched uncorrected
    ood      OpenOOD for all three operating points, reusing the deterministic baselines
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import full_cache as fc
from .frontier import (BUDGETS, COARSE_R, EXTEND_DOWN, EXTEND_UP, LAMBDAS, OUT, PRIMARY,
                       SRC, FrontierSearcher, best_lambda, boundary, pick_scale)

ROW_COLS = [
    "r_target", "realized_r_median", "realized_r_min", "realized_r_max", "lambda", "M",
    "n_cal", "top1", "top1_std", "delta_top1_pp", "lcb_pp", "ucb_pp",
    "bootstrap_point_pp", "seed_spread_pp", "top5", "nll", "nll_std", "ece", "ece_std",
    "brier", "base_agreement", "mean_pred_entropy",
    "logit_mse_mean", "logit_mse_median", "logit_mse_p90", "logit_mse_p95",
    "logit_mse_p99", "calib_residual", "heldout_cls_residual",
    "unc_top1", "unc_delta_top1_pp", "unc_nll", "unc_ece", "unc_base_agreement",
    "unc_logit_mse_median", "unc_logit_mse_p99",
    "all_finite", "pathology", "pass_strict", "pass_primary", "pass_relaxed",
]


def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, ROW_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["r_target"], x["lambda"])):
            w.writerow({k: r.get(k, "") for k in ROW_COLS})


def _print_scale(rows: list[dict]):
    r0 = rows[0]
    print(f"  r={r0['r_target']:<6g} (realized median {r0['realized_r_median']:.3f})", flush=True)
    for r in rows:
        b = "".join(n[0].upper() if r[f"pass_{n}"] else "-"
                    for n in ("strict", "primary", "relaxed"))
        flag = f"  [{r['pathology']}]" if r["pathology"] else ""
        print(f"     lam={r['lambda']:<8g} top1 {r['top1']*100:7.3f}  Δ{r['delta_top1_pp']:+.3f}pp  "
              f"LCB {r['lcb_pp']:+.3f}pp  NLL {r['nll']:.4f}  ECE {r['ece']:.4f}  "
              f"agree {r['base_agreement']:.4f}  [{b}]{flag}", flush=True)


def run_search(args):
    s = FrontierSearcher()
    all_rows: list[dict] = []
    tested: set[float] = set()

    def do(r: float):
        if r in tested:
            return
        tested.add(r)
        rows = s.evaluate_scale(r, LAMBDAS)
        _print_scale(rows)
        all_rows.extend(rows)
        _write(OUT / "selection" / "frontier_grid.csv", all_rows)

    print("== coarse grid ==")
    for r in COARSE_R:
        do(r)

    # §9 adaptive extension
    b_rel = boundary(all_rows, "relaxed")
    if b_rel["largest_passing_r"] == max(tested):
        for nxt in EXTEND_UP:
            if nxt not in tested:
                print(f"  r={max(tested)} still passes RELAXED -> extending to {nxt}")
                do(nxt)
                if boundary(all_rows, "relaxed")["largest_passing_r"] != nxt:
                    break
    if b_rel["largest_passing_r"] is None or min(tested) > (b_rel["largest_passing_r"] or 0):
        if best_lambda([r for r in all_rows if r["r_target"] == min(tested)],
                       "relaxed") is None:
            for nxt in EXTEND_DOWN:
                if nxt not in tested:
                    print(f"  r={min(tested)} fails even RELAXED -> extending down to {nxt}")
                    do(nxt)
                    if best_lambda([r for r in all_rows if r["r_target"] == nxt],
                                   "relaxed") is not None:
                        break

    # §14 bisection on the PRIMARY boundary
    print("\n== bisection refinement (PRIMARY budget) ==")
    for _ in range(args.max_bisect):
        b = boundary(all_rows, PRIMARY)
        lo, hi = b["largest_passing_r"], b["smallest_failing_r_above"]
        if lo is None or hi is None:
            print(f"  no bracketed boundary (largest pass {lo}, smallest fail above {hi})")
            break
        if (hi - lo) <= 0.125 or (hi - lo) / hi <= 0.10:
            print(f"  boundary bracketed to [{lo}, {hi}] "
                  f"(width {hi-lo:.3f}, {100*(hi-lo)/hi:.1f}%) — done")
            break
        mid = round((lo + hi) / 2, 4)
        print(f"  bracket [{lo}, {hi}] -> testing midpoint {mid}")
        do(mid)

    _write(OUT / "selection" / "frontier_grid.csv", all_rows)

    # §12 best ridge per (scale, budget)
    by_r: dict[float, list[dict]] = {}
    for r in all_rows:
        by_r.setdefault(r["r_target"], []).append(r)
    bl = []
    for r in sorted(by_r):
        for name in BUDGETS:
            b = best_lambda(by_r[r], name)
            bl.append({"r_target": r, "budget": name,
                       "passes": b is not None,
                       "best_lambda": b["lambda"] if b else "",
                       "top1": b["top1"] if b else "",
                       "delta_top1_pp": b["delta_top1_pp"] if b else "",
                       "lcb_pp": b["lcb_pp"] if b else "",
                       "nll": b["nll"] if b else "", "ece": b["ece"] if b else ""})
    with (OUT / "selection" / "best_lambda_by_scale.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(bl[0]))
        w.writeheader()
        w.writerows(bl)

    # §13/§15 freeze the three operating points
    frozen, bounds = {}, {}
    for name, eps in BUDGETS.items():
        pickd = pick_scale(all_rows, name)
        bounds[name] = boundary(all_rows, name)
        frozen[name] = None if pickd is None else {
            "budget": name, "max_top1_loss_pp": eps * 100,
            "r_target": pickd["r_target"],
            "realized_r_median": pickd["realized_r_median"],
            "realized_r_min": pickd["realized_r_min"],
            "realized_r_max": pickd["realized_r_max"],
            "lambda": pickd["lambda"], "K": 20, "n_cal": pickd["n_cal"],
            "M_search": pickd["M"], "M_final": 20,
            "selection_top1": pickd["top1"], "selection_delta_top1_pp": pickd["delta_top1_pp"],
            "selection_lcb_pp": pickd["lcb_pp"], "selection_nll": pickd["nll"],
            "selection_ece": pickd["ece"],
            "selection_base_agreement": pickd["base_agreement"],
            "seed_spread_pp": pickd["seed_spread_pp"],
            "base_top1_selection_pool": pickd["base_top1"],
            "boundary": bounds[name],
        }
    fc.write_json(OUT / "selection" / "frozen_configs.json",
                  {"budgets_pp": {k: v * 100 for k, v in BUDGETS.items()},
                   "temperature": s.T, "seeds_search": list(s.seeds),
                   "lambdas_searched": LAMBDAS, "scales_tested": sorted(tested),
                   "configs": frozen, "boundaries": bounds,
                   "ood_data_accessed_before_freeze": False})

    print("\n== frozen operating points ==")
    for name in ("strict", "primary", "relaxed"):
        c, b = frozen[name], bounds[name]
        if c is None:
            print(f"  {name.upper():<8} no scale passes")
            continue
        print(f"  {name.upper():<8} r={c['r_target']:<6g} lam={c['lambda']:<8g} "
              f"top1 {c['selection_top1']*100:.3f}% (Δ{c['selection_delta_top1_pp']:+.3f}pp, "
              f"LCB {c['selection_lcb_pp']:+.3f}pp)")
        print(f"           boundary: passes through r={b['largest_passing_r']}, "
              f"fails by r={b['smallest_failing_r_above']}")
    print(f"\nwrote {OUT/'selection'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["search", "final", "ood"])
    ap.add_argument("--max-bisect", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    if args.stage == "search":
        run_search(args)
    elif args.stage == "final":
        from .frontier_final import run_final
        run_final(args)
    else:
        from .frontier_final import run_ood
        run_ood(args)


if __name__ == "__main__":
    main()
