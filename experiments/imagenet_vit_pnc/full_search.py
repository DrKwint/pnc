"""ID-only hyperparameter search: Stage A scale sweep, Stage B joint search, selection.

Every number in this module comes from the 8,192-image **training-derived** ID selection
pool. No validation data and no OOD data is read anywhere here; the OOD modules are not
even imported.

Because the CLS residual for the selection pool is cached, one configuration costs
M member constructions (a GELU, two Grams and a Cholesky each) plus M tail evaluations —
no ViT forward at all.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.banking77_pnc.pnc_metrics import clf_metrics  # REUSED evaluator

from . import full_cache as fc
from . import full_pnc as fp
from .memprobe import GIB, reset_cuda
from .vit_adapter import ViTPnCAdapter

COARSE_R = [0.125, 0.25, 0.5, 1.0, 2.0]
EXTEND_UP = [4.0, 8.0]
EXTEND_DOWN = [0.0625]
STAGE_B_LAMBDAS = [1e-3, 1.0, 1e2]
STAGE_B_NCAL = [16384, 32768]

# ---- ID-stability gate, frozen before any OOD data is read (spec §10) ----
GATE = {
    "max_top1_drop_pp": 0.25,
    "min_base_agreement": 0.99,
    "require_finite": True,
    "require_corrected_median_lt_uncorrected": True,
    "require_corrected_p99_le_uncorrected": True,
}

ROW_FIELDS = [
    "stage", "r_target", "r_realized_median", "scale", "n_cal", "lambda", "M", "seed",
    "top1", "top5", "nll", "ece", "brier", "base_agreement", "mean_pred_entropy",
    "calib_residual", "heldout_cls_residual", "uncorrected_heldout_cls_residual",
    "residual_ratio", "gram_cond",
    "member_top1_agreement", "logit_mse_mean", "logit_mse_median", "logit_mse_p90",
    "logit_mse_p95", "logit_mse_p99", "logit_mse_max",
    "unc_logit_mse_mean", "unc_logit_mse_median", "unc_logit_mse_p99",
    "frac_images_improved", "passes_gate", "gate_reasons",
    "construct_s", "eval_s", "all_finite",
]


def _gate(row: dict, base_top1: float) -> tuple[bool, str]:
    reasons = []
    if (base_top1 - row["top1"]) * 100 > GATE["max_top1_drop_pp"]:
        reasons.append(f"top1_drop={{{(base_top1 - row['top1'])*100:.3f}}}pp")
    if row["base_agreement"] < GATE["min_base_agreement"]:
        reasons.append(f"agreement={row['base_agreement']:.4f}")
    if not row["all_finite"]:
        reasons.append("non_finite")
    if not (row["logit_mse_median"] < row["unc_logit_mse_median"]):
        reasons.append("corrected_median_not_better")
    if not (row["logit_mse_p99"] <= row["unc_logit_mse_p99"]):
        reasons.append("corrected_p99_not_better")
    return (len(reasons) == 0), ";".join(reasons)


class Searcher:
    """Holds the cached tensors once; evaluates configurations cheaply."""

    def __init__(self, out: Path, seed: int = 0, K: int = 20):
        self.out = out
        self.ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
        self.seed, self.K = seed, K

        corr = fc.load_cache(out / "raw" / "cache_correction.npz")
        self.h = torch.as_tensor(corr["h"], device=self.ad.device, dtype=self.ad.dtype)
        self.z0 = torch.as_tensor(corr["z0"], device=self.ad.device, dtype=self.ad.dtype)

        sel = fc.load_cache(out / "raw" / "cache_selection.npz")
        self.Xsel = torch.as_tensor(sel["x_resid_cls"], device=self.ad.device,
                                    dtype=self.ad.dtype)
        self.ysel = torch.from_numpy(sel["labels"])
        self.base_logits = fc.logits_from_cache(self.ad, self.Xsel)
        bm = clf_metrics(torch.softmax(self.base_logits.double(), -1).numpy(),
                         self.ysel.numpy())
        self.base_top1 = bm["accuracy"]
        self.base_pred = self.base_logits.argmax(-1)
        # held-out CLS residual target: the selection pool's own (h, z0)
        self.h_sel = self.ad.block.ln_2(self.Xsel[:, None, :])[:, 0]
        self.z0_sel = (torch.nn.functional.gelu(self.h_sel @ self.ad.W1 + self.ad.b1)
                       @ self.ad.W2 + self.ad.b2)
        self.Theta0 = self.ad.theta().detach().double().cpu().numpy()
        print(f"  base on selection pool: top-1 {self.base_top1*100:.3f}%  "
              f"NLL {bm['nll']:.4f}  ECE {bm['ece']:.4f}")

    # ---- residual of a Theta on the held-out selection pool ----
    def _heldout_residual(self, W1v, Theta=None) -> float:
        y = torch.nn.functional.gelu(self.h_sel @ W1v + self.ad.b1)
        X = fp.pc.SufficientStats.augment(y)
        Th = torch.as_tensor(self.Theta0 if Theta is None else Theta,
                             device=self.ad.device, dtype=torch.float64)
        pred = X.double() @ Th
        num = torch.linalg.norm(pred - self.z0_sel.double())
        return float(num / (torch.linalg.norm(self.z0_sel.double()) + 1e-30))

    @torch.inference_mode()
    def evaluate(self, stage: str, r: float, lam: float, n_cal: int, M: int) -> dict:
        t0 = time.perf_counter()
        factory = fp.MemberFactory(self.ad, self.h, self.z0, self.seed, self.K, M)
        built = factory.build(r, lam, n_cal)
        t_construct = time.perf_counter() - t0

        t1 = time.perf_counter()
        acc_c = acc_u = None
        cor_logits, unc_logits = [], []
        for mem in built["members"]:
            lc = fp.member_logits(self.ad, self.Xsel, mem, corrected=True)
            lu = fp.member_logits(self.ad, self.Xsel, mem, corrected=False)
            cor_logits.append(lc)
            unc_logits.append(lu)
            pc_ = torch.softmax(lc.double(), -1)
            pu_ = torch.softmax(lu.double(), -1)
            acc_c = pc_ if acc_c is None else acc_c + pc_
            acc_u = pu_ if acc_u is None else acc_u + pu_
        pbar = (acc_c / M).numpy()
        t_eval = time.perf_counter() - t1

        m = clf_metrics(pbar, self.ysel.numpy())
        pred = pbar.argmax(-1)
        top5 = torch.from_numpy(pbar).topk(5, -1).indices
        cor_d = fp.logit_mse_distribution(self.base_logits, cor_logits)
        unc_d = fp.logit_mse_distribution(self.base_logits, unc_logits)
        b = self.base_logits.double().numpy()
        per_img_c = np.concatenate([((l.double().numpy() - b) ** 2).mean(-1)
                                    for l in cor_logits])
        per_img_u = np.concatenate([((l.double().numpy() - b) ** 2).mean(-1)
                                    for l in unc_logits])
        mem_agree = float(np.mean([(l.argmax(-1) == self.base_pred).double().mean().item()
                                   for l in cor_logits]))
        W1v0 = built["members"][0]["W1v"]
        Th0 = np.concatenate([built["members"][0]["b2"].detach().cpu().numpy()[None, :],
                              built["members"][0]["W2"].detach().cpu().numpy()], 0)
        ho_cor = self._heldout_residual(W1v0, Th0)
        ho_unc = self._heldout_residual(W1v0, None)
        ent = -np.sum(pbar * np.log(pbar + 1e-12), -1)

        row = {
            "stage": stage, "r_target": r,
            "r_realized_median": built["realized_r"]["median"],
            "scale": built["scale"], "n_cal": n_cal, "lambda": lam, "M": M,
            "seed": self.seed,
            "top1": m["accuracy"],
            "top5": float((top5 == self.ysel[:, None]).any(-1).double().mean()),
            "nll": m["nll"], "ece": m["ece"], "brier": m["brier"],
            "base_agreement": float((pred == self.base_pred.numpy()).mean()),
            "mean_pred_entropy": float(ent.mean()),
            "calib_residual": built["calib_residual_median"],
            "heldout_cls_residual": ho_cor,
            "uncorrected_heldout_cls_residual": ho_unc,
            "residual_ratio": ho_cor / (ho_unc + 1e-30),
            "gram_cond": "",
            "member_top1_agreement": mem_agree,
            "logit_mse_mean": cor_d["mean"], "logit_mse_median": cor_d["median"],
            "logit_mse_p90": cor_d["p90"], "logit_mse_p95": cor_d["p95"],
            "logit_mse_p99": cor_d["p99"], "logit_mse_max": cor_d["max"],
            "unc_logit_mse_mean": unc_d["mean"], "unc_logit_mse_median": unc_d["median"],
            "unc_logit_mse_p99": unc_d["p99"],
            "frac_images_improved": float((per_img_c < per_img_u).mean()),
            "all_finite": built["all_finite"] and bool(np.isfinite(pbar).all()),
            "construct_s": t_construct, "eval_s": t_eval,
        }
        ok, why = _gate(row, self.base_top1)
        row["passes_gate"], row["gate_reasons"] = ok, why
        for mem in built["members"]:
            del mem["W1v"], mem["W2"], mem["b2"]
        del built, cor_logits, unc_logits
        reset_cuda()
        return row


def _write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, ROW_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ROW_FIELDS})


def _print(row: dict, base_top1: float):
    print(f"  r={row['r_target']:<7g} lam={row['lambda']:<7g} n={row['n_cal']:<6} "
          f"top1 {row['top1']*100:6.3f}% (Δ{(row['top1']-base_top1)*100:+.3f}) "
          f"agree {row['base_agreement']:.4f} NLL {row['nll']:.4f} "
          f"mMSE {row['logit_mse_median']:.2e} p99 {row['logit_mse_p99']:.2e} "
          f"{'PASS' if row['passes_gate'] else 'FAIL:' + row['gate_reasons']}", flush=True)


def stage_a(args, out: Path):
    s = Searcher(out, seed=args.seed)
    (out / "selection").mkdir(parents=True, exist_ok=True)
    (out / "selection" / "ID_STABILITY_RULE.md").write_text(_gate_doc(s.base_top1))
    # Adaptive bracketing (spec §9): only after the whole planned grid has been measured
    # do we ask whether its endpoints still pass, and extend outward if so. Deciding this
    # against the largest value tested *so far* would extend spuriously while larger
    # planned values are still pending.
    grid, by_r, i = sorted(COARSE_R), {}, 0
    while i < len(grid):
        r = grid[i]
        i += 1
        if r not in by_r:
            by_r[r] = s.evaluate("A", r, 1e-3, 32768, M=5)
            _print(by_r[r], s.base_top1)
        if i == len(grid):                       # planned grid exhausted; try to extend
            hi, lo = max(grid), min(grid)
            if by_r[hi]["passes_gate"]:
                nxt = next((v for v in EXTEND_UP if v > hi and v not in by_r), None)
                if nxt:
                    grid.append(nxt)
                    print(f"    r={hi} still ID-stable -> extending up to {nxt}", flush=True)
            if not by_r[lo]["passes_gate"]:
                nxt = next((v for v in EXTEND_DOWN if v < lo and v not in by_r), None)
                if nxt:
                    grid.append(nxt)
                    print(f"    r={lo} already unstable -> extending down to {nxt}",
                          flush=True)
    rows = sorted(by_r.values(), key=lambda x: x["r_target"])
    tested = set(by_r)
    _write_rows(out / "selection" / "all_stage_a.csv", rows)

    passing = [r for r in rows if r["passes_gate"]]
    r_boundary = max((r["r_target"] for r in passing), default=None)
    refined = _refined_grid(r_boundary, [r["r_target"] for r in rows]) if passing else []
    fc.write_json(out / "selection" / "stage_a_summary.json",
                  {"base_top1": s.base_top1, "tested_r": sorted(tested),
                   "passing_r": sorted(r["r_target"] for r in passing),
                   "r_boundary": r_boundary, "refined_grid": refined,
                   "ood_accessed": False})
    print(f"\n  r_boundary = {r_boundary}   refined grid = {refined}")
    return refined


def _refined_grid(r_boundary: float, tested: list[float]) -> list[float]:
    """0.5x / 0.75x / 1.0x of the boundary, snapped to already-tested values (spec §11)."""
    want = [0.5 * r_boundary, 0.75 * r_boundary, r_boundary]
    out = []
    for w in want:
        near = min(tested, key=lambda t: abs(t - w))
        out.append(near if abs(near - w) / w <= 0.2 else round(w, 6))
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return sorted(uniq)


def _gate_doc(base_top1: float) -> str:
    return f"""# ID-stability gate (frozen before any OOD data was read)

Base model top-1 on the 8,192-image ID selection pool: **{base_top1*100:.3f}%**.

A configuration is ID-stable iff **all** of the following hold:

1. ensemble top-1 drop from base <= {GATE['max_top1_drop_pp']} percentage points
2. base/ensemble top-1 agreement >= {GATE['min_base_agreement']*100:.0f}%
3. all metrics finite
4. corrected median logit MSE < uncorrected median logit MSE
5. corrected p99 logit MSE <= uncorrected p99 logit MSE

Conditions 4 and 5 require the affine correction to actually improve member preservation
over the identical perturbation left uncorrected. Calibration residual is deliberately
**not** an acceptance criterion: the preflight showed it is lowest exactly where the
correction is most overfitted, so it is recorded as a diagnostic only.

Selection uses the ID selection pool drawn from ImageNet **training** data. The official
50,000-image validation set is not touched during selection, and no OOD data is read
before `selected_config.json` is frozen.
"""


def stage_b(args, out: Path, refined: list[float] | None = None):
    summ = json.loads((out / "selection" / "stage_a_summary.json").read_text())
    refined = refined or summ["refined_grid"]
    if not refined:
        raise SystemExit("no ID-stable scale found in Stage A; nothing to refine")
    s = Searcher(out, seed=args.seed)
    rows = []
    for r in refined:
        for n_cal in STAGE_B_NCAL:
            for lam in STAGE_B_LAMBDAS:
                row = s.evaluate("B", r, lam, n_cal, M=10)
                rows.append(row)
                _print(row, s.base_top1)
    _write_rows(out / "selection" / "all_stage_b.csv", rows)
    sel = select_config(rows, out, s.base_top1)
    return sel


def select_config(rows: list[dict], out: Path, base_top1: float) -> dict:
    """Frozen selection rule (spec §13). Applied to ID metrics only."""
    survivors = [r for r in rows if r["passes_gate"]]
    if not survivors:
        raise SystemExit("no Stage-B configuration passed the ID-stability gate")
    best_nll = min(r["nll"] for r in survivors)
    tied = [r for r in survivors if r["nll"] - best_nll < 0.002]
    # tie-breaks, in frozen order: larger scale, then n_cal=16384, then smaller ridge
    tied.sort(key=lambda r: (-r["r_target"], r["n_cal"] != 16384, r["lambda"]))
    chosen = tied[0]
    cfg = {
        "r_target": chosen["r_target"], "r_realized_median": chosen["r_realized_median"],
        "scale": chosen["scale"], "n_cal": chosen["n_cal"], "lambda": chosen["lambda"],
        "K": 20, "M_final": 20, "token_mode": "cls", "target_block": 11,
        "selection_nll": chosen["nll"], "selection_top1": chosen["top1"],
        "base_top1_selection_pool": base_top1,
        "n_survivors": len(survivors), "n_tied": len(tied),
        "ood_data_accessed_before_freeze": False,
    }
    fc.write_json(out / "selection" / "selected_config.json", cfg)
    (out / "selection" / "SELECTION_REPORT.md").write_text(
        _selection_report(rows, survivors, tied, chosen, base_top1))
    print(f"\n  SELECTED r={chosen['r_target']} lambda={chosen['lambda']} "
          f"n_cal={chosen['n_cal']}  (NLL {chosen['nll']:.4f}, "
          f"{len(survivors)} survivors, {len(tied)} tied)")
    return cfg


def _selection_report(rows, survivors, tied, chosen, base_top1) -> str:
    lines = [
        "# ID-only configuration selection", "",
        "**OOD data accessed before configuration freeze: NO**", "",
        "Selection used only the 8,192-image ID selection pool drawn from ImageNet",
        "*training* data. The official validation set and every OOD dataset were untouched",
        "at the time this configuration was frozen.", "",
        f"- Stage-B configurations evaluated: {len(rows)}",
        f"- Passing the ID-stability gate: {len(survivors)}",
        f"- Tied on NLL (within 0.002): {len(tied)}", "",
        "## Rule (frozen before OOD evaluation)", "",
        "1. apply the ID-stability gate (`ID_STABILITY_RULE.md`)",
        "2. lowest ID-selection NLL",
        "3. ties within 0.002 NLL broken by: larger scale, then n_cal=16,384, then smaller ridge",
        "", "## Selected", "",
        f"- realized r = {chosen['r_realized_median']:.4f} (target {chosen['r_target']})",
        f"- lambda = {chosen['lambda']}", f"- n_cal = {chosen['n_cal']}",
        f"- ID-selection NLL = {chosen['nll']:.4f}, top-1 = {chosen['top1']*100:.3f}% "
        f"(base {base_top1*100:.3f}%)", "",
        "## All Stage-B configurations", "",
        "| r | lambda | n_cal | top-1 | NLL | ECE | agree | median MSE | p99 MSE | gate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["r_target"], x["n_cal"], x["lambda"])):
        lines.append(
            f"| {r['r_target']:g} | {r['lambda']:g} | {r['n_cal']} | "
            f"{r['top1']*100:.3f} | {r['nll']:.4f} | {r['ece']:.4f} | "
            f"{r['base_agreement']:.4f} | {r['logit_mse_median']:.2e} | "
            f"{r['logit_mse_p99']:.2e} | {'PASS' if r['passes_gate'] else 'fail'} |")
    return "\n".join(lines) + "\n"


def robustness(args, out: Path):
    """Local ID-only sensitivity around the selected point (spec §14, diagnostic only).

    This never changes the frozen selection; it only reports whether the chosen operating
    point is isolated or sits on a plateau.
    """
    cfg = json.loads((out / "selection" / "selected_config.json").read_text())
    s = Searcher(out, seed=args.seed)
    rows = []
    for mult in (0.8, 1.0, 1.2):
        r = round(cfg["r_target"] * mult, 6)
        row = s.evaluate("robustness", r, cfg["lambda"], cfg["n_cal"], M=10)
        row["stage"] = f"robustness_{mult:g}x"
        rows.append(row)
        _print(row, s.base_top1)
    _write_rows(out / "selection" / "robustness_local.csv", rows)
    fc.write_json(out / "selection" / "robustness_local.json",
                  {"selected_r": cfg["r_target"], "multipliers": [0.8, 1.0, 1.2],
                   "note": "diagnostic only; the frozen selection is not revisited",
                   "rows": [{k: r[k] for k in ("stage", "r_target", "top1", "nll", "ece",
                                               "base_agreement", "logit_mse_median",
                                               "logit_mse_p99", "passes_gate")}
                            for r in rows]})
    print(f"\n  wrote {out/'selection'/'robustness_local.csv'}")
    return rows


def dispatch(stage: str, args, extra, out: Path, batch: int):
    if stage == "stage_a":
        stage_a(args, out)
    elif stage == "stage_b":
        stage_b(args, out)
    elif stage == "robustness":
        robustness(args, out)
    else:
        raise SystemExit(f"unknown stage {stage!r}")
