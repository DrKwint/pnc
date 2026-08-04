"""Suite-wide one-factor P&C sensitivity sweep (4 submitted MuJoCo envs).

Anchor (submitted/canonical): K=20, M=50, correction_subset=full(10000), λ=0
(toward-zero), bootstrap=0.1, multi-block [l1,l3], random directions, prob base;
perturbation scale selected per (env,seed) by best validation NLL over {5,10,20,50}.

Factors (each varies ONE, all else at anchor):
  scale  : {0.25,0.5,1,2,4} × anchor scale
  rank   : K ∈ {1,2,5,20,40}
  bootstrap: b ∈ {0,0.05,0.1,0.2,0.3,0.5,0.99}
  calib  : n_cal ∈ {100,200,256,400,1000,4096}, bootstrap=0 (clean, without replacement)
  ridge  : λ ∈ {0,1e-4,1e-3,1e-2,1e-1,1} (toward-zero convention)
  layer  : {first=[l1], multi=[l1,l3]}

Emits one CSV row per (env,seed,factor,value) with the Section-13 schema, plus
per-cell JSON under raw/ and failures under failures/.
"""
from __future__ import annotations
import argparse, csv, json, sys, time, traceback
from pathlib import Path
import numpy as np
import jax.numpy as jnp

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "scripts"))
from pnc_theory import harness as H                                  # noqa: E402
from pnc_theory.linalg import Correction                            # noqa: E402
from util import _split_data, _predictive_mean_var                  # noqa: E402
from metrics import compute_ood_metrics, compute_nll               # noqa: E402
from scipy.stats import spearmanr                                 # noqa: E402

OUT = _ROOT / "results/neurips_2026_rebuttal/mujoco_sensitivity"
GIT = "70fb480-calibfix"  # base commit 70fb480 + calib one-at-a-time fix (uncommitted working tree)
# Submitted per-env anchor: lreg=ridge (toward ORIGINAL), bf=bootstrap fraction,
# ps=perturbation scale. K=20, M=50, multi-block, random dirs, subset=4096 for all.
PC_HPARAMS = {
    "Ant-v5":                    {"lreg": 1e-4, "bf": 0.10, "ps": 8.0},
    "HalfCheetah-v5":            {"lreg": 1e-4, "bf": 0.30, "ps": 32.0},
    "Hopper-v5":                 {"lreg": 1e-4, "bf": 0.30, "ps": 32.0},
    "Walker2d-v5":               {"lreg": 1e-2, "bf": 0.30, "ps": 16.0},
    "Swimmer-v5":                {"lreg": 1e-4, "bf": 0.05, "ps": 8.0},
    "Humanoid-v5":               {"lreg": 1e-2, "bf": 0.30, "ps": 32.0},
    "HumanoidStandup-v5":        {"lreg": 1e-2, "bf": 0.20, "ps": 8.0},
    "Reacher-v5":                {"lreg": 1e-4, "bf": 0.05, "ps": 8.0},
    "Pusher-v5":                 {"lreg": 1e-4, "bf": 0.10, "ps": 8.0},
    "InvertedPendulum-v5":       {"lreg": 1e-4, "bf": 0.05, "ps": 16.0},
    "InvertedDoublePendulum-v5": {"lreg": 1e-4, "bf": 0.05, "ps": 16.0},
}
ANCHOR = dict(K=20, M=50, n_cal=4096, layers=[0, 2], toward_orig=True)  # ridge toward ORIGINAL
OOD = ["ood_near", "ood_mid", "ood_far"]

# Per-factor grid = (base, extra). `extra` = extra near-anchor points that densify
# each factor's curve around the operating value; run standalone for backfill.
GRIDS = {
    "scale":     ([0.25, 0.5, 1.0, 2.0, 4.0], [0.75, 1.5]),        # × anchor scale
    "rank":      ([1, 2, 5, 20, 40], [10, 15, 30]),                # K (anchor 20)
    "bootstrap": ([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.99], [0.075, 0.15]),  # (anchor 0.1)
    "calib":     ([512, 1024, 2048, 4096, 8192], []),  # calibration POOL size N (16384 > 10000 cap, excluded)
    # bootfull: pool pinned to FULL id_train (10000); vary ONLY bootstrap fraction.
    # per-member rows = bf*10000, so bf=0.02 -> n/p=1.0 (interpolation peak); bf=0 = clean full.
    "bootfull":  ([0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.99], []),
    "ridge":     ([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], [1e-5, 3e-4, 3e-3]),  # (anchor 0)
    "layer":     (["first", "multi"], []),                          # categorical, no extra
}

CSV_COLS = ["environment", "seed", "factor", "factor_value", "factor_value_numeric",
            "anchor_value", "relative_to_anchor", "status", "error_message", "git_commit",
            "ridge_center", "scale", "subspace_dimension_K", "ensemble_size_M",
            "bootstrap_fraction", "calibration_size", "calibration_pool_size",
            "per_member_calibration_size", "unique_calibration_rows",
            "feature_dimension_p", "nominal_n_over_p", "unique_n_over_p", "layer_scope",
            "direction_family", "id_rmse", "id_nll", "id_total_variance", "id_residual",
            "near_rmse", "near_nll", "near_auroc", "mid_rmse", "mid_nll", "mid_auroc",
            "far_rmse", "far_nll", "far_auroc", "far_spearman", "construction_time_sec",
            "gram_rank", "gram_min_eig", "gram_condition", "normalized_ridge",
            "mean_delta_w_norm", "notes"]


def build_eval(base, ds, va, seed, *, size, K, M, n_cal, lam, boot, layers, toward_orig):
    """Build P&C and evaluate on val + id_eval + OOD tiers; return metric dict + diagnostics."""
    t0 = time.time()
    n_avail = int(np.asarray(ds["id_train"][0]).shape[0])
    if n_cal > n_avail:  # fail loudly rather than silently truncate the pool
        raise ValueError(f"calibration pool N={n_cal} exceeds available id_train rows ({n_avail})")
    ens = H.build_pnc(base, ds, seed, pert_size=size, n_members=M, lambda_reg=lam,
                      ridge_toward_orig=toward_orig, bootstrap_frac=boot, n_cal=n_cal,
                      perturb_indices=layers, n_directions=K)
    ctime = time.time() - t0
    out = {"construction_time_sec": ctime}

    def _eval(X, Y):
        preds = ens.predict(jnp.asarray(X))
        mean, var = _predictive_mean_var(preds)
        pv = np.array(jnp.mean(var, axis=-1))                 # per-point predictive var
        sq = np.array(jnp.mean((mean - Y) ** 2, axis=-1))     # per-point mean sq error
        return (float(jnp.sqrt(jnp.mean((mean - Y) ** 2))), float(compute_nll(mean, var, Y)),
                pv, float(jnp.mean(var)), sq)
    # val (for size selection)
    r, n, _, _, _ = _eval(va[0], va[1]); out["rmse_val"], out["nll_val"] = r, n
    # id
    Xi, Yi = np.asarray(ds["id_eval"][0]), np.asarray(ds["id_eval"][1])
    r, n, pv_id, tv, _ = _eval(Xi, Yi)
    out.update(id_rmse=r, id_nll=n, id_total_variance=tv)
    # ood tiers (+ Far uncertainty↔error Spearman)
    for reg in OOD:
        if reg not in ds:
            out[f"{reg}"] = None; continue
        X, Y = np.asarray(ds[reg][0]), np.asarray(ds[reg][1])
        rr, nn, pv, _, sq = _eval(X, Y)
        au = float(compute_ood_metrics(pv_id, pv)[0])
        d = dict(rmse=rr, nll=nn, auroc=au)
        if reg == "ood_far":
            d["spearman"] = float(spearmanr(pv, sq).statistic)  # does uncertainty rank error?
        out[reg] = d
    # construction diagnostics (block 0) + residual
    bp = H.extract_block_problem(ens, 0, 0)
    c = Correction(bp.X, bp.Xv, bp.Theta, lam=lam, mode=("toward_orig" if toward_orig else "toward_zero"))
    sp = c.spectrum()
    hb, hvb = H.block_reps_for_inputs(ens, 0, 0, Xi[:512])
    r_id = c.residual_formula(H.aug(hb), H.aug(hvb))
    out["gram_rank"] = sp["numerical_rank"]; out["gram_min_eig"] = sp["s_min"] ** 2
    out["gram_condition"] = sp["cond_G"]; out["feature_dimension_p"] = c.p
    out["id_residual"] = float(np.sqrt((r_id ** 2).mean()))
    out["normalized_ridge"] = lam / (sp["trace_G"] / c.p) if lam > 0 else 0.0
    out["mean_delta_w_norm"] = float(np.mean([np.linalg.norm(np.asarray(ens.seq_dWs[j][i]))
                                              for j in range(len(layers)) for i in range(min(M, 8))]))
    n_pool = int(np.asarray(ens.X_sub).shape[0])
    # Disambiguated columns (populated for EVERY factor):
    #   calibration_pool_size        = N (the pool X_sub is drawn from)
    #   per_member_calibration_size  = rows each ensemble member fits on
    #                                  = max(8, int(bf*N)) with bootstrap, else N
    out["calibration_pool_size"] = n_pool
    if boot > 0:
        nominal = max(8, int(boot * n_pool))                     # actual per-member sample size
        out["per_member_calibration_size"] = nominal
        out["calibration_size"] = nominal                        # legacy column (per-member when bootstrapping)
        out["unique_calibration_rows"] = int(round(n_pool * (1 - (1 - 1.0 / n_pool) ** nominal)))
    else:
        out["per_member_calibration_size"] = n_pool
        out["calibration_size"] = n_pool                         # legacy column (pool when no bootstrap)
        out["unique_calibration_rows"] = n_pool                  # clean WOR subset -> all unique
    return out


def anchor_scale(base, ds, va, seed):
    best, best_nll = None, np.inf
    for s in STD_SIZES:
        m = build_eval(base, ds, va, seed, size=s, K=ANCHOR["K"], M=ANCHOR["M"],
                       n_cal=ANCHOR["n_cal"], lam=0.0, boot=ANCHOR["boot"],
                       layers=ANCHOR["layers"], toward_orig=False)
        if m["nll_val"] < best_nll:
            best_nll, best = m["nll_val"], s
    return best


def _row(env, seed, factor, value, num, anchor_val, cfg, m, status="ok", err=""):
    lay = "multi" if cfg["layers"] == [0, 2] else ("first" if cfg["layers"] == [0] else str(cfg["layers"]))
    row = {c: "" for c in CSV_COLS}
    row.update(environment=env, seed=seed, factor=factor, factor_value=value,
               factor_value_numeric=num, anchor_value=anchor_val,
               relative_to_anchor=(num / anchor_val if (isinstance(num, (int, float)) and anchor_val) else ""),
               status=status, error_message=err, git_commit=GIT, ridge_center="zero",
               scale=cfg["size"], subspace_dimension_K=cfg["K"], ensemble_size_M=cfg["M"],
               bootstrap_fraction=cfg["boot"], layer_scope=lay, direction_family="random",
               normalized_ridge=m.get("normalized_ridge", 0.0) if m else "")
    if m:
        p = m.get("feature_dimension_p")
        row.update(calibration_size=m.get("calibration_size"),
                   calibration_pool_size=m.get("calibration_pool_size"),
                   per_member_calibration_size=m.get("per_member_calibration_size"),
                   unique_calibration_rows=m.get("unique_calibration_rows"),
                   feature_dimension_p=p,
                   nominal_n_over_p=(m.get("calibration_size") / p if p else ""),
                   unique_n_over_p=(m.get("unique_calibration_rows") / p if p else ""),
                   id_rmse=m["id_rmse"], id_nll=m["id_nll"], id_total_variance=m["id_total_variance"],
                   id_residual=m.get("id_residual"), construction_time_sec=m.get("construction_time_sec"),
                   gram_rank=m.get("gram_rank"), gram_min_eig=m.get("gram_min_eig"),
                   gram_condition=m.get("gram_condition"), mean_delta_w_norm=m.get("mean_delta_w_norm"))
        for reg, pre in [("ood_near", "near"), ("ood_mid", "mid"), ("ood_far", "far")]:
            d = m.get(reg)
            if d:
                row[f"{pre}_rmse"], row[f"{pre}_nll"], row[f"{pre}_auroc"] = d["rmse"], d["nll"], d["auroc"]
                if pre == "far":
                    row["far_spearman"] = d.get("spearman", "")
            else:
                row[f"{pre}_rmse"] = row[f"{pre}_nll"] = row[f"{pre}_auroc"] = "NA"
    return row


def run(env, seed, factors, out_csv, extra_only=False, tag=""):
    ds = H.load_dataset(env, seed)
    base = H.get_base_model(env, seed, ds)
    x_tr, y_tr, x_va, y_va = _split_data(*ds["id_train"])
    va = (np.asarray(x_va), np.asarray(y_va))
    A = dict(ANCHOR)
    hp = PC_HPARAMS[env]
    a_ps, a_bf, a_lreg = hp["ps"], hp["bf"], hp["lreg"]
    a_size = a_ps
    print(f"[{env} s{seed}] anchor: ps={a_ps} bf={a_bf} lreg={a_lreg} (toward-orig) subset={A['n_cal']}"
          f"{' (extra-only backfill)' if extra_only else ''}")
    rows = []

    def vals(fac):
        base_v, extra_v = GRIDS[fac]
        return extra_v if extra_only else (base_v + extra_v)

    def cell(factor, value, num, anchor_val, **over):
        cfg = dict(size=a_ps, K=A["K"], M=A["M"], n_cal=A["n_cal"], lam=a_lreg,
                   boot=a_bf, layers=A["layers"], toward_orig=A["toward_orig"])
        cfg.update(over)
        try:
            m = build_eval(base, ds, va, seed, **cfg)
            r = _row(env, seed, factor, value, num, anchor_val, cfg, m)
        except Exception as e:
            (OUT / "failures").mkdir(exist_ok=True)
            (OUT / "failures" / f"{env}_s{seed}_{factor}_{value}.log").write_text(traceback.format_exc())
            r = _row(env, seed, factor, value, num, anchor_val, cfg, None, status="FAIL", err=str(e)[:200])
        rows.append(r); return r

    if "scale" in factors:
        for mult in vals("scale"):
            cell("scale", f"{mult}x", round(a_ps * mult, 4), a_ps, size=round(a_ps * mult, 4))
    if "rank" in factors:
        for K in vals("rank"):
            cell("rank", K, K, 20, K=K)
    if "bootstrap" in factors:
        for b in vals("bootstrap"):
            cell("bootstrap", b, b, a_bf, boot=b)
    if "calib" in factors:
        for N in vals("calib"):
            # one-at-a-time: vary ONLY the calibration pool N; keep bootstrap at the
            # env anchor (a_bf, carried by cell's base cfg). Anchor = A["n_cal"]=4096.
            cell("calib", N, N, A["n_cal"], n_cal=N)
    if "bootfull" in factors:
        # full-data calibration pool (n_cal = all id_train rows); vary only bootstrap frac
        n_full = int(np.asarray(ds["id_train"][0]).shape[0])
        for b in vals("bootfull"):
            cell("bootfull", b, b, a_bf, boot=b, n_cal=n_full)
    if "ridge" in factors:
        for lam in vals("ridge"):
            cell("ridge", lam, lam, a_lreg, lam=lam)
    if "layer" in factors:
        for name in vals("layer"):
            ly = [0] if name == "first" else [0, 2]
            cell("layer", name, (1 if name == "first" else 2), 2, layers=ly)

    # write per-cell JSON + append CSV
    (OUT / "raw").mkdir(exist_ok=True)
    _suf = f"_{tag}" if tag else ""
    (OUT / "raw" / f"{env}_seed{seed}{_suf}.json").write_text(json.dumps(rows, indent=2, default=str))
    write_header = not Path(out_csv).exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[{env} s{seed}] {len(rows)} rows -> {out_csv}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--factors", default="scale,rank,bootstrap,calib,ridge,layer")
    ap.add_argument("--extra-only", action="store_true", help="run only the near-anchor extra grid points (backfill)")
    ap.add_argument("--out-csv", default=str(OUT / "aggregates" / "mujoco_sensitivity_raw.csv"))
    ap.add_argument("--tag", default="", help="suffix for the per-seed raw JSON dump (avoids clobbering)")
    a = ap.parse_args()
    run(a.env, a.seed, a.factors.split(","), a.out_csv, extra_only=a.extra_only, tag=a.tag)
