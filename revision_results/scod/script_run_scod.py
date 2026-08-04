"""Phase I: SCOD (native score + ID-calibrated Gaussian) on the MuJoCo benchmark.

Per (env, seed):
  1. load shared base checkpoint + data (common.load_env_seed);
  2. build one max-rank randomized Fisher sketch on the 4096 ID calibration subset;
  3. cache per-point Fisher energy on train / id-val / eval / OOD splits (once);
  4. select (k, Meps, alpha) by ID-validation NLL of the SCOD-Gaussian predictive
     (OOD never touched during selection); freeze config to disk;
  5. evaluate both variants on all available splits and save per-example artifacts.

Variants reported:
  SCOD-native          : ID RMSE (base mean), Far AUROC + Spearman from native u(x), NLL = N/A
  SCOD-Gaussian (ID cal): NLL from Sigma_a + alpha * qtilde * D_y ; AUROC/Spearman from native u
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

import common as C
from scod_adapter import SCODModel, build_sketch, scod_energy, score_from_energy

_ROOT = Path(__file__).resolve().parents[2]
GIT_TAG = "70fb480-posthoc"


def _select_config(energy, base, y_val, Dy, N, k_grid, Meps_vals, alpha_grid):
    """ID-only selection of (k, Meps, alpha) minimizing ID-val Gaussian NLL.

    energy: dict split->(fro2, pe). base: dict split->(mean,var). Uses only
    'train' (median normaliser) and 'val'. Tie-breaks: smaller k, smaller alpha,
    Meps closest to N.
    """
    fro2_tr, pe_tr = energy["train"]
    fro2_va, pe_va = energy["val"]
    mean_va, var_va = base["val"]
    best = None
    for k in k_grid:
        for Meps in Meps_vals:
            q_tr = score_from_energy(fro2_tr, pe_tr, energy["eigs"], k=k, Meps=Meps) ** 2
            med = float(np.median(q_tr)) + C.EPS
            q_va = score_from_energy(fro2_va, pe_va, energy["eigs"], k=k, Meps=Meps) ** 2
            qtil = q_va / med                                    # (Nval,)
            for alpha in alpha_grid:
                var_tot = var_va + alpha * qtil[:, None] * Dy[None, :]
                nll = C.diag_nll(mean_va, var_tot, y_val)
                key = (nll, k, alpha, abs(Meps - N))             # tie-break order
                if best is None or key < best[0]:
                    best = (key, dict(k=k, Meps=Meps, alpha=alpha, med_q=med, nll=nll))
    return best[1]


def run_one(env, seed, cfg, out_dir: Path, score_max_points=None):
    t0 = time.time()
    scfg = cfg["scod"]
    load = C.load_env_seed(env, seed, n_cal=cfg["data"]["calibration_pool_size"])
    # optional scoring subsample for extreme-out_dim envs (Humanoid/HumanoidStandup,
    # out_dim=348): the per-point Jacobian is 348*2 x P, so full 10k-point scoring is
    # prohibitive. A fixed 3000-point subsample is statistically ample for AUROC/Spearman.
    subsampled = False
    if score_max_points:
        rs = np.random.RandomState(12345)
        new_splits = {}
        for sp, (X, Y) in load["splits"].items():
            if len(X) > score_max_points:
                idx = rs.choice(len(X), score_max_points, replace=False)
                new_splits[sp] = (X[idx], Y[idx]); subsampled = True
            else:
                new_splits[sp] = (X, Y)
        load["splits"] = new_splits
    model, Xcal, N = load["model"], load["Xcal"], load["n_cal"]
    Dy = load["Dy"]

    scod = SCODModel(model)                                       # Case B (heteroscedastic)
    sketch_seed = scfg["sketch_seed_base"] + seed
    t_sketch = time.time()
    eigs, basis = build_sketch(scod, Xcal, num_eigs_max=scfg["num_eigs_max"],
                               num_samples=scfg["num_samples"], sketch_seed=sketch_seed)
    sketch_secs = time.time() - t_sketch
    num_rank = int(np.sum(eigs > eigs[0] * 1e-10)) if eigs[0] > 0 else 0
    k_grid = [k for k in scfg["k_grid"] if k <= len(eigs)]

    # per-point Fisher energy (computed once per set, shared across k/Meps).
    # Adaptive point-chunk keeps the transient (chunk x out_transformed x P) Jacobian
    # under ~300MB so the EXACT scorer never OOMs, even at out_dim=348 (Humanoid).
    out_t = 2 * load["out_dim"]                                   # Case B: 2k transformed coords
    chunk = max(1, int(3e8 / (out_t * scod.P * 4)))
    print(f"[SCOD {env} s{seed}] P={scod.P} out_t={out_t} -> score point_chunk={chunk}")

    def _energy(X):
        return scod_energy(scod, X, basis, point_chunk=chunk)

    energy = {"eigs": eigs}
    base = {}
    energy["train"] = _energy(Xcal); base["train"] = C.base_predict(model, Xcal)
    energy["val"] = _energy(load["x_val"]); base["val"] = C.base_predict(model, load["x_val"])
    for split, (X, Y) in load["splits"].items():
        energy[split] = _energy(X)
        base[split] = C.base_predict(model, X)

    Meps_vals = [float(f) * N for f in scfg["Meps_factors"]]
    alpha_grid = [float(a) for a in scfg["alpha_grid"]]
    sel = _select_config(energy, base, load["y_val"], Dy, N,
                         k_grid, Meps_vals, alpha_grid)
    k, Meps, alpha, med_q = sel["k"], sel["Meps"], sel["alpha"], sel["med_q"]

    # ---- freeze config BEFORE any OOD metric is read ----
    cfg_dir = out_dir / "configs" / env; cfg_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": "SCOD", "variant": "native_and_id_calibrated_gaussian",
        "environment": env, "model_seed": seed,
        "num_train_examples": int(N), "n_parameters": int(scod.P),
        "num_eigs_max": scfg["num_eigs_max"], "selected_num_eigs": int(k),
        "num_samples": scfg["num_samples"], "recovered_numerical_rank": int(num_rank),
        "selected_Meps": float(Meps), "selected_Meps_factor": float(Meps / N),
        "selected_alpha": float(alpha), "sketch_seed": int(sketch_seed),
        "id_val_nll_at_selection": float(sel["nll"]),
        "selection_metric": "ID validation NLL", "used_ood_for_selection": False,
        "case": "B_heteroscedastic_gaussian", "git_tag": GIT_TAG,
        "score_subsampled": bool(subsampled),
        "score_max_points": int(score_max_points) if score_max_points else None,
    }
    (cfg_dir / f"{seed}.json").write_text(json.dumps(config, indent=2))

    # spectrum + sketch
    sp_dir = out_dir / "spectra" / env; sp_dir.mkdir(parents=True, exist_ok=True)
    np.savez(sp_dir / f"{seed}.npz", eigs=eigs)
    sk_dir = out_dir / "sketches" / env; sk_dir.mkdir(parents=True, exist_ok=True)
    np.savez(sk_dir / f"{seed}.npz", eigs=eigs, basis=basis, sketch_seed=sketch_seed)

    # ---- evaluate both variants on all splits ----
    def native_u(split):
        fro2, pe = energy[split]
        return score_from_energy(fro2, pe, eigs, k=k, Meps=Meps)

    def qtilde(split):
        return native_u(split) ** 2 / med_q

    mean_ie, var_ie = base["id_eval"]
    y_ie = load["splits"]["id_eval"][1]
    id_rmse = float(np.sqrt(np.mean((mean_ie - y_ie) ** 2)))
    u_id = native_u("id_eval")

    pred_dir = out_dir / "predictions" / env / str(seed)
    metrics = {"id_eval": {"rmse": id_rmse}}
    base_mean_max_dev = 0.0
    for split, (X, Y) in load["splits"].items():
        mean_s, var_a = base[split]
        u = native_u(split)
        var_epi = alpha * qtilde(split)[:, None] * Dy[None, :]
        var_tot = var_a + var_epi
        nll = C.diag_nll(mean_s, var_tot, Y)
        rec = {"nll_gaussian": nll, "mean_uncertainty": float(np.mean(u)),
               "median_uncertainty": float(np.median(u))}
        if split != "id_eval":
            rec["auroc_native"] = C.uncertainty_auroc(u_id, u)
            if split == "ood_far":
                rec["spearman_native"] = C.far_spearman(u, C.per_point_sqerr(mean_s, Y))
        metrics[split] = rec
        C.save_predictions(pred_dir / split, split, mean=mean_s, var_a=var_a,
                           var_epi=var_epi, score=u, y=Y)
        base_mean_max_dev = max(base_mean_max_dev,
                                float(np.max(np.abs(mean_s - base[split][0]))))

    timing = {"construction_seconds": sketch_secs,
              "total_seconds": time.time() - t0, "n_parameters": int(scod.P),
              "num_eigs_max": scfg["num_eigs_max"], "num_samples": scfg["num_samples"],
              "base_mean_max_dev": base_mean_max_dev}
    tm_dir = out_dir / "timing" / env; tm_dir.mkdir(parents=True, exist_ok=True)
    (tm_dir / f"{seed}.json").write_text(json.dumps(timing, indent=2))

    # rollup row for aggregation
    roll = {"method": "SCOD", "environment": env, "model_seed": seed,
            "selected": config, "id_rmse": id_rmse, "metrics": metrics}
    (pred_dir / "_metrics.json").write_text(json.dumps(roll, indent=2, default=float))
    print(f"[SCOD {env} s{seed}] k={k} Meps={Meps:.0f}({Meps/N:.1f}N) alpha={alpha} "
          f"| id_rmse={id_rmse:.4f} far_auroc={metrics.get('ood_far',{}).get('auroc_native','NA')} "
          f"| sketch {sketch_secs:.1f}s tot {time.time()-t0:.1f}s")
    return roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--envs", default=None, help="comma list override")
    ap.add_argument("--seeds", default=None, help="comma list override")
    ap.add_argument("--scope", default=None, choices=["validation", "full"],
                    help="validation = cfg.validation_scope; full = cfg.environments x cfg.seeds")
    ap.add_argument("--score-max-points", type=int, default=None,
                    help="subsample per-split scoring points (for extreme out_dim envs)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.scope == "validation":
        envs = cfg["validation_scope"]["environments"]; seeds = cfg["validation_scope"]["seeds"]
    else:
        envs = cfg["environments"]; seeds = cfg["seeds"]
    if args.envs:
        envs = args.envs.split(",")
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]

    print(f"SCOD run: {len(envs)} envs x {len(seeds)} seeds -> {out_dir}")
    failures = []
    for env in envs:
        for seed in seeds:
            try:
                run_one(env, seed, cfg, out_dir, score_max_points=args.score_max_points)
            except Exception as e:
                import traceback
                (out_dir / "failures").mkdir(exist_ok=True)
                (out_dir / "failures" / f"{env}_s{seed}.log").write_text(traceback.format_exc())
                failures.append((env, seed, str(e)[:200]))
                print(f"[SCOD {env} s{seed}] FAILED: {str(e)[:160]}")
    (out_dir / "run_summary.json").write_text(json.dumps(
        {"envs": envs, "seeds": seeds, "failures": failures, "git_tag": GIT_TAG}, indent=2))
    print(f"SCOD run done. failures={len(failures)}")


if __name__ == "__main__":
    main()
