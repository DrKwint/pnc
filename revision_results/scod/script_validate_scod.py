"""SCOD validation gate (spec 4.8). Exits nonzero unless every check passes; the
orchestration script only creates SCOD_COMPLETE after this returns success.

Checks: completeness; no OOD used before freeze; base means preserved (<1e-6);
native scores finite & nonnegative; predictive variances finite & positive;
selected hyperparameters present; per-example outputs exist for every available
split; timing records exist; and a reproduce check (reload one sketch, recompute
the native id_eval score, confirm it matches the saved artifact).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOL_MEAN = 1e-6


def _load_scalars(p: Path):
    if p.with_suffix(".parquet").exists():
        import pandas as pd
        return pd.read_parquet(p.with_suffix(".parquet"))
    import numpy as _np
    d = _np.load(p.with_suffix(".scalars.npz"))
    return {k: d[k] for k in d.files}


def validate(result_dir: Path) -> list[str]:
    errs: list[str] = []
    cfg_files = sorted((result_dir / "configs").rglob("*.json"))
    if not cfg_files:
        return [f"no SCOD configs under {result_dir}/configs"]

    for cf in cfg_files:
        env, seed = cf.parent.name, cf.stem
        tag = f"{env}/s{seed}"
        cfg = json.loads(cf.read_text())

        # no OOD used before freeze
        if cfg.get("used_ood_for_selection") is not False:
            errs.append(f"{tag}: used_ood_for_selection != False")
        # selected hyperparameters present
        for key in ("selected_num_eigs", "selected_Meps", "selected_alpha", "sketch_seed"):
            if cfg.get(key) is None:
                errs.append(f"{tag}: missing selected hyperparameter {key}")

        # timing + base-mean preservation
        tf = result_dir / "timing" / env / f"{seed}.json"
        if not tf.exists():
            errs.append(f"{tag}: missing timing record")
        else:
            tm = json.loads(tf.read_text())
            if tm.get("base_mean_max_dev", 1.0) > TOL_MEAN:
                errs.append(f"{tag}: base mean deviates {tm['base_mean_max_dev']:.2e} > {TOL_MEAN}")

        # per-example outputs: at least id_eval + ood_far
        pred_dir = result_dir / "predictions" / env / seed
        present = {p.stem for p in pred_dir.glob("*.parquet")} | \
                  {p.name.split(".")[0] for p in pred_dir.glob("*.scalars.npz")}
        for need in ("id_eval", "ood_far"):
            if need not in present:
                errs.append(f"{tag}: missing predictions for {need}")

        # scores finite & nonneg; variances finite & positive
        for split in present:
            df = _load_scalars(pred_dir / split)
            u = np.asarray(df["uncertainty_score"]); v = np.asarray(df["mean_total_var"])
            if not np.all(np.isfinite(u)) or np.any(u < 0):
                errs.append(f"{tag}/{split}: native scores not all finite & nonnegative")
            if not np.all(np.isfinite(v)) or np.any(v <= 0):
                errs.append(f"{tag}/{split}: total predictive variance not all finite & positive")

    # reproduce check on the first config: reload sketch, recompute id_eval native score
    try:
        # prefer a small-out_dim, non-subsampled env so the reproduce recompute is cheap
        SMALL = ["InvertedPendulum-v5", "Swimmer-v5", "Reacher-v5", "InvertedDoublePendulum-v5"]
        pick = next((c for s in SMALL for c in cfg_files if c.parent.name == s), cfg_files[0])
        env, seed = pick.parent.name, int(pick.stem)
        cfg = json.loads(pick.read_text())
        sk = np.load(result_dir / "sketches" / env / f"{seed}.npz")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import common as C
        from scod_adapter import SCODModel, scod_energy, score_from_energy
        load = C.load_env_seed(env, seed)
        scod = SCODModel(load["model"])
        n_sub = 512                                          # subset -> cheap + matches saved order
        Xie = np.asarray(load["splits"]["id_eval"][0])[:n_sub]
        out_t = 2 * load["out_dim"]
        chunk = max(1, int(3e8 / (out_t * scod.P * 4)))      # adaptive: never OOM
        fro2, pe = scod_energy(scod, Xie, sk["basis"], point_chunk=chunk)
        u_re = score_from_energy(fro2, pe, sk["eigs"], k=cfg["selected_num_eigs"],
                                 Meps=cfg["selected_Meps"])
        saved = _load_scalars(result_dir / "predictions" / env / str(seed) / "id_eval")
        u_saved = np.asarray(saved["uncertainty_score"])[:n_sub]
        rel = np.max(np.abs(u_re - u_saved)) / (np.max(np.abs(u_saved)) + 1e-12)
        if rel > 1e-4:
            errs.append(f"reproduce check {env}/s{seed}: id_eval score rel-diff {rel:.2e} > 1e-4")
        else:
            print(f"[validate_scod] reproduce check {env}/s{seed}: rel-diff {rel:.2e} OK")
    except Exception as e:
        errs.append(f"reproduce check raised: {e}")

    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    args = ap.parse_args()
    errs = validate(Path(args.result_dir))
    if errs:
        print(f"SCOD VALIDATION FAILED ({len(errs)} issues):")
        for e in errs[:50]:
            print("  -", e)
        sys.exit(1)
    print("SCOD VALIDATION PASSED")


if __name__ == "__main__":
    main()
