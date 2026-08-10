"""OOD evaluation of the FROZEN configs (global-grid + coordinate-descent) on OpenOOD v1.5.
Runs only after selected/global_config.json exists (config frozen). Reuses CIFAROpenOODPnC (same
OOD scoring + ID-only temperature as the submitted P&C table). Writes final_metrics/.
"""
from __future__ import annotations
import argparse, json, glob, time
from pathlib import Path
import numpy as np, luigi
REPO = Path(__file__).resolve().parents[2]
import sys, os; os.chdir(REPO); sys.path.insert(0, str(REPO))
from cifar_tasks import CIFAROpenOODPnC

ROOT = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "pnc_full_grid"
CD = dict(label="s3b0", stage_idx=3, block_idx=0, scale=25.0, bootstrap_frac=0.05)


def chunk_for(s): return 64 if s >= 3 else 16


def _parse(e, tag, seed, stage_idx, block_idx, scale, frac, runtime):
    idm = e["id_metrics"]
    def pd(fam):
        return {d: {"auroc": v["scores"]["predictive_entropy"]["auroc"] * 100,
                    "fpr95": v["scores"]["predictive_entropy"]["fpr95"] * 100}
                for d, v in e[fam]["per_dataset"].items()}
    return dict(tag=tag, seed=seed, stage_idx=stage_idx, block_idx=block_idx, scale=scale, frac=frac,
                id_acc=idm["accuracy"] * 100, id_nll=idm["nll"], id_ece=idm["ece"],
                id_brier=idm.get("brier"), temperature=idm.get("posthoc_temperature"),
                near_auroc=e["near_ood_auroc"] * 100, far_auroc=e["far_ood_auroc"] * 100,
                near_fpr95=e["near_ood"]["aggregate"]["predictive_entropy"]["mean_fpr95"] * 100,
                far_fpr95=e["far_ood"]["aggregate"]["predictive_entropy"]["mean_fpr95"] * 100,
                per_dataset=dict(near_ood=pd("near_ood"), far_ood=pd("far_ood")),
                runtime_sec=runtime)


def eval_config(tag, stage_idx, block_idx, scale, frac, seed):
    out_dir = ROOT / "final_metrics" / tag / f"seed_{seed}"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "openood.json"
    t0 = time.time()
    if not out_path.exists():
        task = CIFAROpenOODPnC(dataset="cifar10", epochs=300, perturbation_sizes=[float(scale)],
            n_directions=20, n_perturbations=50, subset_size=1024, chunk_size=chunk_for(stage_idx),
            target_stage_idx=stage_idx, target_block_idx=block_idx, random_directions=True, seed=seed,
            lambda_reg=1e-3, posthoc_calibrate=True, bootstrap_frac=float(frac))
        task.output = lambda p=str(out_path): luigi.LocalTarget(p)  # type: ignore
        task.run()
    e = json.load(open(out_path))[str(float(scale))]
    res = _parse(e, tag, seed, stage_idx, block_idx, scale, frac, round(time.time() - t0, 1))
    json.dump(res, open(out_dir / "summary.json", "w"), indent=2)
    print(f"[ood] {tag} seed{seed}: acc {res['id_acc']:.2f} Near {res['near_auroc']:.2f}/{res['near_fpr95']:.2f} "
          f"Far {res['far_auroc']:.2f}/{res['far_fpr95']:.2f} ({res['runtime_sec']:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()
    sel = json.load(open(ROOT / "selected" / "global_config.json"))
    g = sel["global_config"]
    configs = [("global_grid", g["stage_idx"], g["block_idx"], g["scale"], g["bootstrap_frac"]),
               ("coordinate_descent", CD["stage_idx"], CD["block_idx"], CD["scale"], CD["bootstrap_frac"])]
    # dedupe if identical
    seen = set(); uniq = []
    for c in configs:
        k = c[1:]
        if k not in seen: seen.add(k); uniq.append(c)
    for tag, s, b, sc, f in uniq:
        for seed in args.seeds:
            eval_config(tag, s, b, sc, f, seed)
    print("[ood] DONE", flush=True)


if __name__ == "__main__":
    main()
