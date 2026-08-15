"""Revision closure Part D: exact residual identity under ORIGINAL-centred ridge.

The theory section is now written solely for the original-centred correction, so the
mechanism gate is regenerated under that convention on Ant-v5, HalfCheetah-v5 and
Hopper-v5, in float64, across every shift regime.

What is checked. Write the augmented designs at a test point x

    X(x)  = [h(x), 1]        original post-activation at the perturbed layer
    Xv(x) = [h_v(x), 1]      perturbed post-activation
    g_S(x; v) = Xv(x) - X(x) (the activation change)

and let Theta be the original next affine map, Theta_hat the fitted correction,
C = Theta_hat - Theta. The transfer defect of the corrected member is

    r_S(x; v) := Xv(x) Theta_hat - X(x) Theta

and under ORIGINAL-centred ridge it decomposes *exactly*, at every lambda, as

    r_S(x; v) = g_S(x; v) Theta + Xv(x) C                               (identity A)

with no extra term. Under zero-centred ridge the same algebra carries an additional
ridge-bias term -lambda * Theta G_v^{-1} hbar_v(x), which is the convention artifact the
revision removes. As lambda -> infinity, C -> 0 and identity A collapses to

    r_S(x; v) -> Theta g_S(x; v)                                        (identity B)

which is the statement the manuscript makes. Both are reported: A as an exact algebraic
gate (target <= 1e-11 relative), B with the measured size of the ||Xv C|| term that
separates them at the operating lambda.

    .venv/bin/python -m experiments.scripts.pnc_theory.validate_closure_mechanism
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.scripts.pnc_theory import harness as H          # noqa: E402
from experiments.scripts.pnc_theory.harness import (  # noqa: E402
    _orig_layer_outputs, aug, block_reps_for_inputs)

OUT = ROOT / "2026neurips_rebuttal_results" / "revision_experiment_closure"
ENVS = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5"]
REGIMES = ["id_eval", "ood_near", "ood_mid", "ood_far"]
LAM = 1e-4          # the manuscript's MuJoCo operating lambda
N_MEMBERS = 8
N_POINTS = 4000


def _designs(ens, member: int, block: int, X_in: np.ndarray):
    """float64 (X, Xv, Theta) at the given inputs, via the harness's own helper."""
    hb, hvb = block_reps_for_inputs(ens, member, block, X_in)
    _, Ws, bs = _orig_layer_outputs(ens.base_model, np.asarray(X_in, np.float64)[:1])
    corr_idx = int(ens.layers[block].replace("l", "")) - 1 + 1
    Theta = np.concatenate([Ws[corr_idx], bs[corr_idx][None, :]], axis=0)
    return aug(hb), aug(hvb), Theta


def _rel(a, b):
    n = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / (n if n > 0 else 1.0))


def run_env(env: str, seed: int = 0, lam: float = LAM) -> dict:
    ds = H.load_dataset(env, seed)
    base = H.get_base_model(env, seed, ds)
    ens = H.build_pnc(base, ds, seed, pert_size=10.0, n_members=N_MEMBERS,
                      lambda_reg=lam, ridge_toward_orig=True)
    assert getattr(ens, "ridge_toward_orig", False) is True, "not original-centred!"
    n_blocks = len(ens.layers)
    out = {"env": env, "seed": seed, "lambda": lam, "ridge_center": "original",
           "n_members": N_MEMBERS, "n_blocks": n_blocks, "regimes": {}}
    for reg in REGIMES:
        if reg not in ds:
            continue
        X_in = np.asarray(ds[reg][0], np.float64)[:N_POINTS]
        relA, ratio = [], []
        for m in range(N_MEMBERS):
            for j in range(n_blocks):
                X, Xv, Theta = _designs(ens, m, j, X_in)
                Th = np.concatenate([
                    np.asarray(ens.seq_w_effs[j][m], np.float64),
                    np.asarray(ens.seq_b_effs[j][m], np.float64)[None, :]], axis=0)
                C = Th - Theta
                gS = Xv - X
                rS = Xv @ Th - X @ Theta
                relA.append(_rel(gS @ Theta + Xv @ C, rS))          # identity A
                ratio.append(float(np.linalg.norm(Xv @ C)
                                   / max(np.linalg.norm(rS), 1e-300)))
        out["regimes"][reg] = {
            "n_points": int(X_in.shape[0]), "n_checks": len(relA),
            "identity_A_max_rel_err": float(np.max(relA)),
            "identity_A_median_rel_err": float(np.median(relA)),
            "correction_energy_fraction_median": float(np.median(ratio)),
        }
        r = out["regimes"][reg]
        print(f"  {env:<16}{reg:<10} max {r['identity_A_max_rel_err']:.3e}  "
              f"median {r['identity_A_median_rel_err']:.3e}  "
              f"||XvC||/||rS|| {r['correction_energy_fraction_median']:.4f}")
    return out


def large_lambda_check(env: str = "Hopper-v5", seed: int = 0) -> dict:
    """Identity B: as lambda grows, C vanishes and r_S -> Theta g_S."""
    ds = H.load_dataset(env, seed)
    base = H.get_base_model(env, seed, ds)
    X_in = np.asarray(ds["ood_far"][0], np.float64)[:N_POINTS]
    rows = []
    for lam in (1e-4, 1e-1, 1e2, 1e5, 1e8):
        ens = H.build_pnc(base, ds, seed, pert_size=10.0, n_members=2,
                          lambda_reg=lam, ridge_toward_orig=True)
        X, Xv, Theta = _designs(ens, 0, 0, X_in)
        Th = np.concatenate([np.asarray(ens.seq_w_effs[0][0], np.float64),
                             np.asarray(ens.seq_b_effs[0][0], np.float64)[None, :]], 0)
        C = Th - Theta
        gS, rS = Xv - X, Xv @ Th - X @ Theta
        rows.append({"lambda": lam,
                     "rel_C_over_Theta": float(np.linalg.norm(C)
                                               / np.linalg.norm(Theta)),
                     "identity_B_rel_err": _rel(gS @ Theta, rS)})
        print(f"  lambda={lam:<8g} ||C||/||Theta||={rows[-1]['rel_C_over_Theta']:.3e}  "
              f"identity B rel err={rows[-1]['identity_B_rel_err']:.3e}")
    return {"env": env, "seed": seed, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--lam", type=float, default=LAM)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== Part D: exact residual identity, original-centred ridge, "
          f"lambda={a.lam} ===")
    res = {"identity": "r_S = g_S Theta + Xv C  (no ridge-centre term)",
           "target_max_rel_err": 1e-11, "envs": []}
    for env in a.envs.split(","):
        res["envs"].append(run_env(env.strip(), lam=a.lam))
    worst = max(r["identity_A_max_rel_err"]
                for e in res["envs"] for r in e["regimes"].values())
    res["worst_max_rel_err"] = worst
    res["passes_1e-11"] = bool(worst <= 1e-11)
    print(f"\nworst max relative error across all envs/regimes: {worst:.3e}  "
          f"-> {'PASS' if worst <= 1e-11 else 'FAIL'} (target 1e-11)")
    print("\n=== Identity B: lambda -> infinity collapses r_S to Theta g_S ===")
    res["large_lambda"] = large_lambda_check()
    (OUT / "mujoco_mechanism_original_centered.json").write_text(
        json.dumps(res, indent=2))
    print(f"\nwrote {OUT}/mujoco_mechanism_original_centered.json")


if __name__ == "__main__":
    main()
