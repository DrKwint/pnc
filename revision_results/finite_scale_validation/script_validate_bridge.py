"""Local-to-final residual bridge (amendment B.1).

The last perturbed block (l3→correct l4) produces a corrected l4-preactivation
z_v(x); the base produces z_0(x). Their difference is the exact block-1 local
residual r_1(x)=Θ̂_1 h̄_v(x)−Θ_1 h̄(x) (200-dim), and it captures the ENTIRE
deviation entering the downstream head F_down = mean_head∘relu. So the final
member deviation is exactly

    Δy(x) = F_down(z_0+r_1) − F_down(z_0),      J_down(z_0)=W_meanᵀ diag(1[z_0>0]).

For each env/member/regime we measure how well the local residual controls the
final prediction disagreement:
  Spearman(‖r_1‖,‖Δy‖), Pearson on logs, downstream gain ‖Δy‖/‖r_1‖,
  cosine(Δy, J_down r_1), relative Taylor error ‖Δy−J_down r_1‖/‖Δy‖,
  and counterexample rates (large local/small final and vice-versa).
Run at a small scale (linear downstream may hold) and the benchmark scale.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.scripts.pnc_theory import harness as H          # noqa: E402

REGIMES = ["id_eval", "ood_near", "ood_mid", "ood_far"]


def run(env, seed, scales, members, n_test, out_dir):
    ds = H.load_dataset(env, seed)
    base = H.get_base_model(env, seed, ds)
    Wm = np.asarray(base.mean_layer.kernel.get_value(), np.float64)   # (200, out)
    bm = np.asarray(base.mean_layer.bias.get_value(), np.float64)
    results = {"env": env, "seed": seed, "scales": scales, "per_scale": {}}

    for P in scales:
        ens = H.build_pnc(base, ds, seed, pert_size=P, n_members=max(members) + 1,
                          lambda_reg=0.0, ridge_toward_orig=True)
        j = len(ens.layers) - 1                          # last block (l3->l4)
        per_reg = {}
        for reg in REGIMES:
            if reg not in ds:
                continue
            X = np.asarray(ds[reg][0], np.float64)[:n_test]
            F0 = np.asarray(base(jnp.asarray(X))[0], np.float64)      # base mean (B,out)
            means = np.asarray(ens.predict(jnp.asarray(X))[0], np.float64)  # (M,B,out)
            rown, gain, cos, taylor, sp, pe = [], [], [], [], [], []
            big_local_small_final = big_final_small_local = 0
            tot = 0
            for i in members:
                bp = H.extract_block_problem(ens, i, j)
                hb, hvb = H.block_reps_for_inputs(ens, i, j, X)
                z0 = H.aug(hb) @ bp.Theta                            # orig l4 preact (B,200)
                zv = H.aug(hvb) @ bp.theta_hat_impl                  # corrected perturbed preact
                r1 = zv - z0                                          # (B,200) local residual
                dy = means[i] - F0                                    # (B,out) final deviation
                Jr = (np.maximum(np.sign(z0), 0.0) * r1) @ Wm         # J_down r_1  (B,out)
                rn = np.linalg.norm(r1, axis=1); dn = np.linalg.norm(dy, axis=1)
                rown.append((rn, dn))
                mask = dn > 1e-9
                gain.append(dn[mask] / (rn[mask] + 1e-12))
                cos.append(np.sum(dy * Jr, axis=1) / (dn * np.linalg.norm(Jr, axis=1) + 1e-12))
                taylor.append(np.linalg.norm(dy - Jr, axis=1) / (dn + 1e-12))
                # rank correlations per member
                if rn.std() > 0 and dn.std() > 0:
                    sp.append(spearmanr(rn, dn).statistic)
                    lr, ld = np.log(rn + 1e-12), np.log(dn + 1e-12)
                    pe.append(pearsonr(lr, ld).statistic)
                # counterexamples: quartile mismatches
                rq = np.quantile(rn, [0.25, 0.75]); dq = np.quantile(dn, [0.25, 0.75])
                big_local_small_final += int(np.sum((rn > rq[1]) & (dn < dq[0])))
                big_final_small_local += int(np.sum((dn > dq[1]) & (rn < rq[0])))
                tot += len(rn)
            per_reg[reg] = {
                "spearman_r_dy": float(np.median(sp)) if sp else None,
                "pearson_log": float(np.median(pe)) if pe else None,
                "gain_median": float(np.median(np.concatenate(gain))),
                "gain_p95": float(np.quantile(np.concatenate(gain), 0.95)),
                "cos_median": float(np.median(np.concatenate(cos))),
                "taylor_relerr_median": float(np.median(np.concatenate(taylor))),
                "frac_big_local_small_final": big_local_small_final / max(tot, 1),
                "frac_big_final_small_local": big_final_small_local / max(tot, 1),
            }
        results["per_scale"][str(P)] = per_reg

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"bridge_{env}_seed{seed}.json"
    fp.write_text(json.dumps(results, indent=2, default=float))

    print(f"\n{'='*80}\nLocal-to-final bridge — {env} seed{seed}\n{'='*80}")
    for P in scales:
        print(f"\nscale P={P}:")
        print(f"  {'regime':9s} {'ρ(|r|,|Δy|)':>11s} {'r_log':>7s} {'gain(med/p95)':>14s} "
              f"{'cos(Δy,Jr)':>10s} {'Taylor_relerr':>13s} {'big-loc/sml-fin':>15s}")
        for reg in REGIMES:
            if reg not in results["per_scale"][str(P)]:
                continue
            d = results["per_scale"][str(P)][reg]
            print(f"  {reg:9s} {d['spearman_r_dy'] if d['spearman_r_dy'] else 0:>11.2f} "
                  f"{d['pearson_log'] if d['pearson_log'] else 0:>7.2f} "
                  f"{d['gain_median']:>6.2f}/{d['gain_p95']:<7.2f} {d['cos_median']:>10.2f} "
                  f"{d['taylor_relerr_median']:>13.2f} "
                  f"{d['frac_big_local_small_final']:>7.1%}/{d['frac_big_final_small_local']:<7.1%}")
    print(f"\nwrote {fp}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Ant-v5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scales", default="0.1,10")
    ap.add_argument("--members", default="0,1,2,3")
    ap.add_argument("--n-test", type=int, default=1024)
    ap.add_argument("--out-dir", default="artifacts/pnc_theory/bridge")
    a = ap.parse_args()
    run(a.env, a.seed, [float(x) for x in a.scales.split(",")],
        [int(x) for x in a.members.split(",")], a.n_test, a.out_dir)
