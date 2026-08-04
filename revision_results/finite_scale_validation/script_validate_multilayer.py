"""Round H (H.3-H.7): multi-layer re-repair + interaction on real activations.

Uses the float64 MultiStage engine to build the interventional variants and
measure:
  * re-repair Q_{b<-a} = F_ab(va,0) - F_a(va);
  * survival S_{b<-a}(x) = ‖z4dev after corr-b‖/‖z4dev before‖ (ID vs OOD);
  * rotation C_{b<-a}(x) = cos(z4dev_after, z4dev_before);
  * mixed interaction I_ab = F_ab(va,vb)-F_ab(va,0)-F_ab(0,vb)+F_0, ratio η;
  * small-scale asymptotics: ‖Q(t·va)‖∝t and ‖I(t·va,s·vb)‖∝ts (log-log slopes).
Engine sanity: F_ab(0,0)==F_0 (float64) and F_ab(va,vb)≈ensemble predict (float32).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.scripts.pnc_theory import harness as H            # noqa: E402
from experiments.scripts.pnc_theory.multilayer import MultiStage    # noqa: E402
from experiments.scripts.pnc_theory.linalg import rel_err           # noqa: E402

REGIMES = ["id_eval", "ood_near", "ood_mid", "ood_far"]


def _norm(a):
    return np.linalg.norm(a, axis=1)


def run(env, seed, pert_scale, members, n_test, lam, out_dir):
    ds = H.load_dataset(env, seed)
    base = H.get_base_model(env, seed, ds)
    inputs_id = np.asarray(ds["id_train"][0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(inputs_id), min(len(inputs_id), 4096), replace=False)
    X_sub = inputs_id[idx]

    ens = H.build_pnc(base, ds, seed, pert_size=pert_scale, n_members=max(members) + 1,
                      lambda_reg=lam, ridge_toward_orig=True)
    dWa = {i: np.asarray(ens.seq_dWs[0][i], np.float64) for i in members}
    dWb = {i: np.asarray(ens.seq_dWs[1][i], np.float64) for i in members}
    ms = MultiStage(base, X_sub, lam=lam, toward_orig=True)

    res = {"env": env, "seed": seed, "pert_scale": pert_scale, "lam": lam, "sanity": {}}

    # --- sanity ---
    Xs = np.asarray(ds["id_eval"][0], np.float64)[:256]
    zero = np.zeros_like(dWa[members[0]])
    zerob = np.zeros_like(dWb[members[0]])
    # (a) engine correctness: at λ>0 (well-conditioned) F_ab(0,0) must equal F_0 exactly.
    ms1 = MultiStage(base, X_sub, lam=1.0, toward_orig=True)
    Wca1, Wcb1 = ms1.fit(zero, zerob)
    res["sanity"]["Fab00_vs_F0_lam1"] = rel_err(ms1.forward(Xs, zero, Wca1, zerob, Wcb1)["mean"], ms1.F0(Xs))
    # (b) at λ=0 the min-norm correction has a held-out generalization gap (real, not a bug).
    Wca0, Wcb0 = ms.fit(zero, zerob)
    res["sanity"]["Fab00_vs_F0_lam0_gengap"] = rel_err(ms.forward(Xs, zero, Wca0, zerob, Wcb0)["mean"], ms.F0(Xs))
    # (c) full forward vs ensemble (differs at λ=0 by the float32/float64 min-norm ambiguity).
    i0 = members[0]
    Fab_ens = np.asarray(ens.predict(jnp.asarray(Xs))[0], np.float64)[i0]
    res["sanity"]["Fab_vs_ensemble"] = rel_err(ms.F_ab(Xs, dWa[i0], dWb[i0])["mean"], Fab_ens)

    # --- re-repair / interaction at benchmark scale, per regime ---
    per_reg = {}
    for reg in REGIMES:
        if reg not in ds:
            continue
        Xe = np.asarray(ds[reg][0], np.float64)[:n_test]
        # zero-perturbation baselines (cancel the λ=0 held-out generalization gap)
        Fab00 = ms.forward(Xe, zero, Wca0, zerob, Wcb0)          # both stages, zero pert
        Fa0 = ms.F_a(Xe, zero)                                    # stage a only, zero pert
        S_list, C_list, eta_list, Qrel_list = [], [], [], []
        for i in members:
            Fa = ms.F_a(Xe, dWa[i])                        # stage a only
            Fab_a0 = _fit_fwd(ms, Xe, dWa[i], zerob)       # F_ab(va,0)
            Fab_0b = _fit_fwd(ms, Xe, zero, dWb[i])        # F_ab(0,vb)
            Fab = _fit_fwd(ms, Xe, dWa[i], dWb[i])         # F_ab(va,vb)
            # survival & rotation of the va-induced l4-preactivation residual
            before = Fa["z4_dev"] - Fa0["z4_dev"]          # stage-a residual, original l3/l4
            after = Fab_a0["z4_dev"] - Fab00["z4_dev"]     # after stage-b correction
            nb = _norm(before); na = _norm(after)
            m = nb > 1e-9
            S_list.append(na[m] / (nb[m] + 1e-30))
            C_list.append(np.sum(before * after, axis=1)[m] / (nb[m] * na[m] + 1e-30))
            # re-repair magnitude (final output): Q = F_ab(va,0) - F_a(va), baseline-subtracted
            Q = (Fab_a0["mean"] - Fa["mean"]) - (Fab00["mean"] - Fa0["mean"])
            Qrel_list.append(_norm(Q) / (_norm(Fa["mean"] - Fa0["mean"]) + 1e-9))
            # mixed interaction with the correct F_ab(0,0) baseline
            I = Fab["mean"] - Fab_a0["mean"] - Fab_0b["mean"] + Fab00["mean"]
            denom = _norm(Fab_a0["mean"] - Fab00["mean"]) + _norm(Fab_0b["mean"] - Fab00["mean"])
            eta_list.append(_norm(I) / (denom + 1e-9))
        per_reg[reg] = {
            "survival_med": float(np.median(np.concatenate(S_list))),
            "survival_p90": float(np.quantile(np.concatenate(S_list), 0.90)),
            "rotation_med": float(np.median(np.concatenate(C_list))),
            "rerepair_rel_med": float(np.median(np.concatenate(Qrel_list))),
            "interaction_eta_med": float(np.median(np.concatenate(eta_list))),
            "interaction_eta_p90": float(np.quantile(np.concatenate(eta_list), 0.90)),
        }
    res["per_regime"] = per_reg

    # --- small-scale asymptotics on member 0 (ID points), well-conditioned λ=1 so
    #     F_ab(0,0)=F_0 exactly and the limiting scaling is clean (H.6, math-only). ---
    msa = ms1
    Wca0a, Wcb0a = msa.fit(zero, zerob)
    Xa = np.asarray(ds["id_eval"][0], np.float64)[:256]
    ua = dWa[i0] / (np.linalg.norm(dWa[i0]) + 1e-30)
    ub = dWb[i0] / (np.linalg.norm(dWb[i0]) + 1e-30)
    ts = np.logspace(-4, -1, 10)
    F0a = msa.forward(Xa, zero, Wca0a, zerob, Wcb0a)["mean"]      # = F_ab(0,0) = F_0
    Fa0a = msa.F_a(Xa, zero)["mean"]
    Qn, In = [], []
    for t in ts:
        Fa = msa.F_a(Xa, t * ua)["mean"]
        Fab_a0 = _fit_fwd(msa, Xa, t * ua, zerob)["mean"]
        Fab_0b = _fit_fwd(msa, Xa, zero, t * ub)["mean"]
        Fab = _fit_fwd(msa, Xa, t * ua, t * ub)["mean"]
        Qn.append(float(np.mean(_norm((Fab_a0 - Fa) - (F0a - Fa0a)))))
        In.append(float(np.mean(_norm(Fab - Fab_a0 - Fab_0b + F0a))))
    Qn = np.array(Qn); In = np.array(In)
    res["asymptotic"] = {
        "Q_slope": float(np.polyfit(np.log(ts), np.log(Qn + 1e-30), 1)[0]),   # expect ~1
        "I_slope": float(np.polyfit(np.log(ts), np.log(In + 1e-30), 1)[0]),   # expect ~2 (t*s, s=t)
    }

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"P{pert_scale:g}_lam{lam:g}"
    fp = out_dir / f"multilayer_{env}_seed{seed}_{tag}.json"
    fp.write_text(json.dumps(res, indent=2, default=float))

    print(f"\n{'='*80}\nMulti-layer re-repair & interaction — {env} seed{seed} (P={pert_scale}, λ={lam})\n{'='*80}")
    print(f"sanity: F_ab(0,0)=F_0 @λ=1 {res['sanity']['Fab00_vs_F0_lam1']:.2e} (engine correct) | "
          f"@λ=0 gen-gap {res['sanity']['Fab00_vs_F0_lam0_gengap']:.2e} | "
          f"F_ab vs ensemble(f32) {res['sanity']['Fab_vs_ensemble']:.2e}")
    print(f"\n{'regime':9s} {'survival S':>11s} {'rotation C':>11s} {'re-repair Q':>12s} "
          f"{'interaction η':>14s}")
    for reg in REGIMES:
        if reg not in per_reg:
            continue
        d = per_reg[reg]
        print(f"{reg:9s} {d['survival_med']:>11.3f} {d['rotation_med']:>11.3f} "
              f"{d['rerepair_rel_med']:>12.3f} {d['interaction_eta_med']:>14.3f}")
    sid = per_reg["id_eval"]["survival_med"]
    sfar = per_reg.get("ood_far", {}).get("survival_med", float("nan"))
    print(f"\nasymmetric re-repair test  S_ID={sid:.3f}  vs  S_Far={sfar:.3f}  → "
          f"{'S_ID<S_OOD (desired: erases upstream more on ID)' if sid < sfar else 'NOT S_ID<S_OOD'}")
    print(f"small-scale asymptotics: ‖Q(t)‖ slope={res['asymptotic']['Q_slope']:.2f} (expect~1)  "
          f"‖I(t,t)‖ slope={res['asymptotic']['I_slope']:.2f} (expect~2)")
    print(f"\nwrote {fp}")
    return res


def _fit_fwd(ms, Xe, dWa, dWb):
    Wca, Wcb = ms.fit(dWa, dWb, stage_b=True)
    return ms.forward(Xe, dWa, Wca, dWb, Wcb, stage_b=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Ant-v5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pert-scale", type=float, default=10.0)
    ap.add_argument("--members", default="0,1,2,3")
    ap.add_argument("--n-test", type=int, default=1024)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--out-dir", default="artifacts/pnc_theory/multilayer")
    a = ap.parse_args()
    run(a.env, a.seed, a.pert_scale, [int(x) for x in a.members.split(",")],
        a.n_test, a.lam, a.out_dir)
