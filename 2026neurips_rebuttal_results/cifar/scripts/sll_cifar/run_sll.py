"""SLL-Backbone workhorse: selection -> dense GGN -> eigensystem -> ID-only prior/temperature
selection -> posterior -> OpenOOD evaluation. Modes: smoke | pilot | run.

  smoke  : seed 0, S=128, 64 calib / 128 val, M=5 -- exercises the whole path + gates.
  pilot  : seed 0, S in {512,1024,2048}, pick S by ID-val NLL (tie<=0.002 -> smaller).
  run    : seeds 0,1,2 at the frozen S, full OpenOOD eval, tables.

All selection / GGN / prior / temperature use ID data only; OOD is not loaded until S, lambda,
prediction rule, and T are frozen (asserted).
"""
from __future__ import annotations

import argparse, json, os, time, gc
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false"
import numpy as np
REPO = Path(__file__).resolve().parents[2]
import sys; os.chdir(REPO); sys.path.insert(0, str(REPO))
import yaml, jax.numpy as jnp
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

from experiments.scod_cifar import protocol
from experiments.sll_cifar import selection, posterior, predict
from experiments.sll_cifar.subnetwork import make_subnetwork_fns, build_spec_from_flat_indices

ROOT = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "sll"
DlockPATH = ROOT / ".gpu.lock"


class GpuLock:
    """Simple filesystem GPU lock: wait rather than compete with an active CIFAR job."""
    def __init__(self, path=DlockPATH, poll=15):
        self.path = Path(path); self.poll = poll; self.fd = None
    def __enter__(self):
        import fcntl, time as _t
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.path, "w")
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                print("[lock] GPU busy, waiting...", flush=True); _t.sleep(self.poll)
        self.fd.write(f"pid={os.getpid()}\n"); self.fd.flush(); return self
    def __exit__(self, *a):
        import fcntl
        try: fcntl.flock(self.fd, fcntl.LOCK_UN); self.fd.close()
        except Exception: pass


NEAR = ["cifar100", "tiny_imagenet"]; FAR = ["mnist", "svhn", "textures", "places365"]


def ood_metrics(id_s, ood_s):
    labels = np.concatenate([np.zeros(len(id_s)), np.ones(len(ood_s))]); s = np.concatenate([id_s, ood_s])
    auroc = float(roc_auc_score(labels, s)) * 100
    fpr, tpr, _ = roc_curve(labels, s); meets = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[meets[0]]) * 100 if len(meets) else 100.0
    aupr_out = float(average_precision_score(labels, s)) * 100
    aupr_in = float(average_precision_score(1 - labels, -s)) * 100
    return dict(auroc=auroc, fpr95=fpr95, aupr_in=aupr_in, aupr_out=aupr_out)


def build_seed_posterior(seed, S, x_calib, x_val, y_val, M, sample_seed, include_fc=False, verbose=True):
    """Selection + dense GGN + eigensystem + ID-only prior/temperature selection. Returns a bundle."""
    t0 = time.time()
    idx_S, layout = selection.run_selection(seed, x_calib, S, ROOT / "selection", include_fc)
    fns = make_subnetwork_fns(seed, idx_S, temperature=1.0)
    G, ggn_stats = posterior.build_dense_ggn(fns, x_calib, S, log_every=256 if verbose else 0)
    d, U = posterior.eigensystem(G)
    grid, m, floor = posterior.prior_grid(d)
    tj = time.time(); J_val, Z_val = predict.selected_jacobians(fns, x_val); jac_secs = time.time() - tj
    curve = []
    for lam in grid:
        dW, _ = posterior.sample_perturbations(d, U, float(lam), M, sample_seed)
        pl, v, _ = predict.probit_logits_from(J_val, Z_val, dW)
        T = predict.fit_temperature(pl, y_val)
        probs = predict._softmax(pl / T)
        met = predict.id_metrics(probs, y_val)
        curve.append(dict(lam=float(lam), T=T, val_nll=met["nll"], val_acc=met["accuracy"],
                          mean_logit_var=float(v.mean())))
    best = min(curve, key=lambda c: c["val_nll"])
    lam_star, T_star = best["lam"], best["T"]
    dW_star, eps = posterior.sample_perturbations(d, U, lam_star, M, sample_seed)
    D = int(fns["flat_w"].shape[0])
    bundle = dict(seed=seed, S=S, include_fc=include_fc, idx_S=idx_S, d=d, U=U, G=G, ggn_stats=ggn_stats,
                  grid=grid.tolist(), median_eig=m, floor=floor, curve=curve, lam_star=lam_star,
                  T_star=T_star, dW_star=dW_star, eps=eps, M=M, sample_seed=sample_seed,
                  lam_full=lam_star * D / S, val_nll_star=best["val_nll"], jac_secs=round(jac_secs, 1),
                  total_secs=round(time.time() - t0, 1), fns=fns)
    if verbose:
        print(f"[posterior] seed{seed} S={S}: lam*={lam_star:.3g} (=lam_full {bundle['lam_full']:.3g}) "
              f"T*={T_star:.3f} valNLL={best['val_nll']:.4f} valAcc={best['val_acc']:.2f} "
              f"eig[min,max]=[{ggn_stats['eig_min']:.2e},{ggn_stats['eig_max']:.2e}] ({bundle['total_secs']:.0f}s)",
              flush=True)
    return bundle


def save_posterior(bundle, tag="backbone"):
    seed, S = bundle["seed"], bundle["S"]
    (ROOT / "posterior").mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / "posterior" / f"eigensystem_seed{seed}_S{S}_{tag}.npz",
             d=bundle["d"], U=bundle["U"], idx_S=bundle["idx_S"])
    np.savez(ROOT / "posterior" / f"posterior_samples_seed{seed}_S{S}_{tag}.npz",
             dW=bundle["dW_star"], eps=bundle["eps"], lam=bundle["lam_star"], sample_seed=bundle["sample_seed"])
    json.dump(dict(seed=seed, S=S, tag=tag, lam_star=bundle["lam_star"], lam_full=bundle["lam_full"],
                   T_star=bundle["T_star"], median_eig=bundle["median_eig"], floor=bundle["floor"],
                   grid=bundle["grid"], curve=bundle["curve"], val_nll_star=bundle["val_nll_star"],
                   ggn_stats=bundle["ggn_stats"], jac_secs=bundle["jac_secs"], total_secs=bundle["total_secs"]),
              open(ROOT / "posterior" / f"prior_selection_seed{seed}_S{S}_{tag}.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["smoke", "pilot", "run"])
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr, x_va, y_va = protocol.get_splits()
    with GpuLock():
        if args.mode == "smoke":
            xcal, _, _ = protocol.get_calibration_subset(0)
            b = build_seed_posterior(0, 128, np.asarray(xcal)[:64], np.asarray(x_va)[:128],
                                     np.asarray(y_va)[:128], M=5, sample_seed=100000)
            save_posterior(b, tag="smoke")
            print("[smoke] OK", flush=True)
        elif args.mode == "pilot":
            xcal, ycal, _ = protocol.get_calibration_subset(0)
            res = {}
            for S in [512, 1024, 2048]:
                b = build_seed_posterior(0, S, np.asarray(xcal), np.asarray(x_va), np.asarray(y_va),
                                         M=args.M, sample_seed=100000)
                save_posterior(b)
                res[S] = dict(val_nll=b["val_nll_star"], lam=b["lam_star"], T=b["T_star"],
                              total_secs=b["total_secs"], eig_max=b["ggn_stats"]["eig_max"])
                del b; gc.collect()
            # choose S by ID-val NLL; tie<=0.002 -> smaller
            order = sorted(res.keys())
            best_S = order[0]
            for S in order[1:]:
                if res[S]["val_nll"] < res[best_S]["val_nll"] - 0.002:
                    best_S = S
            json.dump(dict(pilot=res, chosen_S=best_S), open(ROOT / "configs" / "pilot_result.json", "w"), indent=2)
            print(f"[pilot] chosen S={best_S}  {json.dumps({s: round(res[s]['val_nll'],4) for s in res})}", flush=True)
        elif args.mode == "run":
            # requires OOD eval -- imported here so pilot/smoke never load OOD
            from experiments.sll_cifar.run_eval import evaluate_seed
            bench = protocol.load_benchmark()
            for seed in args.seeds:
                xcal, ycal, _ = protocol.get_calibration_subset(seed)
                b = build_seed_posterior(seed, args.S, np.asarray(xcal), np.asarray(x_va),
                                         np.asarray(y_va), M=args.M, sample_seed=100000 + seed)
                save_posterior(b)
                evaluate_seed(b, bench, ROOT)
                del b; gc.collect()
            print("[run] ALL SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
