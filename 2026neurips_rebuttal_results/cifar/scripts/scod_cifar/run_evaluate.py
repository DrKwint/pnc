"""Stage 2 -- evaluate SCOD on ID + Near/Far OpenOOD, per seed.

For each model seed: load the tempered (primary) and untempered (diagnostic) sketches, compute
per-example SCOD score features (total energy + per-eigendirection captured energy) for CIFAR-10
test and every OpenOOD dataset, derive scores for k in {5,10,20} and the Meps grid (all post-hoc
from the features -- no Jacobian recompute), plus base-classifier columns (pred, MSP, predictive
entropy, energy). Save per-example Parquet (Section 15) and an ID-vs-OOD metrics JSON with
AUROC/FPR95/AUPR-IN/AUPR-OUT and macro Near/Far aggregates (macro-mean over datasets, matching the
submitted evaluator). Resumable per (seed, dataset). No OOD data touches score selection.

Usage: python -m experiments.scod_cifar.run_evaluate --config configs/scod_cifar.yaml [--force]
"""
from __future__ import annotations

import argparse, json, os, time, gc
from pathlib import Path

import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false"
REPO = Path(__file__).resolve().parents[2]
import sys
os.chdir(REPO); sys.path.insert(0, str(REPO))
import jax.numpy as jnp
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from experiments.scod_cifar import protocol
from experiments.scod_cifar.parameter_layout import make_ztilde_fn, make_logits_fn
from experiments.scod_cifar.sketch import load_sketch
from experiments.scod_cifar.scorer import compute_score_features, score_from_features


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ood_metrics(id_scores, ood_scores):
    """higher score = more OOD. ID=neg(0), OOD=pos(1). Returns auroc/fpr95/aupr_out/aupr_in."""
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = float(roc_auc_score(labels, scores))
    aupr_out = float(average_precision_score(labels, scores))              # OOD positive
    aupr_in = float(average_precision_score(1 - labels, -scores))          # ID positive
    fpr, tpr, _ = roc_curve(labels, scores)
    meets = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[meets[0]]) if len(meets) else 1.0
    return dict(auroc=auroc * 100, fpr95=fpr95 * 100, aupr_out=aupr_out * 100, aupr_in=aupr_in * 100)


def base_columns(logits, T):
    """base-classifier per-example columns from temperature-scaled logits."""
    z = np.asarray(logits) / T
    z = z - z.max(-1, keepdims=True); e = np.exp(z); p = e / e.sum(-1, keepdims=True)
    msp = p.max(1); pred = p.argmax(1)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    energy = -T * (np.log(np.exp(np.asarray(logits) / T).sum(1) + 1e-12))  # -T logsumexp(z/T)
    return pred, msp, ent, energy


def percentiles(s):
    return {f"p{q}": float(np.percentile(s, q)) for q in (5, 25, 50, 75, 95)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = REPO / cfg["root"]
    (root / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    k_list = cfg["sensitivity"]["num_eigs"]; meps_list = cfg["sensitivity"]["Meps"]
    k_prim = int(cfg["reported_num_eigs"]); meps_prim = float(cfg["Meps"])
    base = int(cfg["sketch_seed_base_primary"])

    print("[eval] loading OpenOOD benchmark (ID+Near+Far) ...", flush=True)
    bench = protocol.load_benchmark()
    bench.pop("id_train", None); gc.collect()  # not needed at eval; free ~600MB
    groups = [("id", "cifar10", bench["id_test"])]
    for d in cfg["near_ood"]:
        groups.append(("near", d, bench["near_ood"][d]))
    for d in cfg["far_ood"]:
        groups.append(("far", d, bench["far_ood"][d]))

    for seed in cfg["model_seeds"]:
        sketch_seed = base + seed
        st = load_sketch(root / "sketches" / f"scod1024_seed{seed}_sketchseed{sketch_seed}_tempered.npz")
        stu = load_sketch(root / "sketches" / f"scod1024_seed{seed}_sketchseed{sketch_seed}_untempered.npz")
        # reuse saved temperature (no refit / no duplicate CIFAR load)
        T = float(json.load(open(root / "metrics" / f"seed{seed}_build_meta.json"))["temperature"])
        S = protocol.load_model_only(seed)
        ztilde_t = make_ztilde_fn(S["graphdef"], S["rest"], S["unravel"], T)
        ztilde_u = make_ztilde_fn(S["graphdef"], S["rest"], S["unravel"], 1.0)
        logits_fn = make_logits_fn(S["graphdef"], S["rest"], S["unravel"], temperature=1.0)
        U_t = jnp.asarray(st["basis"]); U_u = jnp.asarray(stu["basis"])
        ev_t = st["eigvals"]; ev_u = stu["eigvals"]

        per_seed = {}   # dataset -> dict(scores per variant, base cols)
        seed_dir = root / "predictions" / f"seed_{seed}" / f"sketch_{sketch_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for regime, name, data in groups:
            out_pq = seed_dir / f"{name}.parquet"
            if out_pq.exists() and not args.force:
                print(f"[eval] seed{seed} {name} parquet exists, loading cached scores", flush=True)
                import pyarrow.parquet as pq
                per_seed[name] = pq.read_table(out_pq).to_pandas()
                continue
            X = np.asarray(data["inputs"]); y = np.asarray(data["targets"]).astype(int)
            t0 = time.time()
            tot_t, cap_t = compute_score_features(ztilde_t, S["flat_w"], X, U_t, tag=f"{name}/T")
            tot_u, cap_u = compute_score_features(ztilde_u, S["flat_w"], X, U_u, tag=f"{name}/U")
            # base columns
            L = np.concatenate([np.asarray(logits_fn(S["flat_w"], jnp.asarray(X[i:i+250])))
                                for i in range(0, len(X), 250)], 0)
            pred, msp, ent, energy = base_columns(L, T)
            import pandas as pd
            df = pd.DataFrame(dict(
                example_index=np.arange(len(X)), dataset=name, regime=regime, label=y,
                predicted_class=pred, correct=(pred == y).astype(int) if regime == "id" else -1,
                temperature=T, max_softmax_probability=msp, predictive_entropy=ent, energy_score=energy,
                scod_score_k5=score_from_features(tot_t, cap_t, ev_t, 5, meps_prim),
                scod_score_k10=score_from_features(tot_t, cap_t, ev_t, 10, meps_prim),
                scod_score_k20=score_from_features(tot_t, cap_t, ev_t, 20, meps_prim),
                scod_score_untempered=score_from_features(tot_u, cap_u, ev_u, k_prim, meps_prim),
            ))
            # keep raw features for the sensitivity grid (Meps/k) without recompute
            df.attrs = {}
            np.savez(seed_dir / f"{name}_features.npz", tot_t=tot_t, cap_t=cap_t, tot_u=tot_u,
                     cap_u=cap_u, ev_t=ev_t, ev_u=ev_u)
            df.to_parquet(out_pq, index=False)
            per_seed[name] = df
            print(f"[eval] seed{seed} {name} ({regime}) n={len(X)} ({time.time()-t0:.0f}s)", flush=True)

        # ---- metrics: per-dataset + macro, for every (k, Meps) and untempered ----
        id_feat = np.load(seed_dir / "cifar10_features.npz")
        metrics = dict(seed=seed, sketch_seed=sketch_seed, temperature=T,
                       base_id=protocol.base_id_metrics(seed, S["flat_w"], S["graphdef"], S["rest"],
                                                        S["unravel"], T, benchmark=bench),
                       variants={})
        def id_score(k, meps, tempered=True):
            f = id_feat
            return score_from_features(f["tot_t" if tempered else "tot_u"],
                                       f["cap_t" if tempered else "cap_u"],
                                       f["ev_t" if tempered else "ev_u"], k, meps)
        variant_defs = [(f"tempered_k{k}_Meps{int(m)}", k, m, True) for k in k_list for m in meps_list]
        variant_defs += [(f"untempered_k{k_prim}_Meps{int(meps_prim)}", k_prim, meps_prim, False)]
        for vname, k, meps, tempered in variant_defs:
            ids = id_score(k, meps, tempered)
            per_ds, near, far = {}, [], []
            for regime, name, _ in groups:
                if regime == "id":
                    continue
                f = np.load(seed_dir / f"{name}_features.npz")
                oods = score_from_features(f["tot_t" if tempered else "tot_u"],
                                           f["cap_t" if tempered else "cap_u"],
                                           f["ev_t" if tempered else "ev_u"], k, meps)
                m = ood_metrics(ids, oods); m["percentiles_ood"] = percentiles(oods)
                per_ds[name] = m
                (near if regime == "near" else far).append(m)
            def macro(lst, key): return float(np.mean([x[key] for x in lst]))
            metrics["variants"][vname] = dict(
                per_dataset=per_ds,
                near=dict(auroc=macro(near, "auroc"), fpr95=macro(near, "fpr95"),
                          aupr_in=macro(near, "aupr_in"), aupr_out=macro(near, "aupr_out")),
                far=dict(auroc=macro(far, "auroc"), fpr95=macro(far, "fpr95"),
                         aupr_in=macro(far, "aupr_in"), aupr_out=macro(far, "aupr_out")),
                id_score_stats=percentiles(ids))
        json.dump(metrics, open(root / "metrics" / f"seed{seed}_metrics.json", "w"), indent=2)
        pk = f"tempered_k{k_prim}_Meps{int(meps_prim)}"
        pv = metrics["variants"][pk]
        print(f"[eval] seed{seed} PRIMARY {pk}: Near AUROC {pv['near']['auroc']:.2f} FPR95 "
              f"{pv['near']['fpr95']:.2f} | Far AUROC {pv['far']['auroc']:.2f} FPR95 {pv['far']['fpr95']:.2f}",
              flush=True)
    print("[eval] ALL SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
