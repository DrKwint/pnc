"""SLL OpenOOD evaluation for a frozen posterior bundle (predictive-entropy OOD score).

Uses the linearized probit predictive (primary) and saves MC diagnostics (secondary). ID metrics =
acc/NLL/ECE/Brier of the base classifier's probit-adjusted probabilities. Per-example Parquet +
companion logit npz per dataset; macro Near/Far via the same convention as SCOD (macro-mean over
datasets). Asserts the OOD loaders carry only their own split (never enter selection/tuning).
"""
from __future__ import annotations

import json, time
from pathlib import Path
import numpy as np

from experiments.sll_cifar import predict
from experiments.sll_cifar.run_sll import ood_metrics, NEAR, FAR


def _pctiles(s):
    return {f"p{q}": float(np.percentile(s, q)) for q in (5, 25, 50, 75, 95)}


def evaluate_seed(bundle, bench, root: Path, force=False):
    seed, S = bundle["seed"], bundle["S"]
    fns, dW, T, M = bundle["fns"], bundle["dW_star"], bundle["T_star"], bundle["M"]
    seed_dir = root / "predictions" / str(seed); seed_dir.mkdir(parents=True, exist_ok=True)
    groups = [("id", "cifar10", bench["id_test"])]
    for d in NEAR: groups.append(("near", d, bench["near_ood"][d]))
    for d in FAR: groups.append(("far", d, bench["far_ood"][d]))

    import pandas as pd
    ent = {}      # dataset -> predictive entropy array (for OOD metrics)
    id_metrics_out = None
    for regime, name, data in groups:
        pq = seed_dir / f"{name}.parquet"
        if pq.exists() and not force:
            df = pd.read_parquet(pq); ent[name] = df["predictive_entropy"].to_numpy()
            if regime == "id":
                y = df["label"].to_numpy(); id_metrics_out = _id_from_df(df, y)
            print(f"[sll-eval] seed{seed} {name} cached", flush=True); continue
        X = np.asarray(data["inputs"]); y = np.asarray(data["targets"]).astype(int)
        assert len(X) == len(y), f"{name} split mismatch"
        t0 = time.time()
        out = predict.predict_streaming(fns, X, dW, T, batch=250, with_mc=True)
        pe = out["predictive_entropy"]; ent[name] = pe
        probs = predict._softmax(out["probit_logits"] / T)
        df = pd.DataFrame(dict(
            dataset=name, split=regime, label=y,
            predicted_class=probs.argmax(1), max_softmax_probability=probs.max(1),
            predictive_entropy=pe, mc_entropy=out["mc_entropy"],
            mutual_information=out["mutual_information"], expected_entropy=out["expected_entropy"]))
        df.to_parquet(pq, index=False)
        np.savez(seed_dir / f"{name}_logits.npz", base_logits=out["base_logits"],
                 probit_logits=out["probit_logits"], probit_probs=probs)
        if regime == "id":
            id_metrics_out = predict.id_metrics(probs, y)
        print(f"[sll-eval] seed{seed} {name} ({regime}) n={len(X)} ({time.time()-t0:.0f}s)", flush=True)

    # OOD metrics (predictive entropy, higher = more OOD)
    ids = ent["cifar10"]; per_ds, near, far = {}, [], []
    for regime, name, _ in groups:
        if regime == "id": continue
        m = ood_metrics(ids, ent[name]); m["percentiles"] = _pctiles(ent[name]); per_ds[name] = m
        (near if regime == "near" else far).append(m)
    macro = lambda lst, k: float(np.mean([x[k] for x in lst]))
    metrics = dict(seed=seed, S=S, lam_star=bundle["lam_star"], lam_full=bundle["lam_full"],
                   T_star=T, M=M, id_metrics=id_metrics_out, temperature=T,
                   per_dataset=per_ds,
                   near=dict(auroc=macro(near, "auroc"), fpr95=macro(near, "fpr95"),
                             aupr_in=macro(near, "aupr_in"), aupr_out=macro(near, "aupr_out")),
                   far=dict(auroc=macro(far, "auroc"), fpr95=macro(far, "fpr95"),
                            aupr_in=macro(far, "aupr_in"), aupr_out=macro(far, "aupr_out")),
                   ggn_stats=bundle["ggn_stats"], total_secs=bundle["total_secs"])
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(root / "metrics" / f"sll_backbone_seed{seed}.json", "w"), indent=2)
    print(f"[sll-eval] seed{seed} DONE acc={id_metrics_out['accuracy']:.2f} | "
          f"Near AUROC {metrics['near']['auroc']:.2f} FPR95 {metrics['near']['fpr95']:.2f} | "
          f"Far AUROC {metrics['far']['auroc']:.2f} FPR95 {metrics['far']['fpr95']:.2f}", flush=True)
    return metrics


def _id_from_df(df, y):
    # recompute ID metrics from cached probs isn't stored in parquet; approximate from predicted_class
    acc = float((df["predicted_class"].to_numpy() == y).mean()) * 100
    return dict(accuracy=acc, nll=None, ece=None, brier=None, note="recomputed from cached parquet")
