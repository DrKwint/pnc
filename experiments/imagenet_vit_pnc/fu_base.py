"""Spec §1 — the missing base-model entropy control.

The previous round inferred that P&C's OOD signal is "mostly inherited base confidence"
from the fact that expected member entropy nearly equals predictive entropy. That is an
argument about the ensemble's internals, not a measurement against the base model. Here we
score base entropy directly, at both the raw softmax and the frozen P&C temperature, and
correlate it with every P&C score inside ID and inside pooled OOD.

Neither temperature variant is selected using OOD: both are reported.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import spearmanr

from . import full_cache as fc
from . import fu_common as F
from .memprobe import reset_cuda
from .vit_adapter import ViTPnCAdapter

# higher value = more out-of-distribution
SIGN = {"msp": -1.0, "base_entropy_raw": +1.0, "base_entropy_T": +1.0,
        "M1_true_label": +1.0, "M3_unconditional": +1.0,
        "predictive_entropy": +1.0, "expected_member_entropy": +1.0,
        "mutual_information": +1.0, "logit_variance": +1.0, "prob_variance": +1.0}
ORDER = ["msp", "base_entropy_raw", "base_entropy_T", "M1_true_label", "M3_unconditional",
         "expected_member_entropy", "predictive_entropy", "mutual_information",
         "logit_variance", "prob_variance"]


def run():
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    T = F.temperature()
    print(f"frozen P&C temperature T = {T}")
    t0 = time.perf_counter()

    scores = {}
    for name in F.SETS:
        c = fc.load_cache(F.cache_path(name))
        X = fc.cache_to_gpu(c, ad)
        p_raw = F.base_probs(ad, X, T=None)
        p_tmp = F.base_probs(ad, X, T=T)
        s = {"msp": p_raw.max(-1),
             "base_entropy_raw": F.entropy(p_raw),
             "base_entropy_T": F.entropy(p_tmp)}
        prev = F.load_scores(name)
        for k in ORDER:
            if k not in s:
                s[k] = prev[k]
        scores[name] = s
        np.savez_compressed(F.OUT / "predictions" / f"base_scores_{name}.npz",
                            **{k: v.astype(np.float32) for k, v in s.items()
                               if k in ("msp", "base_entropy_raw", "base_entropy_T")})
        print(f"  {name:<12} {len(p_raw):>7,} scored", flush=True)
        del X, p_raw, p_tmp
        reset_cuda()
    print(f"base scoring took {time.perf_counter()-t0:.0f}s")

    id_s = scores["val50k"]
    results = {"temperature": T, "scores": {}}
    print(f"\n{'score':<28}{'Near AUROC':>12}{'Far AUROC':>11}"
          f"{'Near FPR95':>12}{'Far FPR95':>11}")
    for k in ORDER:
        m = F.ood_metrics(SIGN[k] * id_s[k],
                          {d: SIGN[k] * scores[d][k] for d in F.DS})
        results["scores"][k] = m
        print(f"  {k:<26}{m['near_auroc']*100:>11.2f}{m['far_auroc']*100:>11.2f}"
              f"{m['near_fpr95']*100:>12.2f}{m['far_fpr95']*100:>11.2f}")

    # ---- Spearman among all scores, separately within ID and within pooled OOD ----
    pooled = {k: np.concatenate([scores[d][k] for d in F.DS]) for k in ORDER}
    mats = {}
    for gname, g in (("ID", id_s), ("pooled_OOD", pooled)):
        M = np.zeros((len(ORDER), len(ORDER)))
        for i, a in enumerate(ORDER):
            for j, b in enumerate(ORDER):
                M[i, j] = (1.0 if i == j else
                           spearmanr(SIGN[a] * g[a], SIGN[b] * g[b]).statistic)
        mats[gname] = {a: {b: float(M[i, j]) for j, b in enumerate(ORDER)}
                       for i, a in enumerate(ORDER)}
    results["spearman"] = mats
    results["score_order"] = ORDER

    key = ["base_entropy_raw", "base_entropy_T", "predictive_entropy",
           "expected_member_entropy", "mutual_information"]
    print("\n=== Spearman vs base entropy (raw / temperature-scaled) ===")
    for grp in ("ID", "pooled_OOD"):
        print(f"  [{grp}]")
        for k in key[2:]:
            print(f"    {k:<28} vs raw {mats[grp]['base_entropy_raw'][k]:+.4f}"
                  f"   vs T {mats[grp]['base_entropy_T'][k]:+.4f}")

    F.write_json(F.OUT / "metrics" / "base_entropy_control.json", results)
    print(f"\nwrote {F.OUT/'metrics'/'base_entropy_control.json'}")
    return results


if __name__ == "__main__":
    run()
