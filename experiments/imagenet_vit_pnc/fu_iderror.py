"""Spec §8 — ImageNet misclassification detection.

OOD detection and predictive-uncertainty quality are different tasks. Here the positive
class is "the frozen base model's top-1 prediction on the untouched 50k validation set is
wrong", and every score is evaluated as an error detector. Nothing about the base model's
predictions changes; only the ranking score varies.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import full_cache as fc
from . import fu_common as F

# name -> (source, key, sign) with sign chosen so that higher = more likely to be an error
SPECS = [
    ("Mahalanobis",                    "geom", "M1_true_label",           +1),
    ("Mahalanobis (unconditional)",    "geom", "M3_unconditional",        +1),
    ("MSP",                            "base", "msp",                     -1),
    ("Base entropy (raw)",             "base", "base_entropy_raw",        +1),
    ("Base entropy (T=0.7)",           "base", "base_entropy_T",          +1),
    ("P&C predictive entropy",         "geom", "predictive_entropy",      +1),
    ("P&C expected member entropy",    "geom", "expected_member_entropy", +1),
    ("P&C mutual information",         "geom", "mutual_information",      +1),
    ("P&C logit variance",             "geom", "logit_variance",          +1),
    ("P&C mean pairwise KL",           "dis",  "mean_pairwise_kl",        +1),
    ("P&C member agreement",           "dis",  "member_agreement",        -1),
    ("LLLA-Kron entropy",              "llla", "id_entropy",              +1),
    ("LLLA-Kron+Temp entropy",         "lltp", "id_entropy",              +1),
    ("SCOD-linear",                    "scod_linear", "val50k",           +1),
    ("SCOD-FFN",                       "scod_ffn", "val50k",              +1),
    ("SCOD-last-block",                "scod_last_block", "val50k",       +1),
] + [(f"P&C K={K} predictive entropy", f"k{K}", "val50k_predictive_entropy", +1)
     for K in (5, 20, 40, 80)]


def _sources():
    src = {"geom": F.load_scores("val50k")}
    b = np.load(F.OUT / "predictions" / "base_scores_val50k.npz")
    src["base"] = {k: b[k].astype(np.float64) for k in b.files}
    for tag, path in (("dis", F.OUT / "predictions" / "disagreement_val50k.npz"),
                      ("llla", F.PREV / "predictions" / "llla-kron_scores.npz"),
                      ("lltp", F.OUT / "predictions" / "llla-kron-temp_scores.npz"),
                      ("scod_linear", F.OUT / "predictions" / "scod_linear_scores.npz"),
                      ("scod_ffn", F.OUT / "predictions" / "scod_ffn_scores.npz"),
                      ("scod_last_block",
                       F.OUT / "predictions" / "scod_last_block_scores.npz"),
                      *[(f"k{K}", F.OUT / "predictions" / f"ksweep_K{K}_seed0.npz")
                        for K in (5, 20, 40, 80)]):
        if path.exists():
            z = np.load(path)
            src[tag] = {k: z[k].astype(np.float64) for k in z.files}
    return src


def run():
    val = fc.load_cache(F.cache_path("val50k"))
    y = val["labels"]
    pred = np.load(F.SRC / "raw" / "base_val_logits.npy").argmax(-1)
    err = (pred != y).astype(int)
    print(f"base top-1 {100*(1-err.mean()):.3f}%  ->  {err.sum():,} errors "
          f"of {len(err):,}")

    src = _sources()
    rows = []
    print(f"\n{'score':<32}{'AUROC':>9}{'AUPR':>9}{'mean(correct)':>16}"
          f"{'mean(error)':>14}")
    for name, tag, key, sign in SPECS:
        if tag not in src or key not in src[tag]:
            rows.append({"score": name, "auroc": None, "aupr": None,
                         "status": "NOT_RUN"})
            print(f"  {name:<30}{'NA':>9}{'NA':>9}{'—':>16}{'—':>14}")
            continue
        v = sign * src[tag][key]
        raw = src[tag][key]
        r = {"score": name, "auroc": float(roc_auc_score(err, v)),
             "aupr": float(average_precision_score(err, v)),
             "mean_correct": float(raw[err == 0].mean()),
             "mean_incorrect": float(raw[err == 1].mean()),
             "sign": sign, "status": "OK"}
        rows.append(r)
        print(f"  {name:<30}{r['auroc']*100:>9.2f}{r['aupr']*100:>9.2f}"
              f"{r['mean_correct']:>16.4f}{r['mean_incorrect']:>14.4f}")

    out = {"n": int(len(err)), "n_errors": int(err.sum()),
           "base_top1": float(1 - err.mean()),
           "positive_class": "base model top-1 incorrect",
           "rows": rows}
    F.write_json(F.OUT / "metrics" / "id_error_detection.json", out)
    print(f"\nwrote {F.OUT/'metrics'/'id_error_detection.json'}")
    return out


if __name__ == "__main__":
    run()
