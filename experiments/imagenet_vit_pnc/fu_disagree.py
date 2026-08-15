"""Spec §6-§7 — where Mahalanobis and P&C fundamentally disagree.

Two extra member-level quantities are needed that the previous round did not save:

  member_agreement   fraction of the M members whose top-1 equals the base prediction
  mean_pairwise_kl   (1/M(M-1)) sum_{i != j} KL(p_i || p_j)

The pairwise KL is computed in one pass rather than over M(M-1)=380 explicit pairs, using

    sum_{i!=j} KL(p_i||p_j) = -M*sum_i H_i - sum_c (sum_j log p_jc) * (sum_i p_ic)

so only two (n, 1000) accumulators are needed. This is what made the quantity affordable
here after it was dropped from the previous round.

Everything else reuses the saved per-example scores. Sets are defined by ID-referenced
empirical percentiles so that Mahalanobis distances and entropies are comparable.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import full_cache as fc
from . import fu_common as F
from .memprobe import reset_cuda
from .vit_adapter import ViTPnCAdapter

BINS = [(0, 50), (50, 75), (75, 90), (90, 95), (95, 100)]
REPORT = ["M1_true_label", "predictive_entropy", "expected_member_entropy",
          "mutual_information", "logit_variance", "hidden_perturbation_change",
          "post_correction_residual", "ridge_leverage"]


@torch.inference_mode()
def member_disagreement(ad, X: torch.Tensor, members: list, T: float,
                        chunk: int = 4096) -> dict:
    """member_agreement and mean_pairwise_kl for one cached image set."""
    n, M = X.shape[0], len(members)
    sum_H = torch.zeros(n, dtype=torch.float64)
    sum_p = torch.zeros(n, 1000, dtype=torch.float64)
    sum_logp = torch.zeros(n, 1000, dtype=torch.float64)
    agree = torch.zeros(n, dtype=torch.float64)
    base_pred = torch.empty(n, dtype=torch.long)
    for s in range(0, n, chunk):
        base_pred[s:s + chunk] = ad.tail(X[s:s + chunk][:, None, :],
                                         cls_only=True).argmax(-1).cpu()
    for mem in members:
        for s in range(0, n, chunk):
            lg = F.member_logits(ad, X[s:s + chunk], mem, chunk=chunk).double()
            logp = torch.log_softmax(lg / T, -1)
            p = logp.exp()
            sum_H[s:s + chunk] += -(p * logp).sum(-1)
            sum_p[s:s + chunk] += p
            sum_logp[s:s + chunk] += logp
            agree[s:s + chunk] += (lg.argmax(-1) == base_pred[s:s + chunk]).double()
            del lg, logp, p
    pbar = sum_p / M
    cross = (sum_logp * pbar).sum(-1)                 # sum_c L_c * pbar_c
    mean_kl = (-sum_H - M * cross) / (M - 1)
    return {"member_agreement": (agree / M).numpy(),
            "mean_pairwise_kl": mean_kl.numpy(),
            "base_pred": base_pred.numpy()}


def _summ(v: np.ndarray) -> dict:
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def run():
    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    T = F.temperature()
    members, cfg = F.load_primary_members(ad)
    print(f"P&C primary: r={cfg['r_target']} lam={cfg['lambda']} M={len(members)}")

    t0 = time.perf_counter()
    S, extra = {}, {}
    for name in F.SETS:
        c = fc.load_cache(F.cache_path(name))
        X = fc.cache_to_gpu(c, ad)
        d = member_disagreement(ad, X, members, T)
        b = np.load(F.OUT / "predictions" / f"base_scores_{name}.npz")
        s = F.load_scores(name)
        s.update({k: b[k].astype(np.float64) for k in b.files})
        s["member_agreement"] = d["member_agreement"]
        s["mean_pairwise_kl"] = d["mean_pairwise_kl"]
        S[name] = s
        extra[name] = {"base_pred": d["base_pred"],
                       "labels": c["labels"] if name == "val50k" else None}
        np.savez_compressed(F.OUT / "predictions" / f"disagreement_{name}.npz",
                            member_agreement=d["member_agreement"].astype(np.float32),
                            mean_pairwise_kl=d["mean_pairwise_kl"].astype(np.float32),
                            base_pred=d["base_pred"].astype(np.int32))
        print(f"  {name:<12} {len(s['M1_true_label']):>7,} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        del X
        reset_cuda()

    # ---------- ID-referenced percentiles ----------
    ref = S["val50k"]
    def pct(name, key):
        return np.searchsorted(np.sort(ref[key]), S[name][key],
                               side="right") / len(ref[key]) * 100.0

    P = {n: {"M": pct(n, "M1_true_label"), "P": pct(n, "predictive_entropy"),
             "MI": pct(n, "mutual_information")} for n in F.SETS}

    results = {"config": cfg, "temperature": T, "thresholds": {}, "sets": {},
               "conditional": {}, "report_fields": REPORT}

    for hi, lo, tag in ((95, 50, "strict"), (90, 60, "relaxed")):
        geo = {n: (P[n]["M"] >= hi) & (P[n]["P"] <= lo) for n in F.SETS}
        pnc = {n: (P[n]["P"] >= hi) & (P[n]["M"] <= lo) for n in F.SETS}
        results["thresholds"][tag] = {"high_pct": hi, "low_pct": lo,
                                      "n_geometry_only": int(sum(m.sum() for m in geo.values())),
                                      "n_pnc_only": int(sum(m.sum() for m in pnc.values()))}
        for label, masks in (("geometry_only", geo), ("pnc_only", pnc)):
            comp = {n: int(masks[n].sum()) for n in F.SETS}
            allm = np.concatenate([masks[n] for n in F.SETS])
            fields = {}
            for k in REPORT + ["msp", "base_entropy_T", "member_agreement",
                               "mean_pairwise_kl"]:
                v = np.concatenate([S[n][k][masks[n]] for n in F.SETS])
                fields[k] = _summ(v) if len(v) else None
            # background: all examples, for contrast
            bg = {}
            for k in REPORT + ["msp", "base_entropy_T", "member_agreement",
                               "mean_pairwise_kl"]:
                bg[k] = _summ(np.concatenate([S[n][k] for n in F.SETS]))
            idm = masks["val50k"]
            id_extra = None
            if idm.sum():
                lab = extra["val50k"]["labels"]
                bp = extra["val50k"]["base_pred"]
                id_extra = {"n_id": int(idm.sum()),
                            "base_correct_rate": float((bp[idm] == lab[idm]).mean()),
                            "overall_base_correct_rate": float((bp == lab).mean()),
                            "n_distinct_pred_classes": int(len(np.unique(bp[idm])))}
            results["sets"][f"{tag}/{label}"] = {
                "count": int(allm.sum()), "composition": comp,
                "fields": fields, "background": bg, "id_detail": id_extra}

    # ---------- §7 conditional performance ----------
    for by, other in (("M", "P"), ("P", "M")):
        rows = []
        for lo, hi in BINS:
            m = {n: (P[n][by] >= lo) & (P[n][by] < (hi + 1e-9 if hi == 100 else hi))
                 for n in F.SETS}
            n_ood = int(sum(m[d].sum() for d in F.DS))
            if n_ood < 50:
                rows.append({"bin": f"{lo}-{hi}", "n_ood": n_ood, "auroc": None})
                continue
            # score the OTHER method on OOD examples in this bin vs the full ID reference
            key = "predictive_entropy" if other == "P" else "M1_true_label"
            om = F.ood_metrics(ref[key], {d: S[d][key][m[d]] for d in F.DS
                                          if m[d].sum() >= 20})
            rows.append({"bin": f"{lo}-{hi}", "n_ood": n_ood,
                         "n_id": int(m["val50k"].sum()),
                         "near_auroc": om.get("near_auroc"),
                         "far_auroc": om.get("far_auroc"),
                         "per_dataset_n": {d: int(m[d].sum()) for d in F.DS}})
        results["conditional"][f"bin_by_{by}_score_{other}"] = rows

    F.write_json(F.OUT / "metrics" / "disagreement.json", results)

    print("\n=== disagreement sets (ID-referenced percentiles) ===")
    for tag in ("strict", "relaxed"):
        for label in ("geometry_only", "pnc_only"):
            r = results["sets"][f"{tag}/{label}"]
            print(f"\n[{tag}] {label}: n={r['count']:,}  "
                  f"{ {k: v for k, v in r['composition'].items() if v} }")
            if not r["count"]:
                continue
            for k in ("member_agreement", "mean_pairwise_kl", "mutual_information",
                      "msp", "base_entropy_T", "M1_true_label", "predictive_entropy",
                      "hidden_perturbation_change"):
                f_, b_ = r["fields"][k], r["background"][k]
                print(f"    {k:<28} median {f_['median']:>10.4f}   "
                      f"(all-data median {b_['median']:>10.4f})")
            if r["id_detail"]:
                d = r["id_detail"]
                print(f"    ID subset n={d['n_id']}  base correct "
                      f"{d['base_correct_rate']*100:.1f}% "
                      f"(overall {d['overall_base_correct_rate']*100:.1f}%)")

    print("\n=== conditional: OOD binned by Mahalanobis pct, scored by P&C entropy ===")
    for r in results["conditional"]["bin_by_M_score_P"]:
        if r.get("near_auroc") is None:
            print(f"  {r['bin']:<8} n_ood={r['n_ood']:>6,}  (too few)")
        else:
            print(f"  {r['bin']:<8} n_ood={r['n_ood']:>6,}  Near {r['near_auroc']*100:6.2f}"
                  f"  Far {r['far_auroc']*100:6.2f}")
    print("\n=== conditional: OOD binned by P&C entropy pct, scored by Mahalanobis ===")
    for r in results["conditional"]["bin_by_P_score_M"]:
        if r.get("near_auroc") is None:
            print(f"  {r['bin']:<8} n_ood={r['n_ood']:>6,}  (too few)")
        else:
            print(f"  {r['bin']:<8} n_ood={r['n_ood']:>6,}  Near {r['near_auroc']*100:6.2f}"
                  f"  Far {r['far_auroc']*100:6.2f}")
    print(f"\nwrote {F.OUT/'metrics'/'disagreement.json'}")
    return results


if __name__ == "__main__":
    run()
