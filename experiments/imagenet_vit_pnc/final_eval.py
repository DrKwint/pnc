"""Revision closure Part G/H/I: THE canonical ImageNet evaluator.

Every ImageNet number that reaches the manuscript is produced here, from
per-example score arrays, through one metric implementation, with one sign
convention and one aggregation. Nothing is copied from a Markdown report.

Why this exists: successive diagnostic rounds reported P&C Far AUROC as 87.74,
87.89 and 87.93. Those are not disagreements about the model — they are three
different *evaluation artifacts*:

  87.74  a seed-0-only recomputation done inside the geometry round
  87.89  the frozen 5-seed preservation-frontier ensemble
  87.93  the K-sweep's own K=20 arm, 5 seeds, rebuilt from a nested basis

This script fixes the canonical choice as **the frozen preservation-frontier
primary ensemble (r=2, lambda=1000, K=20, M=20), 5 seeds**, which is the object
the manuscript describes, and recomputes every comparator against the same
50,000 ImageNet validation predictions and the same five OpenOOD sets.

One trap this pins down: the frontier stores its OOD scores as a single pooled
85,908-vector concatenated in **alphabetical** dataset order
(inaturalist, ninco, openimage_o, ssb_hard, textures), while most other artifacts
use the DS order (ssb_hard first). Splitting with the wrong order silently
scrambles every per-dataset number.

Outputs (into 2026neurips_rebuttal_results/revision_experiment_closure/):
    imagenet_final_scores.csv        one row per method: Near/Far AUROC+FPR95, ID error
    imagenet_final_per_dataset.csv   one row per (method, dataset)
    imagenet_final_table.tex         the manuscript table
    imagenet_final_provenance.json   sha256 of every per-example file consumed
    imagenet_bootstrap.csv           Part H paired bootstrap CIs
    imagenet_mechanism_summary.csv   Part I canonical mechanism quantities
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import full_oodmetrics as fm

R = Path("results/neurips_2026_rebuttal")
OUT = Path("2026neurips_rebuttal_results/revision_experiment_closure")
VIT, FRONT = R / "imagenet_vit", R / "imagenet_vit_preservation_frontier"
BASE, GEO = R / "imagenet_vit_baselines", R / "imagenet_vit_geometry_scod_llla"
FOL = R / "imagenet_vit_geometry_followup"

DS = ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"]
NEAR, FAR = ["ssb_hard", "ninco"], ["inaturalist", "textures", "openimage_o"]
SIZES = {"ssb_hard": 49000, "ninco": 5879, "inaturalist": 10000,
         "textures": 5160, "openimage_o": 15869}
# the frontier's pooled OOD vector is concatenated over sorted(dataset names)
POOLED_ORDER = sorted(DS)
SEEDS = [0, 10, 42, 123, 2026]
BOOT_N, BOOT_SEED = 10000, 20260815

_HASHES: dict[str, str] = {}


def _load(path: Path):
    p = Path(path)
    if str(p) not in _HASHES:
        _HASHES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return np.load(p, allow_pickle=True)


def _split_pooled(v: np.ndarray) -> dict[str, np.ndarray]:
    assert len(v) == sum(SIZES.values()), (len(v), sum(SIZES.values()))
    out, at = {}, 0
    for d in POOLED_ORDER:
        out[d] = v[at:at + SIZES[d]]
        at += SIZES[d]
    return out


# --------------------------------------------------------------- method loading
def _base_logits() -> np.ndarray:
    p = VIT / "raw" / "base_val_logits.npy"
    if str(p) not in _HASHES:
        _HASHES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return np.load(p)


def collect() -> tuple[dict, dict]:
    """Return {method: {'id': arr, 'ood': {ds: arr}, 'seeds': optional list}}.

    Sign convention throughout: HIGHER = MORE OOD.
    """
    M: dict[str, dict] = {}

    # ---- P&C and its matched uncorrected ablation: frozen frontier primary ----
    pooled = _load(FRONT / "predictions" / "ood_scores.npz")
    for name, key in (("P&C", "PandC_primary"), ("Uncorrected perturbation",
                                                 "Uncorrected_primary")):
        ids, oods = [], []
        for s in SEEDS:
            z = _load(FRONT / "predictions" / f"id_val_primary_seed{s}.npz")
            ids.append(z["pnc_entropy" if key.startswith("PandC") else "unc_entropy"])
            oods.append(_split_pooled(pooled[f"{key}(seed{s})"]))
        M[name] = {"id_seeds": ids, "ood_seeds": oods, "seeds": SEEDS}

    # ---- Mahalanobis (frozen headline; class-conditional, shared covariance) ----
    M["Mahalanobis"] = {
        "id": _load(BASE / "predictions" / "id_val_baselines.npz")["mahalanobis"],
        "ood": {d: _load(BASE / "predictions" / f"ood_{d}_baselines.npz")["mahalanobis"]
                for d in DS}}

    # ---- MSP and base entropy: from the followup's persisted base scores ----
    bid = _load(FOL / "predictions" / "base_scores_val50k.npz")
    bood = {d: _load(FOL / "predictions" / f"base_scores_{d}.npz") for d in DS}
    M["MSP"] = {"id": -bid["msp"], "ood": {d: -bood[d]["msp"] for d in DS}}
    M["Base entropy (T=0.7)"] = {"id": bid["base_entropy_T"],
                                 "ood": {d: bood[d]["base_entropy_T"] for d in DS}}
    M["Base entropy (raw)"] = {"id": bid["base_entropy_raw"],
                               "ood": {d: bood[d]["base_entropy_raw"] for d in DS}}

    # ---- Energy: recomputed exactly from the frozen base logits (ID side was
    #      never persisted); OOD side comes from the original experiment's arrays.
    lg = _base_logits().astype(np.float64)
    mx = lg.max(-1, keepdims=True)
    id_energy = -(mx[:, 0] + np.log(np.exp(lg - mx).sum(-1)))     # -logsumexp
    M["Energy"] = {"id": id_energy,
                   "ood": {d: _load(VIT / "predictions" / f"ood_{d}.npz")["Energy"]
                           for d in DS}}

    # ---- LLLA variants ----
    ll = _load(GEO / "predictions" / "llla-kron_scores.npz")
    M["LLLA-Kron"] = {"id": ll["id_entropy"],
                      "ood": {d: ll[f"ood_{d}"] for d in DS}}
    p = FOL / "predictions" / "llla-kron-temp_scores.npz"
    if p.exists():
        lt = _load(p)
        M["LLLA-Kron+Temp"] = {"id": lt["id_entropy"],
                               "ood": {d: lt[f"ood_{d}"] for d in DS}}

    # ---- SCOD scopes (exact categorical Fisher) ----
    for scope, label in (("linear", "SCOD-linear"), ("ffn", "SCOD-FFN")):
        p = FOL / "predictions" / f"scod_{scope}_scores.npz"
        if p.exists():
            z = _load(p)
            M[label] = {"id": z["val50k"], "ood": {d: z[d] for d in DS}}

    # ---- P&C mutual information / expected member entropy (appendix rows) ----
    g_id = _load(GEO / "predictions" / "geometry_scores_val50k.npz")
    g_ood = {d: _load(GEO / "predictions" / f"geometry_scores_{d}.npz") for d in DS}
    for key, label in (("mutual_information", "P&C mutual information (seed 0)"),
                       ("expected_member_entropy",
                        "P&C expected member entropy (seed 0)")):
        M[label] = {"id": g_id[key], "ood": {d: g_ood[d][key] for d in DS}}

    return M, {"pooled_order": POOLED_ORDER, "seeds": SEEDS}


# ------------------------------------------------------------------- metrics
def _one(id_s, ood_s) -> dict:
    per = {d: fm.binary_ood_metrics(id_s, ood_s[d]) for d in DS}
    agg = {}
    for g, members in (("near", NEAR), ("far", FAR)):
        a = fm.aggregate_family_metrics(id_s, {k: ood_s[k] for k in members})
        agg[f"{g}_auroc"] = a["mean_auroc"]
        agg[f"{g}_fpr95"] = a["mean_fpr95"]
    return {"per_dataset": per, **agg}


def evaluate(M: dict) -> dict:
    labels = _load(VIT / "predictions" / "id_val_seed0.npz")["labels"]
    base_pred = _base_logits().argmax(-1)
    err = (base_pred != labels).astype(int)
    out = {}
    for name, m in M.items():
        if "seeds" in m:
            rows = [_one(i, o) for i, o in zip(m["id_seeds"], m["ood_seeds"])]
            agg = {k: float(np.mean([r[k] for r in rows]))
                   for k in ("near_auroc", "far_auroc", "near_fpr95", "far_fpr95")}
            agg.update({f"{k}_std": float(np.std([r[k] for r in rows], ddof=1))
                        for k in ("near_auroc", "far_auroc")})
            agg["per_dataset"] = {d: {"auroc": float(np.mean(
                [r["per_dataset"][d]["auroc"] for r in rows])),
                "fpr95": float(np.mean([r["per_dataset"][d]["fpr95"] for r in rows]))}
                for d in DS}
            ide = [roc_auc_score(err, i) for i in m["id_seeds"]]
            agg["id_error_auroc"] = float(np.mean(ide))
            agg["id_error_auroc_std"] = float(np.std(ide, ddof=1))
            agg["id_error_aupr"] = float(np.mean(
                [average_precision_score(err, i) for i in m["id_seeds"]]))
            agg["n_seeds"] = len(m["seeds"])
        else:
            agg = _one(m["id"], m["ood"])
            agg["per_dataset"] = {d: {"auroc": agg["per_dataset"][d]["auroc"],
                                      "fpr95": agg["per_dataset"][d]["fpr95"]}
                                  for d in DS}
            agg["id_error_auroc"] = float(roc_auc_score(err, m["id"]))
            agg["id_error_aupr"] = float(average_precision_score(err, m["id"]))
            agg["n_seeds"] = 1
        out[name] = agg
    return out


# ---------------------------------------------------------- Part H: bootstrap
class _BootAUROC:
    """Exact Mann-Whitney AUROC under a paired example bootstrap, O(n) per draw.

    A naive implementation re-sorts ~55k scores per (replicate, dataset, method),
    which is hours at 10,000 replicates. Instead, sort the ID scores ONCE and
    precompute each OOD point's insertion interval. A bootstrap replicate then
    only needs a bincount + cumsum over the ID order (giving the resampled ID
    ECDF) and a gather at the precomputed positions. Ties get midranks, so this
    reproduces the sorted-rank AUROC exactly rather than approximating it.
    """

    def __init__(self, id_scores: np.ndarray, ood: dict[str, np.ndarray]):
        self.n_id = len(id_scores)
        order = np.argsort(id_scores, kind="mergesort")
        self.id_rank = np.empty(self.n_id, np.int64)
        self.id_rank[order] = np.arange(self.n_id)
        srt = id_scores[order]
        self.lo = {d: np.searchsorted(srt, v, side="left") for d, v in ood.items()}
        self.hi = {d: np.searchsorted(srt, v, side="right") for d, v in ood.items()}

    def replicate(self, id_idx, ood_idx) -> dict[str, float]:
        c = np.bincount(self.id_rank[id_idx], minlength=self.n_id)
        cc = np.concatenate([[0], np.cumsum(c)])
        out = {}
        for d in self.lo:
            u = 0.5 * (cc[self.lo[d]] + cc[self.hi[d]])     # midrank for ties
            out[d] = float(u[ood_idx[d]].mean()) / self.n_id
        return out


def bootstrap(M: dict, comparators: list[str], target: str = "P&C") -> list[dict]:
    """Paired bootstrap CIs for mean-over-seeds Near/Far AUROC differences.

    Each replicate draws ONE set of example indices (ID and each OOD set) and
    scores every method on it, which is what makes the comparison paired. For a
    multi-seed method the replicate's AUROC is averaged over its seeds first, so
    the interval is on exactly the quantity the table reports.
    """
    names = [target] + comparators
    boots: dict[str, list[_BootAUROC]] = {}
    for n in names:
        m = M[n]
        if "seeds" in m:
            boots[n] = [_BootAUROC(i, o) for i, o in zip(m["id_seeds"], m["ood_seeds"])]
        else:
            boots[n] = [_BootAUROC(m["id"], m["ood"])]

    n_id = len(M["Mahalanobis"]["id"])
    rng = np.random.default_rng(BOOT_SEED)
    acc = {n: {"near": np.empty(BOOT_N), "far": np.empty(BOOT_N)} for n in names}
    for b in range(BOOT_N):
        id_idx = rng.integers(0, n_id, n_id)
        ood_idx = {d: rng.integers(0, SIZES[d], SIZES[d]) for d in DS}
        for n in names:
            per_seed = [bo.replicate(id_idx, ood_idx) for bo in boots[n]]
            for g, members in (("near", NEAR), ("far", FAR)):
                acc[n][g][b] = float(np.mean(
                    [np.mean([ps[d] for d in members]) for ps in per_seed]))
        if (b + 1) % 2000 == 0:
            print(f"    {b + 1}/{BOOT_N} replicates", flush=True)

    rows = []
    for comp in comparators:
        for g in ("near", "far"):
            d = acc[target][g] - acc[comp][g]
            rows.append({"target": target, "comparator": comp, "family": g,
                         "delta_auroc_mean": float(d.mean()),
                         "ci95_lo": float(np.percentile(d, 2.5)),
                         "ci95_hi": float(np.percentile(d, 97.5)),
                         "p_delta_gt_0": float((d > 0).mean()),
                         "excludes_zero": bool(np.percentile(d, 2.5) > 0
                                               or np.percentile(d, 97.5) < 0),
                         "n_replicates": BOOT_N, "seed": BOOT_SEED,
                         "interval": "percentile"})
            print(f"  {target} - {comp:<26} {g:<4} {d.mean()*100:+6.2f}  "
                  f"95% CI [{np.percentile(d, 2.5)*100:+.2f}, "
                  f"{np.percentile(d, 97.5)*100:+.2f}]"
                  f"{'  *' if rows[-1]['excludes_zero'] else ''}", flush=True)
    return rows


def per_dataset_deltas(M: dict, comparators: list[str], target: str = "P&C") -> list[dict]:
    """Point differences per dataset (no resampling) — Part H's second ask."""
    def per_ds(name):
        m = M[name]
        if "seeds" in m:
            return {d: float(np.mean([fm.binary_ood_metrics(i, o[d])["auroc"]
                                      for i, o in zip(m["id_seeds"], m["ood_seeds"])]))
                    for d in DS}
        return {d: fm.binary_ood_metrics(m["id"], m["ood"][d])["auroc"] for d in DS}
    t = per_ds(target)
    rows = []
    for c in comparators:
        cv = per_ds(c)
        for d in DS:
            rows.append({"target": target, "comparator": c, "dataset": d,
                         "target_auroc": t[d], "comparator_auroc": cv[d],
                         "delta_auroc": t[d] - cv[d]})
    return rows


# ------------------------------------------------------- Part I: mechanism
def mechanism_summary() -> list[dict]:
    """Canonical mechanism quantities, regenerated through one script (Part I)."""
    rows = []
    ks = json.loads((FOL / "metrics" / "ksweep_final.json").read_text())
    for K in (5, 20, 40, 80):
        a = ks["K"][str(K)]
        rows += [{"quantity": f"P&C K={K} Near AUROC", "value": a["predictive_entropy_near_auroc"],
                  "std": a["predictive_entropy_near_auroc_std"], "n": 5,
                  "source": "metrics/ksweep_final.json"},
                 {"quantity": f"P&C K={K} Far AUROC", "value": a["predictive_entropy_far_auroc"],
                  "std": a["predictive_entropy_far_auroc_std"], "n": 5,
                  "source": "metrics/ksweep_final.json"}]
    rp = json.loads((FOL / "metrics" / "random_projection_mahalanobis.json").read_text())
    for r in rp["variants"]["class_conditional"]:
        rows.append({"quantity": f"random-projection Mahalanobis K={r['K']} Near AUROC",
                     "value": r["near_auroc_mean"], "std": r["near_auroc_std"],
                     "n": rp["n_projections"],
                     "source": "metrics/random_projection_mahalanobis.json"})
    sp = json.loads((FOL / "metrics" / "mahalanobis_spectrum.json").read_text())
    hi = [c for c in sp["cumulative"] if c["which"] == "highest-variance" and c["k"] == 96][0]
    lo = [c for c in sp["cumulative"] if c["which"] == "lowest-variance" and c["k"] == 96][0]
    rows += [{"quantity": "highest-variance 96 covariance modes Near AUROC",
              "value": hi["near_auroc"], "std": None, "n": 1,
              "source": "metrics/mahalanobis_spectrum.json"},
             {"quantity": "lowest-variance 96 covariance modes Near AUROC",
              "value": lo["near_auroc"], "std": None, "n": 1,
              "source": "metrics/mahalanobis_spectrum.json"}]
    al = json.loads((FOL / "metrics" / "spectral_alignment.json").read_text())
    rows.append({"quantity": "rho(P&C response energy, covariance eigenvalue)",
                 "value": al["spearman"]["E_vs_lambda"], "std": None, "n": 1,
                 "source": "metrics/spectral_alignment.json"})
    rows.append({"quantity": "rho(P&C response energy, inverse eigenvalue)",
                 "value": al["spearman"]["E_vs_inv_lambda"], "std": None, "n": 1,
                 "source": "metrics/spectral_alignment.json"})
    dis = json.loads((FOL / "metrics" / "disagreement.json").read_text())
    for tag, lab in (("relaxed/geometry_only", "geometry-only case count (90/60)"),
                     ("relaxed/pnc_only", "P&C-only case count (90/60)")):
        rows.append({"quantity": lab, "value": dis["sets"][tag]["count"], "std": None,
                     "n": 1, "source": "metrics/disagreement.json"})
    return rows


def _tex(res: dict, order: list[str]) -> str:
    def f(v, s=None):
        if v is None:
            return "--"
        return (f"{v*100:.2f}" if s is None or s == 0
                else f"{v*100:.2f}\\,$\\pm$\\,{s*100:.2f}")
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Method & ID err.\ AUROC & Near AUROC & Far AUROC & Near FPR95 & Far FPR95 \\",
         r"\midrule"]
    for m in order:
        if m not in res:
            continue
        r = res[m]
        L.append(f"{m.replace('&', chr(92)+'&')} & {f(r['id_error_auroc'])} & "
                 f"{f(r['near_auroc'], r.get('near_auroc_std'))} & "
                 f"{f(r['far_auroc'], r.get('far_auroc_std'))} & "
                 f"{f(r['near_fpr95'])} & {f(r['far_fpr95'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


MAIN_ORDER = ["MSP", "Base entropy (T=0.7)", "Mahalanobis", "LLLA-Kron",
              "SCOD-linear", "Uncorrected perturbation", "P&C"]
APPENDIX_ORDER = ["Energy", "Base entropy (raw)", "LLLA-Kron+Temp", "SCOD-FFN",
                  "P&C expected member entropy (seed 0)",
                  "P&C mutual information (seed 0)"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== collecting per-example score arrays ===")
    M, meta = collect()
    print(f"  {len(M)} methods; pooled OOD split order {meta['pooled_order']}")
    res = evaluate(M)

    print(f"\n{'method':<38}{'IDerr':>8}{'Near':>8}{'Far':>8}{'NearFPR':>9}{'FarFPR':>8}")
    for m, r in res.items():
        print(f"  {m:<36}{r['id_error_auroc']*100:>8.2f}{r['near_auroc']*100:>8.2f}"
              f"{r['far_auroc']*100:>8.2f}{r['near_fpr95']*100:>9.2f}"
              f"{r['far_fpr95']*100:>8.2f}")

    with (OUT / "imagenet_final_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_seeds", "id_error_auroc", "id_error_aupr",
                    "near_auroc", "near_auroc_std", "far_auroc", "far_auroc_std",
                    "near_fpr95", "far_fpr95"])
        for m, r in res.items():
            w.writerow([m, r["n_seeds"], r["id_error_auroc"], r["id_error_aupr"],
                        r["near_auroc"], r.get("near_auroc_std", ""),
                        r["far_auroc"], r.get("far_auroc_std", ""),
                        r["near_fpr95"], r["far_fpr95"]])
    with (OUT / "imagenet_final_per_dataset.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "dataset", "family", "auroc", "fpr95"])
        for m, r in res.items():
            for d in DS:
                w.writerow([m, d, "near" if d in NEAR else "far",
                            r["per_dataset"][d]["auroc"], r["per_dataset"][d]["fpr95"]])
    (OUT / "imagenet_final_table.tex").write_text(
        "% main table\n" + _tex(res, MAIN_ORDER) +
        "\n\n% appendix rows\n" + _tex(res, APPENDIX_ORDER) + "\n")

    print("\n=== Part H: paired bootstrap (10,000 replicates, seed 20260815) ===")
    comps = [c for c in ["Mahalanobis", "LLLA-Kron", "SCOD-linear", "MSP",
                         "Uncorrected perturbation"] if c in M]
    boot = bootstrap(M, comps)
    with (OUT / "imagenet_bootstrap.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(boot[0]))
        w.writeheader()
        w.writerows(boot)
    pdd = per_dataset_deltas(M, comps)
    with (OUT / "imagenet_bootstrap_per_dataset.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, list(pdd[0]))
        w.writeheader()
        w.writerows(pdd)

    mech = mechanism_summary()
    with (OUT / "imagenet_mechanism_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["quantity", "value", "std", "n", "source"])
        w.writeheader()
        w.writerows(mech)

    (OUT / "imagenet_final_provenance.json").write_text(json.dumps({
        "canonical_pnc": "preservation-frontier primary, r=2, lambda=1000, K=20, M=20, "
                         "5 seeds (0,10,42,123,2026)",
        "ridge_center": "original affine map (verified in ridge_center_verification.json)",
        "temperature": 0.7,
        "id_set": "ImageNet-1k val, 50,000 images, untouched",
        "ood_sets": {d: SIZES[d] for d in DS},
        "pooled_ood_concat_order": POOLED_ORDER,
        "score_sign": "higher = more OOD",
        "metric_code": "experiments/imagenet_vit_pnc/full_oodmetrics.py",
        "family_aggregation": "macro mean over datasets in the family",
        "bootstrap": {"replicates": BOOT_N, "seed": BOOT_SEED,
                      "interval": "percentile", "paired": True},
        "sha256": _HASHES}, indent=2))
    print(f"\nwrote {OUT}/imagenet_final_*.csv|tex, imagenet_bootstrap.csv, "
          f"imagenet_mechanism_summary.csv ({len(_HASHES)} files hashed)")


if __name__ == "__main__":
    main()
