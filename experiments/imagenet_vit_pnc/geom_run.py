"""Runner for the geometry / SCOD / LLLA follow-up.

    llla      Part C — LLLA-Kron (and optional LLLA-Diag) via official laplace-torch
    analysis  Part A — geometry vs P&C mechanism diagnostic
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.banking77_pnc.pnc_metrics import clf_metrics

from . import full_cache as fc
from . import full_ood as fo
from . import full_oodmetrics as fm
from . import llla_kron as lk
from .memprobe import GIB, cpu_peak_rss_gib, reset_cuda
from .vit_adapter import ViTPnCAdapter

SRC = Path("results/neurips_2026_rebuttal/imagenet_vit")
OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_geometry_scod_llla")
GROUPS = {k: v[2] for k, v in fo.DATASETS.items()}
DS = ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"]


def _phi(ad, cache_name: str) -> torch.Tensor:
    c = fc.load_cache(SRC / "raw" / cache_name)
    X = fc.cache_to_gpu(c, ad)
    p = fc.features_from_cache(ad, X)
    p = torch.from_numpy(p.cpu().numpy()).to(ad.device, ad.dtype)
    del X
    reset_cuda()
    return p


def stage_llla(args):
    structures = ["kron"] + (["diag"] if args.diag else [])
    results = {}
    for st in structures:
        name = f"LLLA-{'Kron' if st == 'kron' else 'Diag'}"
        print(f"\n== {name} ==")
        ad, la, rows, best, info = lk.fit(st)
        head = ad.model.heads[-1] if hasattr(ad.model.heads, "__getitem__") \
            else ad.model.heads

        gate = lk.validate_closed_form()
        print(f"  closed-form predictive gate: max rel error {gate['max_rel_error']:.2e} "
              f"-> {'OK' if gate['ok'] else 'FAIL'}")

        (OUT / "id_selection").mkdir(parents=True, exist_ok=True)
        with (OUT / "id_selection" / f"{name.lower()}_selection.csv").open(
                "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        fc.write_json(OUT / "id_selection" / f"{name.lower()}_selected.json", {
            "method": name, "package": "laplace-torch",
            "version": info["laplace_version"], "curvlinops": info["curvlinops_version"],
            "subset_of_weights": "last_layer", "hessian_structure": st,
            "last_layer_name": "head", "n_cal": info["n_cal"],
            "prior_precision": best["prior_precision"],
            "selection_criterion": "ID selection-pool NLL",
            "pred_type": "glm", "link_approx": "probit",
            "predictive_note": "closed-form functional variance; laplace-torch's own "
                               "predictive materialises an 11,718 GiB Jacobian for a "
                               "1000-class head",
            "closed_form_gate": gate,
            "OOD data accessed before selection": "NO",
            "grid": [r["prior_precision"] for r in rows]})

        val = fc.load_cache(SRC / "raw" / "cache_val50k.npz")
        phi_v = _phi(ad, "cache_val50k.npz")
        yv = val["labels"]
        t0 = time.perf_counter()
        p_id = lk.glm_probit_probs(la, head, phi_v)
        t_id = time.perf_counter() - t0
        base_pred = head(phi_v).argmax(-1).cpu().numpy()
        m = clf_metrics(p_id, yv)
        top5 = torch.from_numpy(p_id).topk(5, -1).indices.numpy()
        ent = lambda p: -np.sum(p * np.log(p + 1e-12), -1)
        idm = {"top1": m["accuracy"],
               "top5": float((top5 == yv[:, None]).any(-1).mean()),
               "nll": m["nll"], "ece": m["ece"],
               "base_agreement": float((p_id.argmax(-1) == base_pred).mean()),
               "mean_pred_entropy": float(ent(p_id).mean()),
               "predict_seconds_50k": t_id}
        print(f"  ID: top-1 {idm['top1']*100:.3f}%  NLL {idm['nll']:.4f}  "
              f"ECE {idm['ece']:.4f}  agree {idm['base_agreement']:.4f}")
        parity = lk.mc_parity(la, head, phi_v)
        print(f"  GLM-probit vs MC link: entropy Spearman "
              f"{parity['entropy_spearman']:.4f}, top-1 agreement "
              f"{parity['top1_agreement']:.4f}")

        id_s = ent(p_id)
        per_ds, ood_s = {}, {}
        for ds in DS:
            phi_o = _phi(ad, f"cache_ood_{ds}.npz")
            p_o = lk.glm_probit_probs(la, head, phi_o)
            ood_s[ds] = ent(p_o)
            per_ds[ds] = {"group": GROUPS[ds], "n_ood": int(len(ood_s[ds])),
                          **fm.binary_ood_metrics(id_s, ood_s[ds])}
            print(f"    {ds:<13} AUROC {per_ds[ds]['auroc']*100:6.2f}  "
                  f"FPR95 {per_ds[ds]['fpr95']*100:6.2f}")
            del phi_o, p_o
            reset_cuda()
        agg = {g: fm.aggregate_family_metrics(
            id_s, {k: v for k, v in ood_s.items() if GROUPS[k] == g})
            for g in ("near", "far")}
        print(f"  Near AUROC {agg['near']['mean_auroc']*100:.2f}  "
              f"Far AUROC {agg['far']['mean_auroc']*100:.2f}")
        np.savez_compressed(OUT / "predictions" / f"{name.lower()}_scores.npz",
                            id_entropy=id_s.astype(np.float32),
                            **{f"ood_{d}": ood_s[d].astype(np.float32) for d in DS})
        results[name] = {"selected": best, "info": info, "id_metrics": idm,
                         "mc_parity": parity, "per_dataset": per_ds, "aggregate": agg,
                         "closed_form_gate": gate}
        del phi_v, la
        reset_cuda()
    fc.write_json(OUT / "metrics" / "llla_results.json", results)
    print(f"\nwrote {OUT/'metrics'/'llla_results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["llla", "analysis", "scod"])
    ap.add_argument("--diag", action="store_true")
    args = ap.parse_args()
    if args.stage == "llla":
        stage_llla(args)
    elif args.stage == "analysis":
        from .geom_analysis import run
        run()
    else:
        from .scod_ll import run
        run()


if __name__ == "__main__":
    main()
