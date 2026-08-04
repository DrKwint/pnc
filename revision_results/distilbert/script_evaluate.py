"""Banking77 DistilBERT post-hoc P&C — full evaluation pipeline.

Per construction seed: ID-only scale selection, temperature fit, then evaluate every
method (deterministic base MSP/Energy/Entropy, internal FFN P&C, uncorrected
perturbation, head P&C, MC Dropout) on ID test + CLINC Near/Cross/Far. Saves member
artifacts, per-example predictions, and a raw metrics CSV. OOD is never used for tuning.

Scope: --scope cpu_pilot (small, CPU parity) or full (config seeds/scales, all data).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import sys
import numpy as np
import jax
import jax.numpy as jnp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))   # allow bare sibling imports under -m
import construct as C
import baselines as B
import pnc_metrics as PM
from transformer_adapter import DistilBertPnCAdapter, CKPT

OUT = Path("results/banking77_distilbert_pnc")
DATA = OUT / "data"


def load_splits(cfg, scope):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(CKPT / "tokenizer"))
    ML = cfg["model"]["max_length"]
    sp = np.load(DATA / "banking77_split.npz")
    cal_idx, val_idx, ds2ck = sp["cal_idx"], sp["val_idx"], sp["ds_to_ckpt"]
    b77 = load_dataset(cfg["data"]["banking77_id"], trust_remote_code=True)
    tr_txt = b77["train"]["text"]; tr_lab = np.array(b77["train"]["label"])
    te_txt = b77["test"]["text"]; te_lab = np.array([ds2ck[l] for l in b77["test"]["label"]])

    dom = json.load(open(cfg["data"]["clinc_domains"]))
    clinc = load_dataset(cfg["data"]["clinc_id"], cfg["data"]["clinc_config"], trust_remote_code=True)
    cn = clinc["test"].features["intent"].names
    fin = {cn.index(x) for x in dom["banking"] + dom["credit_cards"]}; oos = cn.index("oos")
    ci = np.array(clinc["test"]["intent"]); ct = clinc["test"]["text"]
    near = [ct[i] for i in np.where(np.isin(ci, list(fin)))[0]]
    cross = [ct[i] for i in np.where(~np.isin(ci, list(fin)) & (ci != oos))[0]]
    far = [ct[i] for i in np.where(ci == oos)[0]]

    if scope == "cpu_pilot":
        p = cfg["runtime"]["cpu_pilot"]
        cal_idx = cal_idx[:p["n_calibration"]]; val_idx = val_idx[:min(len(val_idx), 256)]
        te_sel = slice(0, p["n_id_eval"]); n_ood = p["n_ood_per_set"]
        te_txt, te_lab = te_txt[te_sel], te_lab[te_sel]
        near, cross, far = near[:n_ood], cross[:n_ood], far[:n_ood]

    def tok_np(texts):
        e = tok(list(texts), max_length=ML, padding="max_length", truncation=True, return_tensors="np")
        return e["input_ids"], e["attention_mask"]

    splits = {
        "id_val": dict(texts=[tr_txt[i] for i in val_idx], labels=tr_lab[val_idx]),
        "id_eval": dict(texts=list(te_txt), labels=te_lab),
        "near": dict(texts=near, labels=None), "cross": dict(texts=cross, labels=None),
        "far": dict(texts=far, labels=None),
    }
    cal = dict(texts=[tr_txt[i] for i in cal_idx], labels=tr_lab[cal_idx])
    return tok, tok_np, splits, cal, ds2ck


def capture(ad, tok_np, texts, bs=32):
    """h0 (FFN input, token 0), block-out (head input), and ids/mask for all texts."""
    ids, mask = tok_np(texts)
    H, O = [], []
    for i in range(0, len(texts), bs):
        h = ad.capture_h0(ids[i:i + bs], mask[i:i + bs])
        H.append(np.asarray(h)); O.append(np.asarray(ad.block_output(jnp.asarray(h))))
    return np.concatenate(H), np.concatenate(O), ids, mask


def base_logits_from_h0(ad, H, bs=256):
    return np.concatenate([np.asarray(ad.tail(jnp.asarray(H[i:i + bs]))) for i in range(0, len(H), bs)])


def select_scale(ad, Hcal, Zcal, Hval, yval, scales, seed, M, K, ridge, constraints):
    """ID-only scale selection by ensemble ID-val NLL, subject to acc/agreement constraints."""
    base_val = base_logits_from_h0(ad, Hval); base_pred = base_val.argmax(1)
    base_acc = (base_pred == yval).mean()
    results = []
    for sm in scales:
        members, U, diag = C.build_members(ad, Hcal, Zcal, seed, sm, M=M, K=K, ridge=ridge)
        ens = C.ensemble_logits(ad, jnp.asarray(Hval), members, U, diag["numerical_scale"])
        sc = C.uncertainty_scores(ens); pbar = sc["pbar"]
        nll = float(-np.mean(np.log(pbar[np.arange(len(yval)), yval] + 1e-12)))
        acc = (pbar.argmax(1) == yval).mean(); agree = (pbar.argmax(1) == base_pred).mean()
        ok = (base_acc - acc) * 100 <= constraints["max_acc_drop"] and agree >= constraints["min_agree"]
        results.append(dict(scale_mult=sm, nll=nll, acc=float(acc), agreement=float(agree),
                            ok=bool(ok), numerical_scale=diag["numerical_scale"],
                            cal_residual=diag["calibration_residual_median"]))
    valid = [r for r in results if r["ok"]] or results
    best = min(valid, key=lambda r: r["nll"])
    # tie-break toward larger scale within ~1 SE (approx: 5% of best NLL)
    close = [r for r in valid if r["nll"] <= best["nll"] * 1.05]
    best = max(close, key=lambda r: r["scale_mult"])
    return best, results


def run_seed(ad, seed, cfg, scope, splits, caps, cal_h, cal_Z, cal_O, out_dir):
    pc = cfg["pnc"]; M = pc["ensemble_size"]; K = pc["perturbation_rank"]; ridge = float(pc["ridge"])
    scales = cfg["runtime"]["cpu_pilot"]["scale_multipliers"] if scope == "cpu_pilot" else pc["scale_multipliers"]
    cons = {"max_acc_drop": cfg["selection"]["max_accuracy_drop_percentage_points"],
            "min_agree": cfg["selection"]["min_base_member_agreement"]}
    yval = splits["id_val"]["labels"]; Hval = caps["id_val"][0]

    # --- ID-only scale selection ---
    best, sweep = select_scale(ad, cal_h, cal_Z, Hval, yval, scales, seed, M, K, ridge, cons)
    sm = best["scale_mult"]
    # --- freeze + build final members (FFN P&C, uncorrected, head P&C) ---
    members, U, diag = C.build_members(ad, cal_h, cal_Z, seed, sm, M=M, K=K, ridge=ridge)
    scale = diag["numerical_scale"]
    unc_m, unc_U, unc_s = B.build_uncorrected(ad, seed, sm, M=M, K=K)
    head_m, head_s = B.build_head_pnc(ad, cal_O, seed, sm, M=M, K=K, ridge=ridge)

    # --- temperature on ID-val base logits ---
    T = PM.fit_temperature(base_logits_from_h0(ad, Hval), yval) if cfg["evaluation"]["temperature_scaling"] else 1.0

    # --- save members + config ---
    md = out_dir / "members" / f"seed_{seed}"; md.mkdir(parents=True, exist_ok=True)
    np.savez(md / "basis.npz", U=U.astype(np.float32))
    np.savez(md / "coefficients.npz", coeffs=np.stack([m["coeff"] for m in members]))
    np.savez(md / "corrected_lin2.npz", W2=np.stack([m["W2"] for m in members]),
             b2=np.stack([m["b2"] for m in members]))
    (md / "config.json").write_text(json.dumps({
        "seed": seed, "selected_scale_multiplier": sm, "numerical_scale": scale,
        "temperature": T, "ridge": ridge, "M": M, "K": K,
        "rel_dW1_median": diag["rel_dW1_median"], "calibration_residual": diag["calibration_residual_median"],
        "scale_sweep": sweep, "used_ood_for_selection": False}, indent=2, default=float))

    # --- evaluate all methods on all splits ---
    def scores_pnc(H):
        return C.uncertainty_scores(C.ensemble_logits(ad, jnp.asarray(H), members, U, scale), T)
    def scores_unc(H):
        return C.uncertainty_scores(B.uncorrected_logits(ad, jnp.asarray(H), unc_m, unc_U, unc_s), T)
    def scores_head(O):
        return C.uncertainty_scores(B.head_pnc_logits(ad, jnp.asarray(O), head_m), T)

    rows = []
    for split, d in splits.items():
        H, O, ids, mask = caps[split]
        lab = d["labels"]
        base = base_logits_from_h0(ad, H)
        det = B.deterministic_scores(base, T)
        pnc = scores_pnc(H); unc = scores_unc(H); head = scores_head(O)
        mc = B.mc_dropout_logits(ad.model, ad.P, ids, mask, n_passes=(5 if scope == "cpu_pilot" else 20), seed=seed)
        mc_sc = C.uncertainty_scores(mc, T)
        methods = {
            "base_msp": dict(u=det["msp_uncertainty"], p=det["probs"]),
            "base_energy": dict(u=det["energy"], p=det["probs"]),
            "base_entropy": dict(u=det["predictive_entropy"], p=det["probs"]),
            "ffn_pnc": dict(u=pnc["predictive_entropy"], p=pnc["pbar"]),
            "uncorrected": dict(u=unc["predictive_entropy"], p=unc["pbar"]),
            "head_pnc": dict(u=head["predictive_entropy"], p=head["pbar"]),
            "mc_dropout": dict(u=mc_sc["predictive_entropy"], p=mc_sc["pbar"]),
        }
        for name, mm in methods.items():
            row = {"method": name, "seed": seed, "split": split, "n": len(H),
                   "selected_scale": sm, "temperature": T}
            if lab is not None:
                row.update({f"id_{k}": v for k, v in PM.clf_metrics(mm["p"], lab).items()})
            rows.append((name, split, mm, row))
    return rows, best, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scope", default="full", choices=["cpu_pilot", "full"])
    ap.add_argument("--output-dir", default=str(OUT))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out_dir = Path(a.output_dir)
    seeds = cfg["runtime"]["cpu_pilot"]["seeds"] if a.scope == "cpu_pilot" else cfg["pnc"]["construction_seeds"]

    t0 = time.time()
    ad = DistilBertPnCAdapter()
    tok, tok_np, splits, cal, ds2ck = load_splits(cfg, a.scope)
    print(f"[eval] splits: " + ", ".join(f"{k}={len(v['texts'])}" for k, v in splits.items()))
    caps = {k: capture(ad, tok_np, v["texts"]) for k, v in splits.items()}
    cal_h, cal_O, _, _ = capture(ad, tok_np, cal["texts"])
    cal_Z = np.asarray(jax.nn.gelu(jnp.asarray(cal_h) @ ad.W1 + ad.b1, approximate=False) @ ad.W2 + ad.b2)
    print(f"[eval] captured activations for all splits + {len(cal_h)} calib rows ({time.time()-t0:.0f}s)")

    # ID-vs-OOD uncertainty arrays for OOD metrics, collected per (method, seed)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    raw = []
    for seed in seeds:
        ts = time.time()
        rows, best, T = run_seed(ad, seed, cfg, a.scope, splits, caps, cal_h, cal_Z, cal_O, out_dir)
        # collect uncertainty per (method, split) to compute OOD metrics vs id_eval
        u = {}
        for name, split, mm, row in rows:
            u.setdefault(name, {})[split] = mm["u"]
            # per-example parquet
            pdir = out_dir / "predictions" / name / str(seed); pdir.mkdir(parents=True, exist_ok=True)
            try:
                import pandas as pd
                pd.DataFrame({"uncertainty": np.asarray(mm["u"]),
                              "pred": mm["p"].argmax(1)}).to_parquet(pdir / f"{split}.parquet", index=False)
            except Exception:
                np.savez(pdir / f"{split}.npz", uncertainty=np.asarray(mm["u"]), pred=mm["p"].argmax(1))
        for name, split, mm, row in rows:
            if split in ("near", "cross", "far"):
                om = PM.ood_metrics(u[name]["id_eval"], u[name][split])
                row.update({f"ood_{k}": v for k, v in om.items()})
            raw.append(row)
        print(f"[eval] seed {seed}: scale={best['scale_mult']}x T={T:.2f} nll={best['nll']:.3f} ({time.time()-ts:.0f}s)")

    cols = sorted({k for r in raw for k in r})
    with open(out_dir / "metrics" / "raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in raw:
            w.writerow(r)
    print(f"[eval] wrote {len(raw)} rows -> metrics/raw.csv ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
