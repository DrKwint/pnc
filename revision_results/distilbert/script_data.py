"""Banking77 (ID) + CLINC-OOS (OOD) data prep for the DistilBERT P&C experiment.

All CPU-only. Label mapping is validated by STRING against the checkpoint's id2label
(never assume integer IDs coincide). A deterministic stratified 90/10 calibration/ID-val
split is drawn with split_seed=20260726 and the exact row indices saved. OOD data is only
loaded/labelled here; it is never used for any selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BANKING77_ID = "PolyAI/banking77"
CLINC_ID = "clinc_oos"                # config 'plus' includes the oos label
SPLIT_SEED = 20260726
CKPT = Path("results/banking77_distilbert_pnc/checkpoint")
DATA_OUT = Path("results/banking77_distilbert_pnc/data")
NEAR_FINANCIAL_DOMAINS = ["banking", "credit_cards"]


def _hash_list(xs) -> str:
    return hashlib.sha256(np.asarray(xs).tobytes()).hexdigest()[:16]


def ckpt_id2label() -> dict[int, str]:
    cfg = json.load(open(CKPT / "config.json"))
    return {int(k): v for k, v in cfg["id2label"].items()}


def load_banking77():
    from datasets import load_dataset
    # datasets>=4 dropped script datasets; PolyAI/banking77 ships a loader script,
    # so we pin datasets<4 and pass trust_remote_code.
    ds = load_dataset(BANKING77_ID, trust_remote_code=True)
    return ds


def validate_labels(ds) -> dict:
    """Match dataset ClassLabel names to checkpoint id2label by STRING. Returns a map
    dataset_label_id -> checkpoint_label_id and asserts a total bijection."""
    id2label = ckpt_id2label()
    label_to_ckptid = {v: k for k, v in id2label.items()}
    names = ds["train"].features["label"].names            # dataset's ordered class names
    ds_to_ckpt = {}
    missing = []
    for ds_id, name in enumerate(names):
        # normalize spacing/underscores conservatively; require exact string membership
        key = name if name in label_to_ckptid else name.replace("_", " ")
        if key in label_to_ckptid:
            ds_to_ckpt[ds_id] = label_to_ckptid[key]
        else:
            missing.append(name)
    ok = (len(ds_to_ckpt) == len(names) == len(id2label)) and not missing
    return {"ok": ok, "n_labels": len(names), "ds_to_ckpt": ds_to_ckpt,
            "unmatched": missing}


def make_stratified_split(train_labels, val_fraction=0.10, seed=SPLIT_SEED):
    """Deterministic stratified split -> (cal_idx, val_idx). ~10% per class to val."""
    rng = np.random.RandomState(seed)
    train_labels = np.asarray(train_labels)
    cal, val = [], []
    for c in np.unique(train_labels):
        idx = np.where(train_labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val.extend(idx[:n_val].tolist()); cal.extend(idx[n_val:].tolist())
    return np.sort(np.array(cal)), np.sort(np.array(val))


def length_audit(texts, tokenizer, cap=64):
    lens = [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]
    lens = np.array(lens)
    return {"median": float(np.median(lens)), "p95": float(np.percentile(lens, 95)),
            "p99": float(np.percentile(lens, 99)), "max": int(lens.max()),
            "frac_truncated_at_64": float(np.mean(lens > cap))}


def prepare_banking77():
    """Download, validate labels, split, length-audit. Saves a manifest. Light memory."""
    from transformers import AutoTokenizer
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(CKPT / "tokenizer"))
    ds = load_banking77()
    lab = validate_labels(ds)
    if not lab["ok"]:
        raise SystemExit(f"Banking77 label mapping FAILED: unmatched={lab['unmatched']}")

    cal_idx, val_idx = make_stratified_split(ds["train"]["label"])
    audit = length_audit(ds["train"]["text"][:2000], tok)   # sample for speed

    np.savez(DATA_OUT / "banking77_split.npz", cal_idx=cal_idx, val_idx=val_idx,
             ds_to_ckpt=np.array([lab["ds_to_ckpt"][i] for i in range(lab["n_labels"])]))
    manifest = {
        "banking77_id": BANKING77_ID, "n_train": len(ds["train"]), "n_test": len(ds["test"]),
        "n_labels": lab["n_labels"], "label_mapping_ok": lab["ok"],
        "split_seed": SPLIT_SEED, "n_calibration": int(len(cal_idx)), "n_id_val": int(len(val_idx)),
        "cal_idx_hash": _hash_list(cal_idx), "val_idx_hash": _hash_list(val_idx),
        "length_audit": audit, "recommended_max_length": 64 if audit["frac_truncated_at_64"] < 0.01
        else int(np.ceil(audit["p99"] / 16) * 16),
    }
    (DATA_OUT / "banking77_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    m = prepare_banking77()
    print(json.dumps(m, indent=2))
