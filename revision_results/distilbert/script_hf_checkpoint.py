"""Download + convert the Banking77 DistilBERT checkpoint to Flax (CPU-only).

optimum/distilbert-base-uncased-finetuned-banking77 is distributed as a PyTorch /
safetensors sequence classifier. We convert it once to a Flax parameter tree
(FlaxDistilBertForSequenceClassification, from_pt=True) and serialize it, so P&C never
re-runs the PT->Flax conversion. The Hub revision is pinned and all hashes recorded.

Run in the cloned env with GPU disabled:
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" .venv_bank/bin/python -m experiments.banking77_pnc.hf_checkpoint
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

MODEL_ID = "optimum/distilbert-base-uncased-finetuned-banking77"
OUT = Path("results/banking77_distilbert_pnc/checkpoint")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(revision: str | None = None) -> dict:
    import jax.numpy as jnp
    from flax.serialization import to_bytes
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer, AutoConfig, FlaxDistilBertForSequenceClassification

    OUT.mkdir(parents=True, exist_ok=True)

    # pin the exact commit revision (never rely on floating main in the final run)
    if revision is None:
        info = HfApi().model_info(MODEL_ID)
        revision = info.sha
    print(f"resolved revision: {revision}")
    local = snapshot_download(MODEL_ID, revision=revision)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=revision)
    model = FlaxDistilBertForSequenceClassification.from_pretrained(
        MODEL_ID, revision=revision, from_pt=True, dtype=jnp.float32)

    # serialize the Flax params + config + tokenizer
    (OUT / "flax_params.msgpack").write_bytes(to_bytes(model.params))
    config.to_json_file(OUT / "config.json")
    tokenizer.save_pretrained(OUT / "tokenizer")

    # shape + dtype manifests
    import jax
    flat = jax.tree_util.tree_leaves_with_path(model.params)
    shapes = {".".join(str(getattr(k, "key", k)) for k in path): list(leaf.shape) for path, leaf in flat}
    dtypes = {".".join(str(getattr(k, "key", k)) for k in path): str(leaf.dtype) for path, leaf in flat}
    (OUT / "param_shapes.json").write_text(json.dumps(shapes, indent=2))
    (OUT / "param_dtypes.json").write_text(json.dumps(dtypes, indent=2))

    n_params = int(sum(int(__import__("numpy").prod(s)) for s in shapes.values()))
    manifest = {
        "model_id": MODEL_ID, "revision": revision, "local_snapshot": local,
        "n_parameters": n_params, "num_labels": int(config.num_labels),
        "id2label": {int(k): v for k, v in config.id2label.items()},
        "flax_params_sha256": _sha256(OUT / "flax_params.msgpack"),
        "config_sha256": _sha256(OUT / "config.json"),
        "download_date_unix": time.time(),
        "reported_accuracy": 0.925,
        "jax_stack": {"note": "converted in cloned env; JAX/Flax versions pinned"},
    }
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"n_parameters = {n_params:,} (expect ~67M) | num_labels = {config.num_labels}")
    print(f"saved -> {OUT}/flax_params.msgpack ({(OUT/'flax_params.msgpack').stat().st_size/1e6:.1f} MB)")
    return manifest


if __name__ == "__main__":
    m = convert()
    print(json.dumps({k: m[k] for k in ("revision", "n_parameters", "num_labels")}, indent=2))
