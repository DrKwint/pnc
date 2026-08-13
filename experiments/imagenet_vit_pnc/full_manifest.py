"""Provenance manifest for the full experiment (spec §34)."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

COMMANDS = [
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_splits",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage parity",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage cache",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage stage_a",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage stage_b",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage robustness",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage final --n-seeds 5",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_run --stage ood",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_timing",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_tables",
    ".venv_vit/bin/python -m experiments.imagenet_vit_pnc.full_validate",
]


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {exc}>"


def build(out: Path) -> str:
    import torch
    import torchvision

    from .full_data import N_TRAIN_SHARDS, REPO_ID
    from .full_ood import CANONICAL_N, DATASETS, SPLITS_REPO
    from .full_search import COARSE_R, GATE, STAGE_B_LAMBDAS, STAGE_B_NCAL

    j = lambda p: json.loads((out / p).read_text()) if (out / p).exists() else {}
    splits = j("splits/splits_summary.json")
    sel = j("selection/selected_config.json")
    base = j("metrics/base_id_metrics.json")
    idf = j("metrics/id_final.json")
    temp = j("metrics/temperature.json")
    ck = base.get("checkpoint", {})
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None

    L = ["# MANIFEST — ImageNet ViT-B/16 final-block P&C", "",
         "## Code", "",
         f"- git commit: `{_sh('git rev-parse HEAD')}`",
         f"- branch: `{_sh('git rev-parse --abbrev-ref HEAD')}`",
         f"- working tree clean: {'yes' if not _sh('git status --porcelain -- experiments/imagenet_vit_pnc') else 'no (see diff)'}",
         "", "## Environment", "",
         f"- GPU: {props.name if props else 'n/a'} "
         f"({props.total_memory/1024**3:.2f} GiB, sm_{props.major}{props.minor})"
         if props else "- GPU: n/a",
         f"- driver: {_sh('nvidia-smi --query-gpu=driver_version --format=csv,noheader')}",
         f"- CUDA (torch runtime): {torch.version.cuda}",
         f"- torch: {torch.__version__}",
         f"- torchvision: {torchvision.__version__}",
         f"- python: {sys.version.split()[0]}  ({sys.executable})",
         f"- CPU: {_sh('grep -m1 \'model name\' /proc/cpuinfo').split(':', 1)[-1].strip()}"
         f" ({_sh('nproc')} threads), RAM "
         f"{_sh('grep MemTotal /proc/meminfo').split(':', 1)[-1].strip()}",
         f"- platform: {platform.platform()}",
         "", "## Checkpoint", "",
         f"- weight enum: `{ck.get('weight_enum')}`",
         f"- url: {ck.get('url')}",
         f"- SHA-256: `{ck.get('sha256')}`",
         f"- parameters: {ck.get('n_params'):,}" if ck.get("n_params") else "",
         f"- preprocessing: `{ck.get('transform')}`",
         "", "## Data", "",
         f"- ImageNet source: `{REPO_ID}` (gated ILSVRC-2012 originals), "
         f"{N_TRAIN_SHARDS} train shards + all validation shards",
         f"- training rows indexed: {splits.get('n_train_rows'):,}"
         if splits.get("n_train_rows") else "",
         "- ID evaluation: the official 50,000-image validation split, untouched by "
         "construction, selection and calibration",
         f"- OOD splits: OpenOOD canonical image lists from `{SPLITS_REPO}`; images from "
         "the `torch-uncertainty` mirrors",
         "", "| pool | n | classes | per-class | SHA-256 |", "|---|---|---|---|---|"]
    for name, m in sorted(splits.get("pools", {}).items()):
        L.append(f"| {name} | {m['n_images']} | {m['n_classes']} | "
                 f"{m['per_class_min']}-{m['per_class_max']} | `{m['sha256']}` |")
    L += ["", f"- split seed: {splits.get('split_seed')}", "",
          "| OOD dataset | n | group |", "|---|---|---|"]
    for k, (_, _, g) in DATASETS.items():
        L.append(f"| {k} | {CANONICAL_N[k]} | {g} |")

    L += ["", "## P&C configuration", "",
          f"- target block: `encoder.layers.encoder_layer_{sel.get('target_block')}`",
          "- perturbed: `.mlp.0` (W1, 768->3072); corrected: `.mlp.3` (W2, 3072->768)",
          f"- correction observation: {sel.get('token_mode')} token only, "
          "1 row per calibration image",
          f"- K: {sel.get('K')}   M (final): {sel.get('M_final')}",
          f"- construction seeds: {sorted(int(s) for s in idf.get('seeds', {}))}",
          "", "### Searched (ID data only)", "",
          f"- scales r (Stage A): {COARSE_R}",
          f"- scales r (Stage B): {j('selection/stage_a_summary.json').get('refined_grid')}",
          f"- Stage A r_boundary: {j('selection/stage_a_summary.json').get('r_boundary')}",
          f"- ridge lambda: {STAGE_B_LAMBDAS}",
          f"- correction sizes: {STAGE_B_NCAL}",
          f"- ID-stability gate: {json.dumps(GATE)}",
          "", "### Selected", "",
          f"```json\n{json.dumps(sel, indent=2)}\n```",
          "", "## Calibration and scoring", "",
          f"- temperature: {temp.get('temperature')} — {temp.get('protocol')}",
          "- P&C OOD score: predictive entropy of the temperature-scaled mean member "
          "softmax (fixed before OOD evaluation)",
          "- MSP: -max softmax. Energy: -logsumexp(logits). "
          "ReAct+Energy: Energy after clipping penultimate features at the ID p90",
          "- metrics/aggregation: `full_oodmetrics`, gated against "
          "`pnc_core/openood_eval.py`; Near/Far are macro means over their datasets",
          "", "## Commands", "", "```bash"] + COMMANDS + ["```", ""]
    L += ["## Reproducibility notes", "",
          "- every table is generated from `metrics/*.json` and `selection/*.csv` by "
          "`full_tables.py`; no number is transcribed by hand",
          "- images, checkpoints and activation caches are **not** committed; their "
          "sources and checksums are recorded above",
          "- `raw/cache_*.npz` and `raw/base_val_logits.npy` are regenerable from the "
          "commands above",
          ""]
    text = "\n".join(x for x in L if x != "")
    (out / "MANIFEST.md").write_text(text + "\n")
    return text


if __name__ == "__main__":
    print(build(Path("results/neurips_2026_rebuttal/imagenet_vit"))[:2500])
