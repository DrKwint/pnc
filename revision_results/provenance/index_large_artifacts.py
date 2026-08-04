#!/usr/bin/env python
"""Index the large artifacts deliberately NOT copied into git, with checksums.

Anything below is either a raw dataset, a model checkpoint, or a per-example
prediction dump. Committing them would add gigabytes to the repository, so the
manifest records the exact original path, size, mtime and SHA-256 instead.

Directories are indexed file-by-file into a per-directory checksum manifest under
revision_results/provenance/large_artifact_checksums/, and summarized (path, file
count, total bytes, SHA-256 of the sorted checksum manifest) in
revision_results/provenance/large_artifacts.csv. The manifest checksum lets a
future reader verify a whole tree with one comparison.

Usage: .venv/bin/python revision_results/provenance/index_large_artifacts.py
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_results/provenance"
MANIFEST_DIR = OUT / "large_artifact_checksums"

# (slug, path relative to repo root, glob, what it is, which result group needs it)
TARGETS = [
    ("scod_predictions", "results/posthoc_mujoco/scod/predictions", "**/*",
     "SCOD per-example ID/Near/Mid/Far scores (parquet + npz) for 11 envs x 3 seeds", "scod"),
    ("scod_sketches", "results/posthoc_mujoco/scod/sketches", "**/*.npz",
     "SCOD Nystrom Fisher sketches per env-seed", "scod"),
    ("scod_configs", "results/posthoc_mujoco/scod/configs", "**/*.json",
     "SCOD run configurations per env-seed", "scod"),
    ("scod_timing", "results/posthoc_mujoco/scod/timing", "**/*",
     "SCOD wall-clock timing records", "scod/efficiency"),
    ("pnc_theory_base_models", "artifacts/pnc_theory/base_models", "*.npz",
     "Frozen MuJoCo base MLPs used by the finite-scale theory validation", "finite_scale_validation"),
    ("posthoc_base_models_backup", "archive/artifacts_backup", "*",
     "Regenerated canonical base-model batch backing every SCOD run", "scod"),
    ("distilbert_checkpoint", "results/banking77_distilbert_pnc/checkpoint", "**/*",
     "Flax-converted DistilBERT Banking77 checkpoint + tokenizer", "distilbert"),
    ("distilbert_members", "results/banking77_distilbert_pnc/members", "**/*",
     "Per-seed corrected lin2 weights for the 5 P&C construction seeds", "distilbert"),
    ("distilbert_predictions", "results/banking77_distilbert_pnc/predictions", "**/*",
     "Per-method per-example logits/scores on Banking77 test + CLINC tiers", "distilbert"),
    ("finite_transfer_tier1_v2", "artifacts/finite_transfer/mujoco_tier1_v2", "*.parquet",
     "Per-example finite-transfer detail and example tables", "finite_scale_validation"),
    ("mujoco_sensitivity_raw_json", "results/neurips_2026_rebuttal/mujoco_sensitivity/raw", "*.json",
     "Per env-seed sensitivity sweep outputs (573 files) feeding far_sensitivity_raw.csv",
     "mujoco_sensitivity"),
    ("mechanism_pnc_npz", "results/neurips_2026_rebuttal/priority3/pnc", "**/*",
     "P&C ensembles used by the HalfCheetah/Hopper mechanism replication", "mechanism"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for slug, rel, pattern, what, group in TARGETS:
        base = ROOT / rel
        if not base.exists():
            rows.append(dict(slug=slug, original_path=rel, needed_by=group, description=what,
                             status="ABSENT", n_files=0, total_bytes=0,
                             manifest_file="", manifest_sha256=""))
            continue
        files = sorted(p for p in base.glob(pattern) if p.is_file())
        lines, total = [], 0
        for p in files:
            digest = sha256(p)
            size = p.stat().st_size
            total += size
            lines.append(f"{digest}  {size}  {p.relative_to(ROOT)}")
        mf = MANIFEST_DIR / f"{slug}.sha256"
        mf.write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append(
            dict(
                slug=slug,
                original_path=rel,
                needed_by=group,
                description=what,
                status="PRESENT",
                n_files=len(files),
                total_bytes=total,
                total_megabytes=round(total / 1e6, 2),
                newest_mtime_utc=(
                    datetime.fromtimestamp(
                        max(p.stat().st_mtime for p in files), timezone.utc
                    ).isoformat()
                    if files
                    else ""
                ),
                manifest_file=str(mf.relative_to(ROOT)),
                manifest_sha256=sha256(mf),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "large_artifacts.csv", index=False)
    print(df[["slug", "status", "n_files", "total_megabytes", "manifest_sha256"]].to_string(index=False))
    print(f"\ntotal indexed: {df.total_bytes.sum() / 1e9:.2f} GB across {int(df.n_files.sum())} files")


if __name__ == "__main__":
    main()
