"""Grid setup: checkpoint-hash verification, block-path audit, splits, GRID_SPEC, MANIFEST."""
from __future__ import annotations
import json, subprocess
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
import sys, os; os.chdir(REPO); sys.path.insert(0, str(REPO))
from flax import nnx
from experiments.scod_cifar.parameter_layout import load_base_model, checkpoint_path
from experiments.scod_cifar import protocol

ROOT = REPO / "results" / "neurips_2026_rebuttal" / "cifar" / "pnc_full_grid"
TARGET_BLOCKS = [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "splits").mkdir(exist_ok=True)
    # checkpoint hashes vs SCOD manifest
    scod = json.load(open(REPO / "results" / "scod_cifar" / "MANIFEST.json"))
    ck = {}
    for s in [0, 1, 2]:
        h = protocol.sha256_file(checkpoint_path(s))
        ck[str(s)] = dict(path=str(checkpoint_path(s)), sha256=h,
                          matches_scod=(h == scod["checkpoint_hashes"][str(s)]))
    json.dump(ck, open(ROOT / "checkpoint_manifest.json", "w"), indent=2)
    print("checkpoint hashes match SCOD:", all(v["matches_scod"] for v in ck.values()))

    # block-path audit (introspect model)
    model = load_base_model(0)
    stages = [model.stage1, model.stage2, model.stage3, model.stage4]
    rows = []
    for (si, bi) in TARGET_BLOCKS:
        blk = stages[si][bi]
        flax_stage = f"stage{si+1}"   # stage_idx s -> Flax module stage{s+1}
        c1 = np.asarray(blk.conv1.kernel.value); c2 = np.asarray(blk.conv2.kernel.value)
        has_ds = getattr(blk, "downsample", None) is not None
        rows.append(dict(label=f"s{si}b{bi}", stage_idx=si, block_idx=bi, flax_module=flax_stage,
                         conv1_path=f"{flax_stage}.{bi}.conv1.kernel", conv1_shape=list(c1.shape),
                         conv2_path=f"{flax_stage}.{bi}.conv2.kernel", conv2_shape=list(c2.shape),
                         in_channels=int(c1.shape[2]), out_channels=int(c1.shape[3]),
                         has_downsample=bool(has_ds)))
    json.dump(rows, open(ROOT / "block_path_audit.json", "w"), indent=2)
    md = ["# Block-path audit (mandatory)\n",
          "Label `sNbM` uses **stage_idx N** (0-indexed into [stage1..stage4]); the Flax module is "
          "`stage{N+1}`. So `s3b0` = stage_idx 3 = `stage4[0]`. The manuscript's selected block "
          "**(3,0) = s3b0**; the stale code default was **s3b1**. Both are in the grid; the report "
          "states which minimizes mean val NLL.\n",
          "| label | stage_idx | block_idx | Flax conv1 | conv2 | in→out ch | downsample |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']} | {r['stage_idx']} | {r['block_idx']} | {r['conv1_path']} | "
                  f"{r['conv2_path']} | {r['in_channels']}→{r['out_channels']} | {r['has_downsample']} |")
    (ROOT / "BLOCK_PATH_AUDIT.md").write_text("\n".join(md) + "\n")

    # splits (reuse submitted protocol exactly)
    perm = np.random.RandomState(99).permutation(50000)
    val_idx = perm[:5000]; train_idx = perm[5000:]
    np.save(ROOT / "splits" / "validation_indices.npy", val_idx)
    np.save(ROOT / "splits" / "train_indices.npy", train_idx)
    for s in [0, 1, 2]:
        cidx = np.random.RandomState(s).choice(45000, 1024, replace=False)
        np.save(ROOT / "splits" / f"calibration_indices_seed{s}.npy", cidx)

    # GRID_SPEC
    spec = dict(blocks=[f"s{si}b{bi}" for (si, bi) in TARGET_BLOCKS],
                target_blocks=TARGET_BLOCKS, scales=[25.0, 50.0, 100.0], fractions=[0.05, 0.10, 0.20],
                fixed=dict(direction_family="random", n_directions=20, n_members=50, lambda_reg=1e-3,
                           subset_size=1024, bn_mode="frozen", ridge_toward_original=True,
                           chunk_size="64 for stage_idx>=3 else 16 (memory; math-invariant)"),
                n_candidates_per_seed=54, n_total=162, checkpoint_seeds=[0, 1, 2],
                selection="lowest mean temperature-scaled ID-val NLL; tie: std, then lex(stage,block,scale,frac)",
                coordinate_descent_config=dict(label="s3b0", scale=25.0, bootstrap_frac=0.05,
                    note="artifact-backed submitted config; manuscript (3,0); s3b1 was stale default"))
    json.dump(spec, open(ROOT / "GRID_SPEC.json", "w"), indent=2)

    import jax, flax
    git = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
    man = dict(experiment="CIFAR-10 full P&C grid (6x3x3x3=162)", git_commit=git, git_dirty=dirty,
               jax=jax.__version__, flax=flax.__version__, gpu="RTX 5060 8GB",
               checkpoint_manifest="checkpoint_manifest.json", grid_spec="GRID_SPEC.json",
               protocol="split seed 99; 45000/5000; 1024 calib RandomState(seed).choice; ID-val NLL "
                        "selection after temperature scaling; no OOD in construction/selection")
    json.dump(man, open(ROOT / "MANIFEST.json", "w"), indent=2)
    print("wrote BLOCK_PATH_AUDIT, checkpoint_manifest, splits, GRID_SPEC, MANIFEST")
    for r in rows: print(f"  {r['label']}: {r['conv1_path']} in→out {r['in_channels']}→{r['out_channels']} ds={r['has_downsample']}")


if __name__ == "__main__":
    main()
