#!/usr/bin/env python
"""Copy every located rebuttal artifact into revision_results/ and checksum it.

Each entry names a source (a path in the working tree, or a blob recovered from
git HEAD) and the result group it belongs to. Nothing is transformed: files land
byte-identical under revision_results/<group>/. Normalized derivatives are produced
by the sibling verify_*.py scripts and are checksummed here too.

Emits revision_results/provenance/artifact_inventory.csv with, per file:
source path, destination path, SHA-256, size, source mtime, and git-tracking state.

Usage: .venv/bin/python revision_results/provenance/collect_artifacts.py
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RR = ROOT / "revision_results"

# (group, source-spec, destination filename)
# source-spec "git:<path>" recovers the blob from HEAD (the working tree deletes it).
COPY: list[tuple[str, str, str]] = []


def add(group: str, src: str, dst: str | None = None) -> None:
    COPY.append((group, src, dst or Path(src).name))


# ------------------------------------------------------------ 1 MuJoCo sensitivity
S = "results/neurips_2026_rebuttal/mujoco_sensitivity"
for f in [
    "MUJOCO_FAR_SENSITIVITY_RESULTS.md",
    "MANIFEST.md",
    "CALIB_FIX_VALIDATION.md",
    "anchors.json",
]:
    add("mujoco_sensitivity", f"{S}/{f}")
add("mujoco_sensitivity", f"{S}/aggregates/far_sensitivity_raw.csv")
add(
    "mujoco_sensitivity",
    f"{S}/aggregates/far_sensitivity_raw.csv.pre_bootfull_merge.bak",
    "far_sensitivity_raw_11720rows_CANONICAL_FOR_QUOTED_NUMBERS.csv",
)
add("mujoco_sensitivity", f"{S}/aggregates/far_by_environment.csv")
add("mujoco_sensitivity", f"{S}/aggregates/far_across_envs.csv")
add("mujoco_sensitivity", f"{S}/aggregates/far_bootfull_raw.csv")
add("mujoco_sensitivity", f"{S}/tables/far_sensitivity_table.csv")
add("mujoco_sensitivity", f"{S}/tables/far_sensitivity_table.md")
for f in ["aggregate_far.py", "aggregate.py", "run_sensitivity.py", "gen_data.py",
          "merge_bootfull.py", "far_plots.py", "plots.py"]:
    add("mujoco_sensitivity", f"{S}/scripts/{f}", f"script_{f}")
add("mujoco_sensitivity",
    f"{S}/archive_wrong_anchor_lam0/MUJOCO_SENSITIVITY_RESULTS.md",
    "SUPERSEDED_wrong_anchor_MUJOCO_SENSITIVITY_RESULTS.md")
add("mujoco_sensitivity",
    "results/neurips_2026_rebuttal/priority1/sensitivity_mujoco.csv",
    "priority1_halfcheetah_3seed_sensitivity.csv")
add("mujoco_sensitivity",
    "results/neurips_2026_rebuttal/priority1/sensitivity_mujoco_agg.csv",
    "priority1_halfcheetah_3seed_sensitivity_agg.csv")
add("mujoco_sensitivity",
    "results/neurips_2026_rebuttal/priority1/sensitivity_summary.md",
    "priority1_halfcheetah_3seed_summary.md")
add("mujoco_sensitivity",
    "scripts/neurips_2026_rebuttal/priority1_sensitivity.py",
    "script_priority1_sensitivity.py")

# ------------------------------------------------------------------ 2 CIFAR
add("cifar_sensitivity", "scripts/neurips_2026_rebuttal/cifar_other_machine/README.md",
    "OTHER_MACHINE_README.md")
add("cifar_sensitivity", "scripts/neurips_2026_rebuttal/cifar_other_machine/run_cifar_sensitivity.sh",
    "OTHER_MACHINE_run_cifar_sensitivity.sh")
add("cifar_sensitivity", "experiments/cifar10_tuning_plan.md")
add("cifar_sensitivity", "experiments/cifar_neurips_strengthening_plan.md")
add("cifar_sensitivity", "experiments/cifar10_ood_detection.md")
add("cifar_sensitivity", "experiments/cifar10_ood_paper_table.md")
add("cifar_sensitivity", "experiments/cifar_paper_tables.md")

# --------------------------------------------------------------------- 3 SCOD
add("scod", "results/posthoc_mujoco/scod/SCOD_REPORT.md")
add("scod", "results/posthoc_mujoco/scod/run_summary.json", "scod_run_summary.json")
add("scod", "results/posthoc_mujoco/MANIFEST.json", "posthoc_mujoco_MANIFEST.json")
add("scod", "experiments/posthoc/run_scod.py", "script_run_scod.py")
add("scod", "experiments/posthoc/scod_adapter.py", "script_scod_adapter.py")
add("scod", "experiments/posthoc/validate_scod.py", "script_validate_scod.py")
add("scod", "experiments/posthoc/scod_distribution.py", "script_scod_distribution.py")

# --------------------------------------------------------------- 4 Efficiency
E = "results/neurips_2026_rebuttal"
for f in ["efficiency_summary.md", "efficiency_cifar_inference.csv",
          "efficiency_construction_mujoco.csv", "efficiency_storage.csv",
          "efficiency_inference_mujoco.csv"]:
    add("efficiency", f"{E}/{f}")
add("efficiency", f"{E}/efficiency_v2/efficiency_v2_rows.csv")
add("efficiency", f"{E}/efficiency_v2/efficiency_v2_table.md")
for env in ["Ant-v5", "HalfCheetah-v5", "Hopper-v5"]:
    add("efficiency", f"{E}/efficiency_v2/efficiency_{env}_seed0_steps5000.json")
add("efficiency", "scripts/neurips_2026_rebuttal/priority4_efficiency.py",
    "script_priority4_efficiency.py")
add("efficiency", "scripts/neurips_2026_rebuttal/priority4_inference_bench.py",
    "script_priority4_inference_bench.py")
add("efficiency", "experiments/scripts/efficiency_benchmark.py", "script_efficiency_benchmark.py")
add("efficiency", "experiments/scripts/aggregate_efficiency.py", "script_aggregate_efficiency.py")
add("efficiency", "git:results/cifar10/inference_cost.json",
    "cifar10_inference_cost_RECOVERED_FROM_GIT_HEAD.json")

# --------------------------------------------------------------- 5 DistilBERT
B = "results/banking77_distilbert_pnc"
add("distilbert", f"{B}/TRANSFORMER_FULL_REPORT.md")
add("distilbert", f"{B}/REPORT.md", "REPORT_short.md")
add("distilbert", f"{B}/tables/banking77_pnc.csv")
add("distilbert", f"{B}/tables/banking77_pnc.md")
add("distilbert", f"{B}/tables/banking77_pnc.tex")
add("distilbert", f"{B}/metrics/raw.csv", "metrics_raw_per_seed.csv")
add("distilbert", f"{B}/metrics/by_dataset.csv", "metrics_by_dataset.csv")
add("distilbert", f"{B}/metrics/across_seeds.json", "metrics_across_seeds.json")
add("distilbert", f"{B}/environment/environment_before.txt", "environment.txt")
add("distilbert", "experiments/banking77_pnc/REUSE_MAP.md")
add("distilbert", "experiments/banking77_pnc/hf_checkpoint.py", "script_hf_checkpoint.py")
add("distilbert", "experiments/banking77_pnc/transformer_adapter.py", "script_transformer_adapter.py")
add("distilbert", "experiments/banking77_pnc/evaluate.py", "script_evaluate.py")
add("distilbert", "experiments/banking77_pnc/aggregate.py", "script_aggregate.py")
add("distilbert", "experiments/banking77_pnc/data.py", "script_data.py")
add("distilbert", "experiments/banking77_pnc/baselines.py", "script_baselines.py")
add("distilbert", "experiments/banking77_pnc/clinc_domains.json")
add("distilbert", "experiments/banking77_pnc/run_queue.sh", "script_run_queue.sh")

# ---------------------------------------------------------------- 6 Mechanism
P3 = "results/neurips_2026_rebuttal/priority3"
add("mechanism", f"{P3}/mechanism_second_domain_summary.md")
add("mechanism", f"{P3}/mechanism_second_domain.csv")
add("mechanism", f"{P3}/mechanism_HalfCheetah-v5_seed0.csv")
add("mechanism", f"{P3}/mechanism_HalfCheetah-v5_seed0.json")
add("mechanism", f"{P3}/mechanism_Hopper-v5_seed0.csv")
add("mechanism", f"{P3}/mechanism_Hopper-v5_seed0.json")
add("mechanism", "scripts/neurips_2026_rebuttal/priority3_mechanism.py",
    "script_priority3_mechanism.py")
add("mechanism", "experiments/random_vs_low_evidence.md")
add("mechanism", "experiments/random_vs_low_hypotheses_plan.md")
add("mechanism", "experiments/scripts/random_vs_low_diagnostic.py",
    "script_random_vs_low_diagnostic.py")
add("mechanism", "experiments/scripts/plot_random_vs_low_evidence.py",
    "script_plot_random_vs_low_evidence.py")
add("mechanism", "results/neurips_2026_rebuttal/priority2/linearization_summary.md")
add("mechanism", "results/neurips_2026_rebuttal/priority2/linearization_diagnostics.csv")
add("mechanism", "scripts/neurips_2026_rebuttal/priority2_linearization.py",
    "script_priority2_linearization.py")

# ------------------------------------------------- 7 Finite-scale theory validation
add("finite_scale_validation", "pnc_theory_reports/THEOREM_VALIDATION.md")
add("finite_scale_validation", "pnc_theory_reports/MULTILAYER_THEORY_VALIDATION.md")
add("finite_scale_validation", "pnc_theory_reports/AUDIT_IMPLEMENTATION.md")
add("finite_scale_validation", "reports/finite_transfer/METHODS.md")
add("finite_scale_validation", "reports/finite_transfer/RESULTS_MANIFEST.md")
add("finite_scale_validation", "reports/finite_transfer/tables/part0_identity.allenv.md")
add("finite_scale_validation", "artifacts/finite_transfer/mujoco_tier1_v2/identity.csv",
    "finite_identity_mujoco_tier1_v2.csv")
add("finite_scale_validation", "artifacts/finite_transfer/mujoco_tier1_v2/provenance.json",
    "finite_identity_mujoco_tier1_v2_provenance.json")
add("finite_scale_validation", "artifacts/finite_transfer/mujoco_tier1/identity.csv",
    "finite_identity_mujoco_tier1_v1.csv")
add("finite_scale_validation", "analysis/finite_transfer/finite_identity.py",
    "script_finite_identity.py")
add("finite_scale_validation", "experiments/scripts/pnc_theory/validate_bridge.py",
    "script_validate_bridge.py")
add("finite_scale_validation", "experiments/scripts/pnc_theory/multilayer.py",
    "script_multilayer.py")
add("finite_scale_validation", "experiments/scripts/pnc_theory/validate_multilayer.py",
    "script_validate_multilayer.py")
add("finite_scale_validation", "experiments/scripts/pnc_theory/linalg.py", "script_linalg.py")
add("finite_scale_validation", "experiments/scripts/pnc_theory/test_linalg.py",
    "unittest_test_linalg.py")

# ------------------------------------------------------------- 8 Shift tiers
add("shift_tiers", "results/neurips_2026_rebuttal/priority5_pnc_tiers.csv")
add("shift_tiers", "results/neurips_2026_rebuttal/priority5_full_mujoco_tables.csv")
add("shift_tiers", "scripts/neurips_2026_rebuttal/priority5_full_mujoco_tables.py",
    "script_priority5_full_mujoco_tables.py")
add("shift_tiers", "pnc_repro/figures/appendix_per_env_table_paper.txt")
add("shift_tiers", "pnc_repro/figures/appendix_per_env_table_paper.tex")
add("shift_tiers", "pnc_repro/figures/appendix_selected_hparams_paper.txt")
add("shift_tiers", "pnc_repro/figures/appendix_selected_hparams_paper.tex")
add("shift_tiers", "reports/tables/gym_tables.tex", "SUBMITTED_gym_tables.tex")
add("shift_tiers", "reports/tables/gym_settings_appendix.tex", "SUBMITTED_gym_settings_appendix.tex")
add("shift_tiers", "pnc_core/json_to_tex_table.py", "script_json_to_tex_table.py")

# ------------------------------------------------------------- 9 Layer scope
add("layer_scope", "pnc_theory_reports/LAYER_SELECTION.md")
add("layer_scope", "pnc_theory_reports/LAYER_PAIR_SELECTION.md")
add("layer_scope", "pnc_theory_reports/MULTILAYER_ID_ONLY_SELECTION.md")

# ------------------------------------------------------------- 10 Provenance
add("provenance", "results/neurips_2026_rebuttal/PROVENANCE.md", "PROVENANCE_original.md")
add("provenance", "results/neurips_2026_rebuttal/INVENTORY.md", "INVENTORY_original.md")
add("provenance", "results/neurips_2026_rebuttal/REPRODUCTION.md", "REPRODUCTION_original.md")
add("provenance", "results/neurips_2026_rebuttal/REBUTTAL_RESULTS.md", "REBUTTAL_RESULTS_original.md")
add("provenance", "results/neurips_2026_rebuttal/protocol_clarifications.md")
add("provenance", "docs/NON_EXPERIMENTAL_REBUTTAL_AUDIT.md")
add("provenance", "docs/pnc_experiment_protocol.md")
add("provenance", "docs/RESULTS_CATALOG.md")
add("provenance", "results/neurips_2026_rebuttal/repro/reproduction_ant.md")
add("provenance", "results/neurips_2026_rebuttal/repro/reproduction_ant.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked(rel: str) -> str:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                       capture_output=True, text=True)
    return "tracked" if r.returncode == 0 else "untracked"


def main() -> None:
    rows, missing = [], []
    for group, spec, dstname in COPY:
        dst = RR / group / dstname
        dst.parent.mkdir(parents=True, exist_ok=True)
        if spec.startswith("git:"):
            rel = spec[4:]
            r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                               capture_output=True)
            if r.returncode != 0:
                missing.append(spec)
                continue
            dst.write_bytes(r.stdout)
            mtime, state = "(git blob @ HEAD)", "deleted-in-worktree/in-git-HEAD"
            src_display = f"{rel} @ git HEAD"
        else:
            src = ROOT / spec
            if not src.exists():
                missing.append(spec)
                continue
            shutil.copy2(src, dst)
            mtime = datetime.fromtimestamp(src.stat().st_mtime, timezone.utc).isoformat()
            state = tracked(spec)
            src_display = spec
        rows.append(
            dict(
                result_group=group,
                source_path=src_display,
                copied_path=str(dst.relative_to(ROOT)),
                sha256=sha256(dst),
                bytes=dst.stat().st_size,
                source_mtime_utc=mtime,
                git_state=state,
            )
        )

    # checksum the normalized derivatives produced by the verify_*.py scripts.
    # artifact_inventory.csv is excluded: it is rewritten below, so any hash taken
    # here would be stale by the time the file lands.
    self_path = RR / "provenance/artifact_inventory.csv"
    for p in sorted(RR.rglob("*")):
        rel = str(p.relative_to(ROOT))
        if not p.is_file() or p == self_path or rel in {r["copied_path"] for r in rows}:
            continue
        if p.suffix not in {".csv", ".md", ".json", ".py", ".sha256"}:
            continue
        rows.append(
            dict(
                result_group=p.parent.name,
                source_path="(derived in this collection run)",
                copied_path=rel,
                sha256=sha256(p),
                bytes=p.stat().st_size,
                source_mtime_utc=datetime.fromtimestamp(
                    p.stat().st_mtime, timezone.utc
                ).isoformat(),
                git_state="new",
            )
        )

    df = pd.DataFrame(rows).sort_values(["result_group", "copied_path"])
    out = RR / "provenance/artifact_inventory.csv"
    df.to_csv(out, index=False)
    print(f"copied/indexed {len(df)} files -> {out.relative_to(ROOT)}")
    print(df.groupby("result_group").agg(files=("sha256", "size"),
                                         megabytes=("bytes", lambda s: round(s.sum() / 1e6, 2))).to_string())
    if missing:
        print("\nSOURCES NOT FOUND (recorded as missing):")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
