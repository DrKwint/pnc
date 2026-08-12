"""Driver for the ViT-B/16 P&C TITAN X preflight.

Stages share state through ``raw/state.json`` so the safe batch size chosen in
``memory`` is reused by every later stage.

    .venv_vit/bin/python -m experiments.imagenet_vit_pnc.run_preflight --stage memory
    .venv_vit/bin/python -m experiments.imagenet_vit_pnc.run_preflight --stage correction
    .venv_vit/bin/python -m experiments.imagenet_vit_pnc.run_preflight --stage ensemble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import stages_memory as sm
from .memprobe import Recorder, nvsmi_used_gib
from .vit_adapter import ViTPnCAdapter

OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_preflight")

THROUGHPUT_COLUMNS = [
    "test", "batch_size", "dtype", "token_mode", "member_count",
    "rows_per_image", "rows_per_s", "images_per_s", "median_s", "mean_s",
    "xtx_accum_s_per_batch", "proj_1000_images_s", "proj_4000_images_s",
    "rows_over_correction_dim", "peak_allocated_gib", "peak_reserved_gib", "notes",
]


def state_path() -> Path:
    return OUT / "raw" / "state.json"


def load_state() -> dict:
    p = state_path()
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    state_path().parent.mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(st, indent=2, default=str))
    return st


def stage_memory(args):
    rec = Recorder(OUT / "memory_preflight.csv")
    thr = Recorder(OUT / "throughput_preflight.csv", THROUGHPUT_COLUMNS)
    baseline = nvsmi_used_gib()
    print(f"GPU baseline usage before load: {baseline:.3f} GiB (other processes)")

    ad = ViTPnCAdapter(device="cuda", dtype=torch.float32)
    model_only = torch.cuda.memory_allocated() / sm.GIB
    print(f"model resident (fp32 weights): {model_only:.3f} GiB\n")

    print("== section 6: batch-size sweep (synthetic B x 3 x 224 x 224) ==")
    sweep = sm.stage_batch_sweep(ad, rec, thr, baseline_other_gib=baseline)
    safe = sweep["safe_batch"]
    print(f"\n  safe batch size = {safe} "
          f"(budget {sweep['vram_budget_gib']:.2f} GiB after {sm.HEADROOM_GIB} GiB headroom)\n")

    print("== sustained throughput at the safe batch (thermal settling) ==")
    sust = sm.stage_sustained(ad, rec, thr, safe, seconds=args.sustain_s)
    print(f"  {sust['first_window_images_per_s']:.1f} -> {sust['last_window_images_per_s']:.1f} "
          f"img/s ({sust['decay_pct']:.1f}% decay); settled = "
          f"{sust['settled_images_per_s']:.1f} img/s")
    print(f"  clock {sust['gpu_state_start'].get('sm_clock_mhz')} -> "
          f"{sust['gpu_state_end'].get('sm_clock_mhz')} MHz, temp "
          f"{sust['gpu_state_start'].get('temp_c')} -> {sust['gpu_state_end'].get('temp_c')} C\n")

    print("== section 7: FFN activation capture ==")
    acts = sm.stage_activations(ad, rec, safe)
    print(f"  T={acts['n_tokens']}  h={acts['h_ffn_input']['mib']:.1f} MiB  "
          f"y={acts['y_post_gelu']['mib']:.1f} MiB  z={acts['z_ffn_output']['mib']:.1f} MiB "
          f"(batch {safe})\n")

    print("== section 8: perturbation basis ==")
    basis, U, coeffs, scale = sm.stage_basis(ad, rec, seed=args.seed, K=args.K, M=args.M)
    print(f"  U {basis['basis_shape']} = {basis['basis_mib']:.1f} MiB CPU in "
          f"{basis['basis_build_s']:.2f}s; one dW1 {basis['member_dW1_mib']:.1f} MiB in "
          f"{basis['member_dW1_build_s']*1e3:.1f} ms")
    print(f"  compact state for M={args.M}: {basis['compact_state_mib']:.1f} MiB vs "
          f"{basis['twenty_full_models_gib']:.2f} GiB for {args.M} full models\n")

    print("== section 9: single perturbed member + restoration ==")
    single = sm.stage_single_member(ad, rec, U, coeffs, scale, safe)
    print(f"  finite={single['finite_perturbed']}  max|dlogit|="
          f"{single['max_abs_logit_change']:.4f}  top1 agree="
          f"{single['top1_agreement_base_vs_uncorrected']:.3f}")
    print(f"  restore bitwise equal = {single['restore_bitwise_equal']}  "
          f"peak alloc = {single['peak_allocated_gib']:.3f} GiB\n")

    sm.write_json(OUT / "raw" / "stage_memory.json", {
        "gpu_baseline_other_gib": baseline, "model_resident_gib": model_only,
        "batch_sweep": sweep, "sustained": sust, "activations": acts,
        "basis": basis, "single_member": single})
    save_state(safe_batch=safe, sustained_images_per_s=sust["settled_images_per_s"],
               model_resident_gib=model_only,
               gpu_baseline_other_gib=baseline, numerical_scale=scale,
               n_tokens=acts["n_tokens"], seed=args.seed, K=args.K, M=args.M)
    print(f"wrote {OUT/'raw'/'stage_memory.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["memory", "correction", "quality", "ensemble", "member"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--n-calib", type=int, default=8192)
    ap.add_argument("--n-heldout", type=int, default=2048)
    ap.add_argument("--sustain-s", type=float, default=60.0)
    ap.add_argument("--row-budget", type=int, default=32768,
                    help="correction rows per member for the ensemble stage")
    ap.add_argument("--token-mode", default="cls")
    ap.add_argument("--naive-members", type=int, default=2,
                    help="members timed on the prefix-re-run path (sec 18 baseline)")
    args = ap.parse_args()

    if args.stage == "memory":
        stage_memory(args)
    elif args.stage == "correction":
        from .stages_correction import run_correction
        run_correction(args, OUT, THROUGHPUT_COLUMNS)
    elif args.stage == "ensemble":
        from .stages_correction import run_ensemble
        run_ensemble(args, OUT, THROUGHPUT_COLUMNS)
    elif args.stage == "quality":
        from .stages_correction import run_quality
        run_quality(args, OUT)
    else:
        from .stages_correction import run_member_only
        run_member_only(args, OUT)


if __name__ == "__main__":
    main()
