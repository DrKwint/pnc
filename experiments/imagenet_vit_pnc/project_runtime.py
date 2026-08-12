"""Project full-experiment runtime from the measured primitives (spec section 18).

Every number traces to a measurement in ``raw/``; nothing is assumed. The cost model
separates the five components the spec asks for:

    prefix computation      one ViT forward up to the target block, shared by all members
    member-tail computation per member, per image, downstream of the target FFN
    correction construction the (h, z) cache pass + per-member FFN/Gram accumulation
    CPU ridge solves        one shared Cholesky per member
    disk I/O                compact member state, written once

Two facts dominate and are measured per target block in ``raw/block_tail_costs.json``:

  * For the **final** block, everything after the FFN is token-wise followed by a CLS
    slice, so a member's tail is CLS-only: 0.015 ms/image/member.
  * For **earlier** blocks the later attention layers mix tokens, so the tail must run all
    remaining blocks on all 197 tokens: 0.85 ms (block 10) and 1.47 ms (block 9).

Wall-clock is derated to sustained throughput -- the card throttles ~13% from cold.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("results/neurips_2026_rebuttal/imagenet_vit_preflight")

# OpenOOD v1.5 ImageNet-1k OOD suite sizes. No OOD data is touched by this preflight;
# these counts come from the benchmark definition and only set the evaluation volume.
OPENOOD = {"SSB-hard": 49000, "NINCO": 5879, "iNaturalist": 10000,
           "Textures": 5640, "OpenImage-O": 15869}
IMAGENET_VAL = 50000
SELECTION_ID_VAL = 5000       # ID-val images used to pick block/scale (ID only, spec sec 12)


def _load(name):
    p = OUT / "raw" / name
    return json.loads(p.read_text()) if p.exists() else {}


def build(blocks=(11, 10, 9), scales=3, m_selection=10, m_final=20):
    mem, ens, blk = _load("stage_memory.json"), _load("stage_ensemble.json"), \
        _load("block_tail_costs.json")

    sustained = mem["sustained"]["settled_images_per_s"]
    cold = mem["batch_sweep"]["safe_batch_images_per_s"]
    derate = cold / sustained                       # ~1.14; block costs were timed warm-ish

    cached = ens["cached_construction"]
    cache_s = cached["cache_time_s"]                # (h, z) pass, per target block
    member_s = cached["median_member_s"]            # FFN + Gram + shared Cholesky
    solve_s = cached["median_cpu_solve_s"]
    gpu_s = cached["median_gpu_s"]
    naive_s = ens["member_build_times_s"][0]        # prefix re-run per member

    per_block = {}
    for b in blocks:
        d = blk[str(b)]
        per_block[b] = {
            "cls_only_tail": d["cls_only"],
            "prefix_ms": d["prefix_ms_per_img"],
            "tail_ms_per_member": d["tail_ms_per_img_per_member"],
            "ms_per_image_M20": d["total_ms_per_img_M20"],
            "images_per_s_M20": d["images_per_s_M20"],
            "peak_gib": d["peak_gib"],
        }

    def eval_s(block, n_images, M):
        d = per_block[block]
        return n_images * (d["prefix_ms"] + M * d["tail_ms_per_member"]) / 1000.0 * derate

    # --- construction: cache once per block, then every member is cheap ---
    n_members = len(blocks) * scales * m_selection + m_final
    construction = {
        "target_blocks": list(blocks), "scales": scales,
        "selection_members": len(blocks) * scales * m_selection,
        "final_members": m_final, "total_members": n_members,
        "cache_passes": len(blocks),
        "cache_s_per_block": cache_s,
        "cache_total_s": len(blocks) * cache_s,
        "member_s_each": member_s,
        "member_total_s": n_members * member_s,
        "cpu_solve_total_s": n_members * solve_s,
        "gpu_accum_total_s": n_members * gpu_s,
        "naive_member_s_each": naive_s,
        "naive_total_s": n_members * naive_s,
    }
    construction["total_s"] = construction["cache_total_s"] + construction["member_total_s"]
    construction["speedup_vs_naive"] = construction["naive_total_s"] / construction["total_s"]

    # --- selection: score every (block, scale) on ID-val only, at M=10 ---
    selection = {"id_val_images": SELECTION_ID_VAL, "M": m_selection,
                 "per_block_s": {b: scales * eval_s(b, SELECTION_ID_VAL, m_selection)
                                 for b in blocks}}
    selection["total_s"] = sum(selection["per_block_s"].values())

    # --- final evaluation at M=20 ---
    id_eval = {b: eval_s(b, IMAGENET_VAL, m_final) for b in blocks}
    ood_per_ds = {b: {k: eval_s(b, v, m_final) for k, v in OPENOOD.items()} for b in blocks}
    ood_total = {b: sum(v.values()) for b, v in ood_per_ds.items()}
    per_10k = {b: eval_s(b, 10000, m_final) for b in blocks}

    final_block = blocks[0]
    scenarios = {
        "final_block_only": {
            "description": f"target block {final_block} only (CLS-only tail)",
            "construction_s": cache_s + n_members * member_s,
            "selection_s": selection["per_block_s"][final_block],
            "id_eval_s": id_eval[final_block],
            "ood_eval_s": ood_total[final_block],
        },
        "all_three_blocks": {
            "description": "construct and evaluate all three target blocks",
            "construction_s": construction["total_s"],
            "selection_s": selection["total_s"],
            "id_eval_s": sum(id_eval.values()),
            "ood_eval_s": sum(ood_total.values()),
        },
    }
    for s in scenarios.values():
        s["total_s"] = sum(v for k, v in s.items() if k.endswith("_s"))
        s["total_h"] = s["total_s"] / 3600
        s["total_h_with_20pct_margin"] = s["total_h"] * 1.2

    return {
        "primitives": {
            "safe_batch": mem["batch_sweep"]["safe_batch"],
            "cold_images_per_s": cold,
            "sustained_images_per_s": sustained,
            "thermal_derate_factor": derate,
            "cache_h_z_s_per_block": cache_s,
            "member_construction_s_cached": member_s,
            "member_gpu_accum_s": gpu_s,
            "member_cpu_solve_s": solve_s,
            "member_construction_s_naive": naive_s,
            "cached_vs_naive_speedup": naive_s / member_s,
        },
        "per_block_costs": per_block,
        "construction": construction,
        "selection": selection,
        "id_evaluation_s": id_eval,
        "ood_evaluation": {"per_dataset_s": ood_per_ds, "total_s": ood_total,
                           "per_10k_examples_s": per_10k,
                           "total_examples": sum(OPENOOD.values())},
        "disk_io": {"compact_state_mib": ens["storage"]["disk_npz_mib"],
                    "note": "written once per member set; negligible against compute"},
        "scenarios": scenarios,
    }


def main():
    p = build()
    (OUT / "raw" / "runtime_projection.json").write_text(json.dumps(p, indent=2))
    pr = p["primitives"]
    print("== measured primitives ==")
    for k, v in pr.items():
        print(f"  {k:<34} {v:.4g}" if isinstance(v, float) else f"  {k:<34} {v}")
    print("\n== per-block cost at M=20 ==")
    for b, d in p["per_block_costs"].items():
        print(f"  block {b}: prefix {d['prefix_ms']:.2f} ms/img | tail "
              f"{d['tail_ms_per_member']:.3f} ms/img/member | {d['images_per_s_M20']:.1f} img/s "
              f"| peak {d['peak_gib']:.3f} GiB | CLS-only tail: {d['cls_only_tail']}")
    c = p["construction"]
    print(f"\n== construction ({c['total_members']} members over {len(c['target_blocks'])} "
          f"blocks x {c['scales']} scales) ==")
    print(f"  (h,z) cache  {c['cache_passes']} x {c['cache_s_per_block']:.0f}s = "
          f"{c['cache_total_s']/60:.1f} min")
    print(f"  members      {c['total_members']} x {c['member_s_each']:.2f}s = "
          f"{c['member_total_s']:.0f}s")
    print(f"  TOTAL        {c['total_s']/60:.1f} min   (naive prefix-per-member: "
          f"{c['naive_total_s']/3600:.1f} h -> {c['speedup_vs_naive']:.0f}x saving)")
    print("\n== evaluation ==")
    for b in p["id_evaluation_s"]:
        print(f"  block {b}: ID 50k {p['id_evaluation_s'][b]/60:.1f} min | "
              f"OOD suite {p['ood_evaluation']['total_s'][b]/60:.1f} min | "
              f"per 10k OOD {p['ood_evaluation']['per_10k_examples_s'][b]/60:.1f} min")
    print("\n== scenarios ==")
    for name, s in p["scenarios"].items():
        print(f"  {name:<20} {s['total_h']:.2f} h  "
              f"({s['total_h_with_20pct_margin']:.2f} h with 20% margin)")
        print(f"    construction {s['construction_s']/60:6.1f} min | selection "
              f"{s['selection_s']/60:6.1f} min | ID {s['id_eval_s']/60:6.1f} min | OOD "
              f"{s['ood_eval_s']/60:6.1f} min")


if __name__ == "__main__":
    main()
