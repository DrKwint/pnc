#!/usr/bin/env python3
"""Phase 7: assemble the CIFAR efficiency table from cached + matched-M timings."""
import json, glob, csv
from pathlib import Path

OUT = Path(__file__).resolve().parent
srcs = sorted(glob.glob("results/cifar10/inference_cost/*.json")) + \
       [str(OUT / "inference_cost_deep_ensemble_n50_matched.json")]
rows = []
for f in srcs:
    d = json.load(open(f))
    for name, m in d.get("methods", {}).items():
        rows.append(dict(method=name, ms_per_sample=round(m["warm_per_sample_ms"], 3),
                         throughput_sps=round(1000.0 / m["warm_per_sample_ms"], 1),
                         n_forward_passes=m["n_forward_passes"],
                         train_cost_factor=m.get("train_cost_factor"),
                         batch_size=d.get("batch_size"), n_bench=d.get("n_bench_samples")))
# order: single-pass, then multi-pass, DE variants explicit
order = ["PreAct ResNet-18 / MSP / Energy", "ReAct+Energy", "Mahalanobis",
         "Epinet n=50", "MC Dropout n=32", "SWAG n=50", "LLLA n=50",
         "PnC single-block scale=25", "PnC multi-block scale=7",
         "Deep Ensemble n=5", "Deep Ensemble n=50 (matched-M)"]
rows.sort(key=lambda r: order.index(r["method"]) if r["method"] in order else 99)

with open(OUT / "efficiency_cifar_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def g(name):
    return next((r for r in rows if r["method"] == name), None)
pnc = g("PnC single-block scale=25"); de5 = g("Deep Ensemble n=5")
de50 = g("Deep Ensemble n=50 (matched-M)"); one = g("PreAct ResNet-18 / MSP / Energy")

L = []
L.append("# Phase 7 — CIFAR-10 Inference Efficiency (verified)\n")
L.append("Methodology (from `scripts/benchmark_inference_cost.py`, seed 0): warm per-sample latency over "
         "N=5000 test images, batch_size=256, GPU-synced with `block_until_ready`, first (cold/JIT) batch "
         "excluded. Member counts VERIFIED against the submitted methods: P&C M=50, MC Dropout n=32, "
         "Deep Ensemble n=5, SWAG/LLLA/Epinet n=50.\n")
L.append("| Method | ms/sample | throughput (samp/s) | fwd passes | train cost |")
L.append("|---|---|---|---|---|")
for r in rows:
    tc = f"{r['train_cost_factor']:g}×" if r['train_cost_factor'] is not None else "—"
    L.append(f"| {r['method']} | {r['ms_per_sample']:.3f} | {r['throughput_sps']:.0f} | "
             f"{r['n_forward_passes']} | {tc} |")
L.append("")
L.append("## Two Deep-Ensemble comparisons (kept distinct, per task)\n")
L.append(f"1. **Submitted-cost:** Deep Ensemble **n=5** — {de5['ms_per_sample']:.3f} ms/sample, 5 fwd, **5× training**.\n")
L.append(f"2. **Matched-M:** Deep Ensemble **n=50** — {de50['ms_per_sample']:.3f} ms/sample, 50 fwd, **50× training**.\n")
L.append("## Honest reading (P&C is NOT cheaper at inference)\n")
L.append(f"- At **matched member count**, P&C single ({pnc['ms_per_sample']:.2f} ms, 50 fwd) and a 50-member Deep "
         f"Ensemble ({de50['ms_per_sample']:.2f} ms, 50 fwd) have **essentially identical inference latency** — as "
         f"expected, since both do 50 forward passes. P&C buys nothing at inference over an equal-member ensemble.\n")
L.append(f"- Versus the **submitted** Deep Ensemble (n=5, {de5['ms_per_sample']:.2f} ms), P&C is "
         f"**~{pnc['ms_per_sample']/de5['ms_per_sample']:.1f}× SLOWER** at inference (50 vs 5 forward passes), while "
         f"being **5× cheaper to train** (1× vs 5×). Against the matched-M ensemble the training gap is 50× (1× vs 50×).\n")
L.append(f"- P&C's genuine efficiency advantages are **training cost (1×)** and **model storage** (one base network "
         f"plus 50 small conv2 corrections, vs 50 full network copies for the matched-M ensemble) — NOT inference "
         f"latency. Single forward pass reference: {one['ms_per_sample']:.3f} ms.\n")
L.append("Source rows: `efficiency_cifar_table.csv`; matched-M measurement: "
         "`inference_cost_deep_ensemble_n50_matched.json`; cached submitted timings: "
         "`results/cifar10/inference_cost/*.json`. Peak GPU memory was not captured in the cached run "
         "(methodology times latency only); the storage argument above is structural, not a measured peak.")

(OUT / "efficiency_cifar_table.md").write_text("\n".join(L))
print("wrote efficiency_cifar_table.{csv,md}")
