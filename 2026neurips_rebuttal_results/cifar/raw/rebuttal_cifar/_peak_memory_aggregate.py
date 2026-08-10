#!/usr/bin/env python3
"""Aggregate per-method peak-memory JSONs into a markdown + csv table.

Also cross-checks two things worth knowing before the numbers are quoted:
  1. measured latency vs the published efficiency table (harness validity)
  2. measured resident memory vs the DERIVED storage figures in sections 3.3/3.4
"""
import json
from pathlib import Path

D = Path("/home/elean/pnc/results/neurips_2026_rebuttal/cifar/peak_memory")

LABEL = {
    "single_model": "PreAct ResNet-18 (single)",
    "react": "ReAct+Energy",
    "mc_dropout": "MC Dropout n=32",
    "deep_ensemble": "Deep Ensemble n=5",
    "deep_ensemble_50": "Deep Ensemble n=50 (matched-M)",
    "swag": "SWAG n=50",
    "llla": "LLLA n=50 (Laplace)",
    "epinet": "Epinet n=50",
    "pnc_single_s3b1": "P&C single-block s3b1 (as shipped)",
    "pnc_anchor_s3b0": "P&C single-block s3b0 (SUBMITTED anchor)",
    "pnc_multi": "P&C multi-block",
}
ORDER = list(LABEL)

# published latency table, for harness validation
PUB_MS = {
    "single_model": 0.617, "react": 0.614, "mc_dropout": 5.177,
    "deep_ensemble": 1.338, "deep_ensemble_50": 7.261, "swag": 7.325,
    "llla": 7.680, "epinet": 1.617, "pnc_single_s3b1": 7.419,
    "pnc_multi": 9.688,
}
# analytic storage predictions from the report (MiB), model/posterior only
PRED_RESIDENT = {
    "single_model": 42.7, "react": 42.7, "mc_dropout": 42.7,
    "deep_ensemble": 213.3, "deep_ensemble_50": 2132.9,
    "swag": 980.3, "llla": 42.7 + 200.8, "epinet": 43.0,
    "pnc_anchor_s3b0": 42.7 + 675.1,
}

rows = []
for m in ORDER:
    f = D / f"peak_memory_{m}.json"
    if not f.exists():
        continue
    d = json.load(open(f))
    rows.append((m, d))

if not rows:
    raise SystemExit("no results yet")

print("| Method | fwd | resident after build (MiB) | build peak (MiB) | inference peak (MiB) | % of 8151 MiB card |")
print("|---|---:|---:|---:|---:|---:|")
for m, d in rows:
    pk = d["predict_peak_mib"]
    print(f"| {LABEL[m]} | {d['n_forward_passes']} | {d['resident_after_build_mib']:.1f} "
          f"| {d['build_peak_mib']:.1f} | {pk:.1f} | {100.0*pk/8151:.1f}% |")

print("\n\n### Harness validation — measured vs published latency")
print("| Method | measured ms | published ms | Δ% |")
print("|---|---:|---:|---:|")
for m, d in rows:
    if m in PUB_MS:
        me, pu = d["warm_per_sample_ms"], PUB_MS[m]
        print(f"| {LABEL[m]} | {me:.3f} | {pu:.3f} | {100.0*(me-pu)/pu:+.1f}% |")

print("\n\n### Derived storage vs measured resident")
print("| Method | analytic (MiB) | measured resident (MiB) | Δ |")
print("|---|---:|---:|---:|")
for m, d in rows:
    if m in PRED_RESIDENT:
        pr, me = PRED_RESIDENT[m], d["resident_after_build_mib"]
        print(f"| {LABEL[m]} | {pr:.1f} | {me:.1f} | {me-pr:+.1f} |")

with open(D / "peak_memory_table.csv", "w") as fh:
    fh.write("method,label,n_forward_passes,resident_after_build_mib,build_peak_mib,"
             "predict_peak_mib,warm_per_sample_ms\n")
    for m, d in rows:
        fh.write(f"{m},\"{LABEL[m]}\",{d['n_forward_passes']},"
                 f"{d['resident_after_build_mib']:.1f},{d['build_peak_mib']:.1f},"
                 f"{d['predict_peak_mib']:.1f},{d['warm_per_sample_ms']:.3f}\n")
print(f"\nwrote {D/'peak_memory_table.csv'}  ({len(rows)}/{len(ORDER)} methods)")
