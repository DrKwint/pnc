"""Turn efficiency_benchmark.py JSONs into the rebuttal's efficiency tables.

    .venv/bin/python experiments/scripts/aggregate_efficiency.py

Reads results/neurips_2026_rebuttal/efficiency_v2/*.json, writes
efficiency_v2_table.md and efficiency_v2_rows.csv alongside them.

Latency uses median_ms, not mean_ms: even with the 20-call warm-up an occasional
rep gets descheduled and skews the mean (seen in smoke runs at ~10x the median).
"""

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDIR = REPO / "results" / "neurips_2026_rebuttal" / "efficiency_v2"

LABEL = {
    "deep_ensemble": "Deep Ensemble",
    "pnc": "P&C",
    "swag": "SWAG",
    "laplace": "Laplace",
    "mc_dropout": "MC Dropout",
    "single_base": "Single net (reference)",
}
ORDER = ["deep_ensemble", "pnc", "swag", "laplace", "mc_dropout", "single_base"]


def main():
    indir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--indir" else INDIR
    files = sorted(indir.glob("efficiency_*.json"))
    if not files:
        sys.exit(f"no benchmark JSONs in {indir} -- run run_efficiency_benchmark.sh first")

    rows = []
    by_env = defaultdict(lambda: defaultdict(list))
    for f in files:
        rec = json.load(open(f))
        env, seed = rec["env"], rec["seed"]
        for meth, e in rec["methods"].items():
            c, s = e["construction"], e["storage_inference_resident"]
            lat = e.get("latency", {})
            row = {
                "env": env,
                "seed": seed,
                "steps": rec["steps"],
                "method": meth,
                "n_trained_nets": c["n_trained_nets"],
                "construction_total_s": c["total_s"],
                "base_train_s": c["base_train_s"],
                "build_s": c["build_s"],
                "storage_params": s["params"],
                "storage_mb": s["megabytes"],
                "minimal_ckpt_mb": e.get("minimal_checkpoint", {}).get("megabytes", ""),
                "device": rec["device_kind"],
            }
            for bname, b in lat.items():
                row[f"latency_{bname}_median_ms"] = b["median_ms"]
            rows.append(row)
            by_env[env][meth].append(row)

    fields = sorted({k for r in rows for k in r})
    fields = [f for f in ("env", "seed", "steps", "method") if f in fields] + [
        f for f in fields if f not in ("env", "seed", "steps", "method")
    ]
    csv_path = indir / "efficiency_v2_rows.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def agg(rs, key):
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        if not vals:
            return None, None
        return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)

    out = ["# Efficiency accounting (measured, matched protocol)", ""]
    out.append(
        "All methods share one base-training budget and one machine. P&C's "
        "construction time includes the least-squares correction solve. Latency "
        "is median over timed reps after a same-shape warm-up.\n"
    )
    for env in sorted(by_env):
        meths = by_env[env]
        any_row = next(iter(meths.values()))[0]
        n_seeds = len({r["seed"] for rs in meths.values() for r in rs})
        out.append(f"## {env}  (steps={any_row['steps']}, {n_seeds} seed(s), {any_row['device']})")
        out.append("")
        lat_keys = sorted(
            {k for rs in meths.values() for r in rs for k in r if k.startswith("latency_")}
        )
        hdr = ["method", "nets trained", "construction (s)", "of which build (s)", "storage (MB)"]
        hdr += [k.replace("latency_", "").replace("_median_ms", "") + " lat (ms)" for k in lat_keys]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "---|" * len(hdr))

        de_total, _ = agg(meths.get("deep_ensemble", []), "construction_total_s")
        for meth in ORDER:
            if meth not in meths:
                continue
            rs = meths[meth]
            tot, tsd = agg(rs, "construction_total_s")
            bld, _ = agg(rs, "build_s")
            stor, _ = agg(rs, "storage_mb")
            cell_tot = f"{tot:.1f}" + (f" ± {tsd:.1f}" if tsd else "")
            if de_total and meth != "deep_ensemble" and tot:
                ratio = de_total / tot
                cell_tot += (
                    f"  ({ratio:.0f}× cheaper)" if ratio >= 1 else f"  ({1 / ratio:.1f}× dearer)"
                )
            line = [
                LABEL.get(meth, meth),
                str(rs[0]["n_trained_nets"]),
                cell_tot,
                f"{bld:.2f}" if bld is not None else "--",
                f"{stor:.1f}" if stor is not None else "--",
            ]
            for k in lat_keys:
                m, _ = agg(rs, k)
                line.append(f"{m:.2f}" if m is not None else "--")
            out.append("| " + " | ".join(line) + " |")
        out.append("")

        pnc = meths.get("pnc", [])
        if pnc:
            mb, _ = agg(pnc, "minimal_ckpt_mb")
            stor, _ = agg(pnc, "storage_mb")
            de_stor, _ = agg(meths.get("deep_ensemble", []), "storage_mb")
            note = (
                f"P&C storage: {stor:.1f} MB resident for inference "
                f"(shared base + per-member corrected blocks + per-member dW), "
                f"vs {de_stor:.1f} MB for the Deep Ensemble."
                if de_stor
                else f"P&C storage: {stor:.1f} MB resident."
            )
            if mb:
                note += (
                    f" Minimal checkpoint is {mb:.2f} MB (base + latent coefficients; "
                    "directions regenerate from the seed), but that requires re-solving "
                    "the corrections at load time -- not what the code currently does."
                )
            out.append(note)
            out.append("")

    md_path = indir / "efficiency_v2_table.md"
    md_path.write_text("\n".join(out))
    print(f"wrote {csv_path}\nwrote {md_path}\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
