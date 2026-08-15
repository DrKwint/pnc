"""Spec §23 — the main diagnostic table.

Assembled from whatever metric files exist. Methods that were not run appear as NA rows
rather than being dropped, so the table never silently hides a missing result.
"""
from __future__ import annotations

import csv
import json

from . import fu_common as F

NA = "NA"


def _load(name):
    p = F.OUT / "metrics" / name
    return json.loads(p.read_text()) if p.exists() else None


def build() -> list[dict]:
    base = _load("base_entropy_control.json")
    err = _load("id_error_detection.json")
    ks = _load("ksweep_final.json")
    llla = _load("llla_temperature.json")
    scod = _load("scod_results.json")
    prev = json.loads((F.PREV / "metrics" / "llla_results.json").read_text())

    eauroc = {r["score"]: r.get("auroc") for r in (err["rows"] if err else [])}
    rows = []

    def add(label, near, far, nfpr, ffpr, id_err, note=""):
        rows.append({"method": label,
                     "id_error_auroc": id_err, "near_auroc": near, "far_auroc": far,
                     "near_fpr95": nfpr, "far_fpr95": ffpr, "note": note})

    def from_base(key, label, err_key):
        if not base or key not in base["scores"]:
            return add(label, None, None, None, None, None, "NOT_RUN")
        m = base["scores"][key]
        add(label, m["near_auroc"], m["far_auroc"], m["near_fpr95"], m["far_fpr95"],
            eauroc.get(err_key))

    from_base("msp", "MSP", "MSP")
    from_base("base_entropy_raw", "Base entropy (raw)", "Base entropy (raw)")
    from_base("base_entropy_T", "Base entropy (T=0.7)", "Base entropy (T=0.7)")
    from_base("M1_true_label", "Mahalanobis", "Mahalanobis")
    from_base("M3_unconditional", "Mahalanobis (unconditional)",
              "Mahalanobis (unconditional)")
    from_base("expected_member_entropy", "P&C expected member entropy",
              "P&C expected member entropy")
    from_base("predictive_entropy", "P&C predictive entropy",
              "P&C predictive entropy")
    from_base("mutual_information", "P&C mutual information",
              "P&C mutual information")

    for K in (5, 20, 40, 80):
        a = (ks or {}).get("K", {}).get(str(K))
        if not a or "predictive_entropy_near_auroc" not in a:
            add(f"P&C K={K}", None, None, None, None, None, "NOT_RUN")
            continue
        add(f"P&C K={K}", a["predictive_entropy_near_auroc"],
            a["predictive_entropy_far_auroc"], a["predictive_entropy_near_fpr95"],
            a["predictive_entropy_far_fpr95"],
            eauroc.get(f"P&C K={K} predictive entropy"),
            f"r={a['r_target']}, lam={a['lambda']:g}, realized r="
            f"{a['realized_r_median']:.3f}, dAcc={a['delta_top1_pp']:+.3f} pp")

    if llla and "LLLA-Kron" in llla["variants"]:
        for v in ("LLLA-Kron", "LLLA-Kron+Temp"):
            m = llla["variants"].get(v)
            if m:
                add(v, m["near_auroc"], m["far_auroc"], m["near_fpr95"],
                    m["far_fpr95"], eauroc.get(f"{v} entropy"),
                    f"prior={m['prior_precision']:g}, T={m['temperature']:.4f}")
            else:
                add(v, None, None, None, None, None, "NOT_RUN")
    else:
        a = prev["LLLA-Kron"]["aggregate"]
        add("LLLA-Kron", a["near"]["mean_auroc"], a["far"]["mean_auroc"],
            a["near"]["mean_fpr95"], a["far"]["mean_fpr95"],
            eauroc.get("LLLA-Kron entropy"), "from previous round")
        add("LLLA-Kron+Temp", None, None, None, None, None, "NOT_RUN")

    for s in ("linear", "ffn", "last_block"):
        key = f"SCOD-{s}"
        m = (scod or {}).get("scopes", {}).get(key)
        label = f"SCOD-{s.replace('_', '-')}"
        if not m:
            add(label, None, None, None, None, None, "NOT_RUN")
            continue
        add(label, m["near_auroc"], m["far_auroc"], m["near_fpr95"], m["far_fpr95"],
            m["id_error_auroc"],
            f"k={m['actual_k']}, T={m['actual_T']}, q={m['q']}, N={m['n_cal']}"
            + (f"; fallback: {m['fallback_reason']}" if m["fallback_reason"] else ""))
    return rows


def run():
    rows = build()
    cols = ["method", "id_error_auroc", "near_auroc", "far_auroc", "near_fpr95",
            "far_fpr95", "note"]
    (F.OUT / "tables").mkdir(parents=True, exist_ok=True)
    with (F.OUT / "tables" / "geometry_followup.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r[c] is None else r[c]) for c in cols})

    def f(v):
        return NA if v is None else f"{v*100:.2f}"

    lines = ["| Method / score | ID error AUROC | Near AUROC | Far AUROC | "
             "Near FPR95 | Far FPR95 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {f(r['id_error_auroc'])} | "
                     f"{f(r['near_auroc'])} | {f(r['far_auroc'])} | "
                     f"{f(r['near_fpr95'])} | {f(r['far_fpr95'])} |")
    lines.append("")
    lines.append("All values are percentages. `NA` marks a method that was not run in this "
                 "round; see the report for why. ID error AUROC treats a wrong base-model "
                 "top-1 on the 50k validation set as the positive class.")
    notes = [r for r in rows if r["note"] and r["note"] != "NOT_RUN"]
    if notes:
        lines += ["", "| Method | configuration |", "|---|---|"]
        lines += [f"| {r['method']} | {r['note']} |" for r in notes]
    (F.OUT / "tables" / "geometry_followup.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {F.OUT/'tables'/'geometry_followup.md'}")


if __name__ == "__main__":
    run()
