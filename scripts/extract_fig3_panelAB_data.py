#!/usr/bin/env python3
"""Recover the panel (A) and (B) data of Figure 3 from the submitted vector PDF.

Figure 3 (``pnc_repro/figures/pnc_bridge_Ant-v5.pdf``) is a three-panel composite.
Panel C has canonical per-example data on disk
(``pnc_repro/artifacts/panel_c_diagnostic_Ant-v5_seed0.csv``), but panels A and B
do not: their run is a bespoke configuration (perturbation scales 4/8/16/32,
bootstrap_frac 0.3, lambda 0.1, calibration size 1024) that is not cached under
``results/``, and the script that drew them no longer exists anywhere on this
machine. To rebuild Figure 3 as a single fresh vector export from plotting code,
their data has to come back as *data*.

This reads it out of the PDF's **vector content stream** — the exact device
coordinates matplotlib wrote for every marker centre and polyline vertex — and
inverts the axis transform to recover data values. It is not image
approximation: no rasterization is involved and the numbers are recovered to
~1e-4 relative (see the panel-B cross-check below).

Axis calibration: tick marks are located as the short segments on the axes
spines, paired with their rendered tick labels, and ``position = a + b * f(value)``
is fit by least squares (``f = log10`` on log axes, identity on linear axes).
Residuals are reported; they run ~1e-3 pt.

Independent accuracy check: panel B's x-values are hidden-space Mahalanobis
distances, and those *are* cached, in
``pnc_repro/artifacts/pnc_bridge_hidden_mahal_Ant-v5_seed0.npz``. Every recovered
scatter x is matched to its nearest cached value; the match residual measures the
recovery error end to end, and the matched index identifies which sample each
plotted point is. The CSV stores the exact cached x alongside the recovered one.

Outputs:
  figures/fig3_panelA_source.csv   frontier points: series, ps, ID RMSE, Far NLL
  figures/fig3_panelB_source.csv   scatter + per-tier binned means + pooled trend
  figures/fig3_panelAB_recovery.json  calibration fits and recovery residuals

Run:  PYTHONPATH=<pymupdf> .venv/bin/python scripts/extract_fig3_panelAB_data.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMITTED_PDF = REPO_ROOT / "pnc_repro" / "figures" / "pnc_bridge_Ant-v5.pdf"
MAHAL_NPZ = REPO_ROOT / "pnc_repro" / "artifacts" / "pnc_bridge_hidden_mahal_Ant-v5_seed0.npz"
FIG_DIR = REPO_ROOT / "figures"

# Axes background rectangles (matplotlib draws one opaque white rect per axes).
PANEL_AXES = {
    "A": (33.45, 36.18, 153.66, 170.46),
    "B": (208.94, 36.18, 329.15, 170.46),
    "C": (384.43, 36.18, 504.64, 170.46),
}

TIER_RGB = {
    (0.173, 0.627, 0.173): "id",
    (0.122, 0.467, 0.706): "near",
    (0.580, 0.404, 0.741): "mid",
    (0.839, 0.153, 0.157): "far",
}
SERIES_RGB = {(1.000, 0.498, 0.055): "no_correction", (0.839, 0.153, 0.157): "ls_correction"}

TOL = 0.35   # pt, tolerance for "on the spine" and for segment shape tests


def rgb(dr, key):
    c = dr.get(key)
    return None if c is None else tuple(round(v, 3) for v in c)


def in_axes(r, box, pad=1.0):
    x0, y0, x1, y1 = box
    return r.x0 >= x0 - pad and r.x1 <= x1 + pad and r.y0 >= y0 - pad and r.y1 <= y1 + pad


# ── axis calibration ─────────────────────────────────────────────────────────


def _tick_positions(page, box, axis):
    """Device positions of the tick marks on one spine of one axes."""
    x0, y0, x1, y1 = box
    out = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if rgb(dr, "color") != (0.0, 0.0, 0.0):
            continue
        if axis == "x":
            # vertical stub hanging below the bottom spine
            if r.width < TOL and abs(r.height - 2.0) < TOL and abs(r.y0 - y1) < TOL \
                    and x0 - TOL <= r.x0 <= x1 + TOL:
                out.append(0.5 * (r.x0 + r.x1))
        else:
            # horizontal stub left of the left spine
            if r.height < TOL and abs(r.width - 2.0) < TOL and abs(r.x1 - x0) < TOL \
                    and y0 - TOL <= r.y0 <= y1 + TOL:
                out.append(0.5 * (r.y0 + r.y1))
    return sorted(set(round(v, 4) for v in out))


def _tick_labels(page, box, axis):
    """Rendered tick labels as (device position, value).

    Handles both plain numerals and mathtext powers of ten, which matplotlib
    emits as a '10' span followed by a smaller exponent span.
    """
    x0, y0, x1, y1 = box
    spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
             for s in l["spans"]]
    if axis == "x":
        cand = [s for s in spans if y1 + 1 < s["bbox"][1] < y1 + 14
                and x0 - 25 < s["bbox"][0] < x1 + 25]
        pos = lambda bb: 0.5 * (bb[0] + bb[2])
    else:
        cand = [s for s in spans if x0 - 30 < s["bbox"][2] < x0 - 0.5
                and y0 - 6 < s["bbox"][1] < y1 + 6]
        pos = lambda bb: 0.5 * (bb[1] + bb[3])

    # A mathtext power of ten arrives as a '10' base plus a smaller span set to
    # its right and raised. Pair them and use the combined bbox, because the
    # tick is centred on the whole label, not on the base.
    exponent_of, consumed = {}, set()
    for i, s in enumerate(cand):
        if s["text"].strip() != "10":
            continue
        for j, t in enumerate(cand):
            if j == i or j in consumed:
                continue
            if (t["size"] < s["size"] - 0.5
                    and abs(t["bbox"][0] - s["bbox"][2]) < 1.0
                    and t["bbox"][1] < s["bbox"][1]
                    and re.fullmatch(r"-?\d+", t["text"].strip())):
                exponent_of[i] = (int(t["text"].strip()), t["bbox"])
                consumed.add(j)
                break

    out = []
    for i, s in enumerate(cand):
        if i in consumed:
            continue
        txt = s["text"].strip()
        if not re.fullmatch(r"-?\d+(\.\d+)?", txt):
            continue
        bb = s["bbox"]
        if i in exponent_of:
            exp, ebb = exponent_of[i]
            value = 10.0 ** exp
            bb = (min(bb[0], ebb[0]), min(bb[1], ebb[1]),
                  max(bb[2], ebb[2]), max(bb[3], ebb[3]))
        else:
            value = float(txt)
        out.append((pos(bb), value))
    return out


def calibrate(page, box, axis, scale):
    """Fit device_position = a + b * f(value) from the labelled ticks."""
    ticks = _tick_positions(page, box, axis)
    labels = _tick_labels(page, box, axis)
    paired = []
    for lp, val in labels:
        near = min(ticks, key=lambda t: abs(t - lp)) if ticks else None
        if near is not None and abs(near - lp) < 6.0:
            paired.append((near, val))
    if len(paired) < 2:
        raise RuntimeError(f"axis {axis}: only {len(paired)} labelled ticks matched")

    f = (lambda v: np.log10(v)) if scale == "log" else (lambda v: v)
    pos = np.array([p for p, _ in paired], float)
    fv = np.array([f(v) for _, v in paired], float)
    b, a = np.polyfit(fv, pos, 1)
    resid = float(np.max(np.abs(a + b * fv - pos)))

    def to_data(p):
        z = (np.asarray(p, float) - a) / b
        return 10.0 ** z if scale == "log" else z

    return to_data, {"scale": scale, "n_labelled_ticks": len(paired),
                     "max_residual_pt": resid,
                     "values": sorted(v for _, v in paired)}


# ── panel extraction ─────────────────────────────────────────────────────────


def markers(page, box, colour, size, edge=None):
    """Centres of the filled marker glyphs of one colour and nominal size."""
    out = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if not in_axes(r, box) or rgb(dr, "fill") != colour:
            continue
        if abs(r.width - size) > TOL or abs(r.height - size) > TOL:
            continue
        if edge is not None and rgb(dr, "color") != edge:
            continue
        out.append((0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1)))
    return out


def polyline(page, box, colour, width):
    """Vertices of the one stroked polyline of a given colour and line width."""
    best = None
    for dr in page.get_drawings():
        r = dr["rect"]
        if not in_axes(r, box) or rgb(dr, "color") != colour:
            continue
        if abs((dr.get("width") or 0) - width) > 1e-6:
            continue
        pts = []
        for it in dr["items"]:
            if it[0] == "l":
                for q in (it[1], it[2]):
                    if not pts or abs(q.x - pts[-1][0]) > 1e-9 or abs(q.y - pts[-1][1]) > 1e-9:
                        pts.append((q.x, q.y))
        if pts and (best is None or len(pts) > len(best)):
            best = pts
    return best or []


def extract_panel_a(page):
    box = PANEL_AXES["A"]
    to_x, cal_x = calibrate(page, box, "x", "log")
    to_y, cal_y = calibrate(page, box, "y", "log")
    x0, _, x1, _ = box

    rows = []
    for colour, series in SERIES_RGB.items():
        verts = polyline(page, box, colour, 1.2)
        # markers include one legend swatch, which sits outside the data span
        pts = [p for p in markers(page, box, colour, 4.2) if x0 - 1 <= p[0] <= x1 + 1]
        # keep the markers that lie on the polyline, ordered along it
        ordered = []
        for v in verts:
            m = min(pts, key=lambda p: (p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2)
            if (m[0] - v[0]) ** 2 + (m[1] - v[1]) ** 2 < 1.0:
                ordered.append(m)
        if len(ordered) != len(verts):
            ordered = verts
        for scale, (px, py) in zip([4, 8, 16, 32], ordered):
            rows.append({"series": series, "perturbation_scale": scale,
                         "id_rmse": float(to_x(px)), "far_nll": float(to_y(py)),
                         "label_x": "", "label_y": ""})

    # "ps=N" callouts: recover each label's anchor (left edge, vertical centre)
    # so it can be re-placed in data coordinates instead of by eye.
    hexcol = {"no_correction": 0xFF7F0E, "ls_correction": 0xD62728}
    spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])
             for s in l["spans"] if s["text"].strip().startswith("ps=")
             and s["bbox"][0] < PANEL_AXES["B"][0]]
    for s in spans:
        series = next((k for k, v in hexcol.items() if v == s["color"]), None)
        scale = int(s["text"].strip().split("=")[1])
        for r in rows:
            if r["series"] == series and r["perturbation_scale"] == scale:
                r["label_x"] = float(to_x(s["bbox"][0]))
                r["label_y"] = float(to_y(0.5 * (s["bbox"][1] + s["bbox"][3])))
    x0, y0, x1, y1 = box
    cal_x["axis_limits"] = [float(to_x(x0)), float(to_x(x1))]
    cal_y["axis_limits"] = [float(to_y(y1)), float(to_y(y0))]
    return rows, {"x": cal_x, "y": cal_y}


def extract_panel_b(page):
    box = PANEL_AXES["B"]
    to_x, cal_x = calibrate(page, box, "x", "log")
    to_y, cal_y = calibrate(page, box, "y", "linear")
    x0, y0, x1, y1 = box

    cached = np.load(MAHAL_NPZ)
    scatter, binned, resid_rel = [], [], []
    for colour, tier in TIER_RGB.items():
        ref = np.asarray(cached[f"mahal_{tier}"], np.float64)
        for px, py in markers(page, box, colour, 1.2):
            xr, yr = float(to_x(px)), float(to_y(py))
            k = int(np.abs(ref - xr).argmin())
            resid_rel.append(abs(ref[k] - xr) / ref[k])
            scatter.append({"tier": tier, "sample_index": k,
                            "distance_mahal_cached": float(ref[k]),
                            "distance_mahal_recovered": xr,
                            "disagreement_recovered": yr})
        # per-tier binned means: larger markers with a white edge
        for px, py in markers(page, box, colour, 3.2, edge=(1.0, 1.0, 1.0)):
            if not (x0 - 1 <= px <= x1 + 1):
                continue
            binned.append({"tier": tier, "distance_mahal": float(to_x(px)),
                           "mean_disagreement": float(to_y(py))})

    verts = polyline(page, box, (0.1, 0.1, 0.1), 1.5)
    trend = [{"distance_mahal": float(to_x(px)), "mean_disagreement": float(to_y(py))}
             for px, py in verts]

    # +-SEM band: one closed grey polygon. Its vertices run along the upper
    # edge and back along the lower edge, so split it at the x turning point
    # and pair the two halves up with the trend vertices.
    band = []
    for dr in page.get_drawings():
        r = dr["rect"]
        if in_axes(r, box) and rgb(dr, "fill") == (0.25, 0.25, 0.25):
            pts = []
            for it in dr["items"]:
                if it[0] == "l":
                    for q in (it[1], it[2]):
                        if not pts or abs(q.x - pts[-1][0]) > 1e-9 or abs(q.y - pts[-1][1]) > 1e-9:
                            pts.append((q.x, q.y))
            band = pts
            break
    # A fill_between polygon visits every x twice, once on each edge, so group
    # the vertices by x and take the low/high pair. That is order-independent.
    lo_hi = []
    if band:
        by_x = {}
        for px, py in band:
            key = round(px, 3)
            by_x.setdefault(key, []).append(py)
        for px in sorted(by_x):
            ys = by_x[px]
            lo_hi.append({"distance_mahal": float(to_x(px)),
                          "lo": float(to_y(max(ys))), "hi": float(to_y(min(ys)))})

    cal_x["axis_limits"] = [float(to_x(x0)), float(to_x(x1))]
    cal_y["axis_limits"] = [float(to_y(y1)), float(to_y(y0))]
    stats = {"x": cal_x, "y": cal_y, "n_band_vertices": len(lo_hi),
             "scatter_x_vs_cached_mahal": {
                 "n": len(resid_rel),
                 "max_rel_residual": float(np.max(resid_rel)),
                 "median_rel_residual": float(np.median(resid_rel))}}
    return scatter, binned, trend, lo_hi, stats


def panel_b_annotation(page):
    """The rho_s / beta_1 values printed in panel B's stats box.

    Panel B's *full* underlying dataset is not available, so these cannot be
    recomputed; they are carried through verbatim from the submitted panel.
    """
    box = pymupdf.Rect(270.35, 150.52, 270.35 + 57.98, 150.52 + 16.23)
    txt = "".join(s["text"] for b in page.get_text("dict")["blocks"]
                  for l in b.get("lines", []) for s in l["spans"]
                  if box.contains(pymupdf.Point(s["bbox"][0] + 1, s["bbox"][1] + 1)))
    nums = re.findall(r"([+-])\s*(\d+\.\d+)(?:±(\d+\.\d+))?", txt)
    if len(nums) < 2:
        raise RuntimeError(f"could not parse panel B annotation from {txt!r}")
    (s1, rho, _), (s2, beta, se) = nums[0], nums[1]
    return {"raw_text": txt,
            "rho_s": float(s1 + rho),
            "beta_1": float(s2 + beta),
            "beta_1_se": float(se) if se else None}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    page = pymupdf.open(SUBMITTED_PDF)[0]

    a_rows, a_cal = extract_panel_a(page)
    with (FIG_DIR / "fig3_panelA_source.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "perturbation_scale", "id_rmse",
                                          "far_nll", "label_x", "label_y"])
        w.writeheader()
        w.writerows(a_rows)

    b_scatter, b_binned, b_trend, b_band, b_cal = extract_panel_b(page)
    with (FIG_DIR / "fig3_panelB_source.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "tier", "sample_index", "distance_mahal",
                    "distance_mahal_recovered", "disagreement", "sem_lo", "sem_hi"])
        for r in b_scatter:
            w.writerow(["scatter", r["tier"], r["sample_index"],
                        repr(r["distance_mahal_cached"]),
                        repr(r["distance_mahal_recovered"]),
                        repr(r["disagreement_recovered"]), "", ""])
        for r in b_binned:
            w.writerow(["tier_bin_mean", r["tier"], "", repr(r["distance_mahal"]), "",
                        repr(r["mean_disagreement"]), "", ""])
        for r in b_trend:
            w.writerow(["pooled_trend", "", "", repr(r["distance_mahal"]), "",
                        repr(r["mean_disagreement"]), "", ""])
        for r in b_band:
            w.writerow(["sem_band", "", "", repr(r["distance_mahal"]), "", "",
                        repr(r["lo"]), repr(r["hi"])])

    report = {"source_pdf": str(SUBMITTED_PDF.relative_to(REPO_ROOT)),
              "panel_B_printed_annotation": panel_b_annotation(page),
              "panel_A": {"calibration": a_cal, "n_points": len(a_rows)},
              "panel_B": {"calibration": b_cal, "n_scatter": len(b_scatter),
                          "n_tier_bin_means": len(b_binned), "n_trend_vertices": len(b_trend),
                          "n_band_vertices": len(b_band)}}
    (FIG_DIR / "fig3_panelAB_recovery.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    for r in a_rows:
        print(f"  A {r['series']:14s} ps={r['perturbation_scale']:2d} "
              f"ID RMSE={r['id_rmse']:.4f}  Far NLL={r['far_nll']:.4f}")


if __name__ == "__main__":
    main()
