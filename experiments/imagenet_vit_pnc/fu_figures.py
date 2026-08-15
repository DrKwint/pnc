"""Spec §24 — figures for the geometry follow-up.

Analytical clarity first. Colours extend the palette already used by this experiment's
figures (`full_tables.py`): categorical slot 1 blue, slot 2 orange, plus a dark slot 3.
The three were checked for colour-vision separation before use — every pair is at least
26 (OKLab dE x100) under protan/deutan/tritan simulation. Ordinal variables (the P&C rank
K, the projection dimension, the SCOD scope) are put on an axis rather than encoded by
colour, so colour only ever carries a genuinely categorical distinction. Every series is in
a legend, and the ones with few points are direct-labelled as well.
"""
from __future__ import annotations

import json

import numpy as np

from . import fu_common as F

C1, C2, C3 = "#2a78d6", "#eb6834", "#2f2f2d"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")


def _logx_ticks(ax, ticks):
    """Explicit ticks on a log x-axis; matplotlib's minor labels otherwise collide."""
    plt = _mpl()
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def _save(fig, name):
    d = F.OUT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, facecolor=SURFACE,
                    bbox_inches="tight")
    _mpl().close(fig)
    print(f"  wrote figures/{name}.pdf")


def _load(name):
    p = F.OUT / "metrics" / name
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ 1
def fig_base_vs_pnc():
    d = _load("base_entropy_control.json")
    if not d:
        return
    plt = _mpl()
    names = [("msp", "MSP"), ("base_entropy_raw", "base entropy (raw)"),
             ("base_entropy_T", "base entropy (T=0.7)"),
             ("expected_member_entropy", "P&C expected member entropy"),
             ("predictive_entropy", "P&C predictive entropy"),
             ("mutual_information", "P&C mutual information"),
             ("M1_true_label", "Mahalanobis")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    y = np.arange(len(names))[::-1]
    near = [d["scores"][k]["near_auroc"] * 100 for k, _ in names]
    far = [d["scores"][k]["far_auroc"] * 100 for k, _ in names]
    a.barh(y + 0.19, near, 0.36, color=C1, zorder=3, label="Near OOD")
    a.barh(y - 0.19, far, 0.36, color=C2, zorder=3, label="Far OOD")
    for yy, v in zip(y + 0.19, near):
        a.text(v + 0.4, yy, f"{v:.1f}", va="center", fontsize=8, color=INK)
    for yy, v in zip(y - 0.19, far):
        a.text(v + 0.4, yy, f"{v:.1f}", va="center", fontsize=8, color=INK)
    a.set_yticks(y)
    a.set_yticklabels([n for _, n in names], fontsize=9, color=INK)
    a.set_xlim(60, 97)
    # bottom-right holds the Mahalanobis Far bar and its label; the top-right is clear
    a.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    _style(a, xlabel="AUROC (%)", title="a  Base confidence vs P&C ensemble uncertainty")

    sp = d["spearman"]
    keys = ["predictive_entropy", "expected_member_entropy", "mutual_information"]
    x = np.arange(len(keys))
    b.bar(x - 0.19, [sp["ID"]["base_entropy_T"][k] for k in keys], 0.36, color=C1,
          zorder=3, label="within ID")
    b.bar(x + 0.19, [sp["pooled_OOD"]["base_entropy_T"][k] for k in keys], 0.36,
          color=C2, zorder=3, label="within pooled OOD")
    for xx, k in zip(x, keys):
        b.text(xx - 0.19, sp["ID"]["base_entropy_T"][k] + 0.015,
               f"{sp['ID']['base_entropy_T'][k]:.3f}", ha="center", fontsize=8, color=INK)
        b.text(xx + 0.19, sp["pooled_OOD"]["base_entropy_T"][k] + 0.015,
               f"{sp['pooled_OOD']['base_entropy_T'][k]:.3f}", ha="center", fontsize=8,
               color=INK)
    b.set_xticks(x)
    b.set_xticklabels(["predictive\nentropy", "expected member\nentropy",
                       "mutual\ninformation"], fontsize=9, color=INK)
    b.set_ylim(0, 1.08)
    b.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    _style(b, ylabel="Spearman vs base entropy (T=0.7)",
           title="b  How much of P&C's ranking is the base model's?")
    _save(fig, "base_vs_pnc_entropy")


# ------------------------------------------------------------------ 2
def fig_rank_sweep():
    d = _load("ksweep_final.json")
    base = _load("base_entropy_control.json")
    if not d:
        return
    Ks = [int(k) for k in sorted(d["K"], key=int) if "top1" in d["K"][k]]
    if not Ks:
        return
    plt = _mpl()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for tag, col, lab in (("near", C1, "Near OOD"), ("far", C2, "Far OOD")):
        m = np.array([d["K"][str(k)][f"predictive_entropy_{tag}_auroc"] for k in Ks]) * 100
        s = np.array([d["K"][str(k)][f"predictive_entropy_{tag}_auroc_std"]
                      for k in Ks]) * 100
        a.errorbar(Ks, m, yerr=s, fmt="-o", color=col, lw=2, ms=6, capsize=3,
                   zorder=3, label=f"P&C predictive entropy, {lab}")
        if base:
            ref = base["scores"]["M1_true_label"][f"{tag}_auroc"] * 100
            a.axhline(ref, color=col, lw=1.1, ls="--", zorder=2)
            a.text(Ks[-1], ref + 0.25, f"Mahalanobis {lab} {ref:.1f}", ha="right",
                   fontsize=8, color=col)
    _logx_ticks(a, Ks)
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center left")
    _style(a, xlabel="perturbation rank K (nested orthonormal basis, matched realised r)",
           ylabel="AUROC (%)", title="a  Does richer subspace coverage close the gap?")

    sp_r = [d["K"][str(k)]["seed0_diagnostics"]["spearman_maha_vs_hidden_response"]
            for k in Ks]
    sp_e = [d["K"][str(k)]["seed0_diagnostics"]["spearman_maha_vs_predictive_entropy"]
            for k in Ks]
    b.plot(Ks, sp_r, "-o", color=C1, lw=2, ms=6, zorder=3,
           label="Mahalanobis vs hidden perturbation response")
    b.plot(Ks, sp_e, "-s", color=C3, lw=2, ms=6, zorder=3,
           label="Mahalanobis vs P&C predictive entropy")
    _logx_ticks(b, Ks)
    b.set_ylim(0, 1)
    b.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    _style(b, xlabel="perturbation rank K", ylabel="Spearman (ID + OOD pooled)",
           title="b  Alignment with the geometry")
    _save(fig, "pnc_rank_sweep")


# ------------------------------------------------------------------ 3
def fig_random_projection():
    d = _load("random_projection_mahalanobis.json")
    base = _load("base_entropy_control.json")
    if not d:
        return
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, tag, lab in zip(axes, ("near", "far"), ("Near OOD", "Far OOD")):
        for (variant, col, mk) in (("class_conditional", C1, "o"),
                                   ("unconditional", C2, "s")):
            rows = [r for r in d["variants"][variant] if r["K"] < 768]
            K = [r["K"] for r in rows]
            m = np.array([r[f"{tag}_auroc_mean"] for r in rows]) * 100
            s = np.array([r[f"{tag}_auroc_std"] for r in rows]) * 100
            ax.plot(K, m, f"-{mk}", color=col, lw=2, ms=5, zorder=3,
                    label=f"{variant.replace('_', '-')} Mahalanobis")
            ax.fill_between(K, m - s, m + s, color=col, alpha=0.18, lw=0, zorder=2)
            full = [r for r in d["variants"][variant] if r["K"] == 768][0]
            ax.axhline(full[f"{tag}_auroc_mean"] * 100, color=col, lw=1.0, ls=":",
                       zorder=2)
        if base:
            v = base["scores"]["predictive_entropy"][f"{tag}_auroc"] * 100
            ax.axhline(v, color=C3, lw=1.6, ls="--", zorder=4)
            ax.text(6, v + 0.6, f"P&C K=20 predictive entropy  {v:.1f}", fontsize=8.5,
                    color=C3)
        _logx_ticks(ax, [5, 20, 40, 80, 160, 320])
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
        _style(ax, xlabel="random projection dimension K",
               ylabel="AUROC (%)" if tag == "near" else None,
               title=f"{'a' if tag == 'near' else 'b'}  {lab}")
    axes[0].text(5.2, 62, "dotted: full 768-d", fontsize=8, color=MUTED)
    _save(fig, "random_projection_mahalanobis")


# ------------------------------------------------------------------ 4, 5
def fig_spectrum():
    d = _load("mahalanobis_spectrum.json")
    if not d:
        return
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(d["bands"]))
    ax.plot(x, [b["near_auroc"] * 100 for b in d["bands"]], "-o", color=C1, lw=2, ms=6,
            zorder=3, label="Near OOD")
    ax.plot(x, [b["far_auroc"] * 100 for b in d["bands"]], "-s", color=C2, lw=2, ms=6,
            zorder=3, label="Far OOD")
    ax.set_xticks(x)
    ax.set_xticklabels([b["band"] for b in d["bands"]], fontsize=8, rotation=30,
                       color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax.text(0.02, 0.96, "each band scored alone (96 modes)", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, va="top")
    _style(ax, xlabel="covariance modes, ordered by decreasing ID variance",
           ylabel="AUROC (%)",
           title="Mahalanobis power is concentrated in low-variance ID directions")
    _save(fig, "mahalanobis_spectral_auroc")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for which, col, mk in (("highest-variance", C2, "s"), ("lowest-variance", C1, "o")):
        rows = [r for r in d["cumulative"] if r["which"] == which]
        ax.plot([r["k"] for r in rows], [r["near_auroc"] * 100 for r in rows],
                f"-{mk}", color=col, lw=2, ms=6, zorder=3, label=f"{which} k modes")
    full = [r for r in d["cumulative"] if r["k"] == 768][0]["near_auroc"] * 100
    ax.axhline(full, color=C3, lw=1.3, ls="--", zorder=2)
    ax.text(100, full + 0.35, f"all 768 modes  {full:.2f}", fontsize=8.5, color=C3)
    _logx_ticks(ax, [96, 192, 384, 768])
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right")
    _style(ax, xlabel="number of modes retained", ylabel="Near AUROC (%)",
           title="The lowest-variance 96 modes alone beat the full detector")
    _save(fig, "mahalanobis_cumulative_modes")


# ------------------------------------------------------------------ 6
def fig_alignment():
    d = _load("spectral_alignment.json")
    if not d:
        return
    plt = _mpl()
    c = d["curves"]
    rank = np.arange(1, len(c["eigenvalue"]) + 1)

    def smooth(v, w=24):
        v = np.asarray(v, float)
        pad = np.pad(v, w // 2, mode="reflect")
        return np.convolve(pad, np.ones(w) / w, mode="same")[w // 2:w // 2 + len(v)]

    def share(v):
        v = np.asarray(v, float)
        return v / v.sum()

    # Two independent x-axes: panel a is per-mode (768 points), panel b is per-band (8).
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.8, 7.0),
                              gridspec_kw={"height_ratios": [1.9, 1.1], "hspace": 0.42})
    inv = share(1.0 / np.asarray(c["eigenvalue"]))
    sep = share(c["maha_separation"])
    en = share(c["pnc_energy_norm"])
    a.plot(rank, smooth(inv), color=C2, lw=2, zorder=3,
           label=r"Mahalanobis weight  $1/\lambda_j$")
    a.plot(rank, smooth(sep), color=C3, lw=2, zorder=3,
           label="Mahalanobis OOD separation of mode $j$")
    a.plot(rank, smooth(en), color=C1, lw=2.6, zorder=4,
           label="P&C perturbation-response energy $E_j$")
    a.set_yscale("log")
    a.set_xlim(1, len(rank))
    # the band between the 1/lambda curve and the other two is empty; park it there
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="center left",
             bbox_to_anchor=(0.01, 0.30))
    _style(a, xlabel="covariance mode rank (1 = highest ID variance)",
           ylabel="share of total (log scale)",
           title="a  P&C excites the modes Mahalanobis discounts")
    sp = d["spearman"]
    a.text(0.98, 0.95, f"Spearman  $E_j$ vs $\\lambda_j$ = {sp['E_vs_lambda']:+.3f}\n"
                       f"$E_j$ vs $1/\\lambda_j$ = {sp['E_vs_inv_lambda']:+.3f}\n"
                       f"$E_j$ vs separation = {sp['E_vs_maha_separation']:+.3f}",
           transform=a.transAxes, ha="right", va="top", fontsize=8.5, color=INK)

    w = 0.38
    x = np.arange(len(d["bands"]))
    b.bar(x - w / 2, [r["pnc_energy_share"] * 100 for r in d["bands"]], w, color=C1,
          zorder=3, label="P&C response-energy share")
    b.bar(x + w / 2, [r["maha_separation_share"] * 100 for r in d["bands"]], w,
          color=C3, zorder=3, label="Mahalanobis separation share")
    b.set_xticks(x)
    b.set_xticklabels([r["band"] for r in d["bands"]], fontsize=8, rotation=25,
                      color=INK, ha="right")
    b.set_xlim(-0.6, len(x) - 0.4)
    b.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    _style(b, xlabel="covariance modes, ordered by decreasing ID variance",
           ylabel="share (%)", title="b  Aggregated into bands of 96")
    _save(fig, "pnc_mahalanobis_spectral_alignment")


# ------------------------------------------------------------------ 7
def fig_scatter():
    plt = _mpl()
    rng = np.random.default_rng(0)
    idv = F.load_scores("val50k")
    ref_m = np.sort(idv["M1_true_label"])
    ref_p = np.sort(idv["predictive_entropy"])

    def pct(v, ref):
        return np.searchsorted(ref, v, side="right") / len(ref) * 100

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    for name, col, lab, al in (("val50k", C1, "ID (ImageNet val)", 0.25),
                               ("ninco", C2, "NINCO (Near OOD)", 0.45),
                               ("inaturalist", C3, "iNaturalist (Far OOD)", 0.45)):
        s = idv if name == "val50k" else F.load_scores(name)
        n = len(s["M1_true_label"])
        k = rng.choice(n, min(4000, n), replace=False)
        ax.scatter(pct(s["M1_true_label"][k], ref_m),
                   pct(s["predictive_entropy"][k], ref_p),
                   s=4, color=col, alpha=al, lw=0, zorder=3, label=lab)
    ax.axvline(90, color=MUTED, lw=1.0, ls="--", zorder=4)
    ax.axhline(90, color=MUTED, lw=1.0, ls="--", zorder=4)
    ax.axhline(60, color=MUTED, lw=1.0, ls=":", zorder=4)
    ax.axvline(60, color=MUTED, lw=1.0, ls=":", zorder=4)
    ax.add_patch(plt.Rectangle((90, 0), 12, 60, facecolor=C2, alpha=0.10, zorder=1))
    ax.add_patch(plt.Rectangle((0, 90), 60, 12, facecolor=C1, alpha=0.10, zorder=1))
    ax.text(99, 57, "geometry-only", ha="right", va="top", fontsize=8.5, color=INK)
    ax.text(3, 99, "P&C-only", fontsize=8.5, color=INK, va="top")
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 102)
    # the upper-left quadrant is the sparsest region, and it is not one of the two
    # highlighted disagreement boxes
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left",
              bbox_to_anchor=(0.02, 0.90), markerscale=3)
    _style(ax, xlabel="Mahalanobis percentile (referenced to ID)",
           ylabel="P&C predictive-entropy percentile (referenced to ID)",
           title="Where the two detectors disagree")
    _save(fig, "mahalanobis_vs_pnc_scatter")


# ------------------------------------------------------------------ 8
def fig_layerwise():
    d = _load("layerwise_mahalanobis.json")
    if not d:
        return
    plt = _mpl()
    L = ["block8", "block9", "block10", "block11", "final_ln"]
    L = [l for l in L if l in d["layers"]]
    x = np.arange(len(L))
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for tag, col, mk, lab in (("near_auroc", C1, "o", "Near OOD"),
                              ("far_auroc", C2, "s", "Far OOD")):
        v = [d["layers"][l]["class_conditional"][tag] * 100 for l in L]
        ax.plot(x, v, f"-{mk}", color=col, lw=2, ms=7, zorder=3,
                label=f"class-conditional, {lab}")
        u = [d["layers"][l]["unconditional"][tag] * 100 for l in L]
        ax.plot(x, u, f"--{mk}", color=col, lw=1.3, ms=5, alpha=0.65, zorder=3,
                label=f"unconditional, {lab}")
        for xx, vv in zip(x, v):
            ax.text(xx, vv + 0.5, f"{vv:.1f}", ha="center", fontsize=8, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(["block 8", "block 9", "block 10", "block 11",
                        "final LayerNorm"][:len(L)], fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left", ncol=1)
    _style(ax, ylabel="AUROC (%)",
           title="Layerwise Mahalanobis (all fitted on the same ID training pool)")
    _save(fig, "layerwise_mahalanobis")


# ------------------------------------------------------------------ 9
def fig_scod():
    d = _load("scod_results.json")
    if not d or not d.get("scopes"):
        return
    plt = _mpl()
    order = ["SCOD-linear", "SCOD-ffn", "SCOD-last_block"]
    have = [s for s in order if s in d["scopes"]]
    if len(have) < 1:
        return
    x = np.arange(len(have))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    # slope chart, not bars: a non-zero axis is honest for point marks and the change
    # between scopes is the quantity of interest
    for tag, col, mk, lab in (("near_auroc", C1, "o", "Near OOD"),
                              ("far_auroc", C2, "s", "Far OOD"),
                              ("id_error_auroc", C3, "^", "ID error detection")):
        v = [d["scopes"][s][tag] * 100 for s in have]
        ax.plot(x, v, f"-{mk}", color=col, lw=2, ms=9, zorder=3, label=lab)
        for xx, vv in zip(x, v):
            ax.annotate(f"{vv:.2f}", (xx, vv), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"SCOD-{s.replace('SCOD-', '').replace('_', '-')}\n"
                        f"{d['scopes'][s]['n_params']:,} params"
                        for s in have], fontsize=9.5, color=INK)
    ax.set_xlim(-0.45, len(have) - 0.55 + 0.45)
    lo = min(d["scopes"][s][t] for s in have
             for t in ("near_auroc", "far_auroc", "id_error_auroc")) * 100
    hi = max(d["scopes"][s][t] for s in have
             for t in ("near_auroc", "far_auroc", "id_error_auroc")) * 100
    ax.set_ylim(lo - 4, hi + 4)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center right")
    _style(ax, ylabel="AUROC (%)",
           title="Enlarging the SCOD parameter scope makes it worse (k=30, exact Fisher)")
    if "SCOD-last_block" not in have:
        ax.text(0.5, 0.02, "SCOD-last-block not run — see report §11",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    _save(fig, "scod_scope_comparison")


def run():
    for f in (fig_base_vs_pnc, fig_rank_sweep, fig_random_projection, fig_spectrum,
              fig_alignment, fig_scatter, fig_layerwise, fig_scod):
        try:
            f()
        except Exception as e:                     # a missing input must not kill the rest
            print(f"  SKIP {f.__name__}: {type(e).__name__}: {e}")
    print(f"\nfigures in {F.OUT/'figures'}")


if __name__ == "__main__":
    run()
