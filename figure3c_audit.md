# Figure 3(C) regeneration audit

Panel (C) of Figure 3, "Probes predict disagreement" (Ant-v5), was re-rendered so its
y-axis reads **P&C disagreement _D(x)_** instead of the submitted **Finite P&C
disagreement _D(x)_**. Nothing else about the panel changed: the plotted data, the
statistics and the drawing are reproduced from the canonical source.

Figure 3 is emitted as **one flat vector PDF from a single matplotlib figure**
(`scripts/make_fig3.py`). No PDF compositing, overlaying or page-importing is
involved — every mark on the page is drawn from a data file in one pass.

## 1. What Figure 3 actually is

Figure 3 in the submitted manuscript is a **single three-panel composite**:

```
pnc_repro/figures/pnc_bridge_Ant-v5.pdf      (516.65 x 202.16 pt, vector)
  (A) Correction frontier
  (B) Disagreement vs distance
  (C) Probes predict disagreement      <- the panel whose label was wrong
```

Identified by extracting the text layer of that PDF and matching it against the text
layer of the submitted manuscript PDF: the page-7 figure block and the Figure 3 caption
("Mechanism-level diagnostics for P&C on Ant-v5") reproduce this file panel for panel,
label for label, including `Finite P&C disagreement D(x)` and `rho_s = +0.73`,
`beta_1 = +0.83±0.01`, `t = +117.7`.

`figures/ant_bridge_q123.pdf` (from `scripts/plot_ant_bridge_q123.py`) is a **different**
figure — its panel C is "(C) Random beats Low on Far AUROC" — and is not Figure 3.

## 2. Canonical data source for panel C

```
pnc_repro/artifacts/panel_c_diagnostic_Ant-v5_seed0.csv    4000 rows
```

Confirmed canonical, not merely similar: recomputing the panel statistics from this CSV
reproduces all three printed annotations exactly (§5). Supporting artifacts:

| file | role |
|---|---|
| `pnc_repro/artifacts/pnc_bridge_panel_c_Ant-v5_seed0_lreg0.1_bf0.3_ps32_J16_epsfrac0.01.npz` | same `s_hat` values, per tier (x-axis only) |
| `pnc_repro/artifacts/pnc_bridge_hidden_mahal_Ant-v5_seed0.npz` | cached hidden-space Mahalanobis distances (panel B x-axis; used as the recovery cross-check in §7) |
| `pnc_repro/figures/notes.txt` | prose definition of `s_hat`, J, eps, and of the rho/beta_1/t annotations |
| `pnc_repro/figures/pnc_bridge_Ant-v5.pdf` | the submitted figure; source of the recovered drawing recipe (§6) and of the panel A/B data (§7) |

Run configuration, from the CSV's own columns: `env=Ant-v5`, `seed=0`, `probe_eps=0.32`,
`n_probes=16`, `pnc_perturbation_scale=32.0`, `bootstrap_frac=0.3`,
`correction_lambda=0.1`, `calibration_size=1024`,
`target_layer="even-indexed (l1, l3, …); LS on next layer"`.

## 3. Definition of the y-axis, D(x)

`finite_disagreement` — the **finite-scale P&C ensemble disagreement** at the full
operating perturbation scale (ps = 32), i.e. the actual finite corrected residual, in
output L2 units. Its counterpart in the panel is the *infinitesimal* probe sketch, and
`notes.txt` describes panel C as "finite-scale ensemble disagreement D(x) (full ps) vs
the small-eps linear probe sketch s_hat(x)".

It is **not** the moment-matched `sqrt(pred_var)` summary used in the paper's *other*
mechanism panel (Figure 3B, and `scripts/neurips_2026_rebuttal/priority3_mechanism.py`),
which lives on a different scale (0–1.75 vs 37–2500 here).

The word "Finite" in the submitted axis label distinguished it from the first-order
quantity on the x-axis. The manuscript text calls it "P&C disagreement", so the label is
now "P&C disagreement D(x)"; the underlying quantity is unchanged.

## 4. Definition of the x-axis, s_hat(x)

`sensitivity_sketch` — the **random-probe sketch** from the infinitesimal
corrected-sensitivity analysis, an *average over random probes* (`notes.txt`):

```
s_hat(x) = (1 / (J * eps^2)) * sum_{j=1..J} || f_probe_j(x) - f_base(x) ||^2
```

with `J = 16` probe directions `u_j ~ N(0, I)` and `eps = 0.01 * ps = 0.32`. It is a
finite-difference estimator of `||A_S(x)||_F^2` (Theorem 4), not the analytic ridge hat
weights of Prop. 3 — consistent with
`docs/NON_EXPERIMENTAL_REBUTTAL_AUDIT.md` items 110/112. The `1/eps^2` normalization
makes it eps-invariant in the locally linear regime.

## 5. Recomputed statistics

All recomputed from the canonical CSV on every run by `scripts/make_fig3c_panel.py`; full
output in `figures/fig3c_panelC_stats.json`.

**Point counts — 4000 total, 1000 per tier: ID 1000, Near 1000, Mid 1000, Far 1000.**
All 4000 enter the statistics and the trend line; the scatter shows a 500-per-tier
subsample (§6) purely for legibility.

| quantity | recomputed | printed in submitted figure | agree? |
|---|---|---|---|
| Spearman rho_s (pooled, n=4000) | +0.73326 | +0.73 | yes |
| beta_1 | +0.82718 | +0.83 | yes |
| SE(beta_1) | 0.00703 | ±0.01 | yes |
| t = beta_1 / SE | +117.697 | +117.7 | yes |

**No discrepancy.** The submitted values are canonical and are reprinted unchanged.

Regression model, exactly as used:

```
log10 D(x) = beta_0 + beta_1 * log10 s_hat(x) + b_near + b_mid + b_far
```

* **log-log**, base 10, on the linear-scale columns. No epsilon offset is applied or
  needed (`s_hat > 0` and `D > 0` everywhere: s_hat in [140.2, 12646.2],
  D in [37.4, 2500.5]). An offset changes nothing material — eps = 1 gives
  beta_1 = 0.824, t = 117.6.
* Regime fixed effects as dummies with **ID as the dropped baseline**, so beta_1 is the
  within-tier power-law exponent, not the pooled slope.
* Ordinary least squares, homoskedastic covariance
  `sigma^2 (X'X)^-1`, dof = 4000 − 5 = 3995. **The "±0.01" is the OLS standard error**,
  not a confidence-interval half-width and not a bootstrap spread.
* `t = beta_1 / SE(beta_1)` against the null beta_1 = 0.

Within-tier Spearman (not printed on the panel, recorded for completeness): ID 0.984,
Near 0.911, Mid 0.899, Far 0.680.

## 6. Drawing recipe

The original plotting script is **gone**. It is not in this repo, not in its git history
(all branches), and a content search of the filesystem for the panel title found nothing;
`/home/elean/conditioned_disagreement/.../generators/pnc_bridge.py` matches only on
filename and is an unrelated import shim. Rather than eyeball the figure, the recipe was
read back out of the submitted **vector** PDF (colours, opacities, line widths, type
sizes, tick pads, and every box and axes rectangle in points) and then verified against
the data. Page size 516.654 x 202.163 pt and all three axes rectangles
((33.45, 208.94, 384.43) x0, each 120.21 x 134.28 pt) reproduce exactly.

Panel C specifically:

* **Scatter** — 500 examples per tier; tab10 green/blue/purple/red
  (`#2ca02c`/`#1f77b4`/`#9467bd`/`#d62728`), markersize 1.4, alpha 0.18. The subsample is
  a single `np.random.default_rng(0)` drawn `choice(1000, 500, replace=False)` in tier
  order id, near, mid, far — recovered by matching the marker centres in the PDF back to
  CSV rows (Near and Far matched the RNG draw exactly; ID and Mid matched all but the
  handful of coincident markers that collapse under nearest-neighbour matching).
* **Trend line** — 12 quantile bins on the pooled `s_hat` over all 4000 points; x =
  geometric centre of the bin edges, y = **mean** D(x) in the bin, `color="0.1"`, lw 1.5,
  with a ±SEM band (`color="0.25"`, alpha 0.18). Confirmed twice over: the recomputed bin
  means reproduce the 12 vertices of the plotted polyline to ≤0.05% in data units, and the
  regenerated polyline lands within **0.003 pt** of the submitted one on the page. Per-bin
  *medians*, means-of-s_hat and arithmetic bin centres all fail to match.
* **Axes** — log-log, autoscaled from the plotted subsample with matplotlib's default 5%
  log margins (reproduces the submitted limits to <0.1%); major-tick grid
  `#b0b0b0`, alpha 0.22, lw 0.4.
* **Type** — title 8.0 pt, axis labels 7.5 pt, ticks/legend 6.5 pt, stats box 6.2 pt in
  `0.15` grey, DejaVu Sans; labelpad 2.5, titlepad 3.0, tick pad 2.0.

## 7. Panels A and B: recovered data, not re-run and not pasted

Panels A and B **cannot be recomputed**. Their run is a bespoke configuration
(perturbation scales 4/8/16/32, bf 0.3, lambda 0.1, calibration size 1024) that is not
cached under `results/`: the repo has only the 5/10/20/50 grid. A sweep of all 191 cached
Ant-v5 seed-0 sidecars against the cached hidden Mahalanobis distances gets no closer to
panel B's printed rho_s = 0.97 / beta_1 = 1.03±0.01 than rho_s = 0.88 / beta_1 = 0.98, on a
y-scale 3–17x too large. Re-running would produce different numbers and would require
changing the manuscript text.

So that Figure 3 can still be drawn in one pass, `scripts/extract_fig3_panelAB_data.py`
recovers their data from the submitted PDF's **vector content stream** — the device
coordinates matplotlib wrote for every marker centre and polyline vertex — and inverts the
axis transform. No rasterization is involved; this is not approximation from an image.

* **Axis calibration.** Tick marks are located as the stubs on the axes spines, paired
  with their rendered tick labels (including mathtext powers of ten), and
  `position = a + b·f(value)` is fit by least squares. Fit residuals: panel A x 3e-3 pt over
  4 labelled ticks, panel A y and panel B x exact (2 ticks), panel B y 4e-5 pt over 8 ticks.
* **Independent accuracy check.** Panel B's x-values are hidden-space Mahalanobis
  distances, and those *are* cached. Matching all 2000 recovered scatter x-values to
  `pnc_bridge_hidden_mahal_Ant-v5_seed0.npz` gives a **median relative residual of 1.5e-6
  and a maximum of 6.4e-6** — the recovery is exact to ~1 part in 10^6 end to end, and the
  match also identifies which sample each plotted point is. Both the cached and recovered
  x are stored in the CSV.
* Panel B's `rho_s` / `beta_1` annotation cannot be recomputed (only the plotted subsample
  survives), so it is carried through verbatim from the submitted panel, parsed out of the
  PDF into `figures/fig3_panelAB_recovery.json` rather than hardcoded.

## 8. Outputs

`.venv/bin/python scripts/make_fig3.py` writes:

| file | what |
|---|---|
| `figures/fig3_pnc_bridge_Ant-v5.pdf` / `.png` | **Figure 3**, one flat vector export, drop-in replacement for `pnc_repro/figures/pnc_bridge_Ant-v5.pdf` |

`.venv/bin/python scripts/make_fig3c_panel.py` writes panel C on its own plus the
normalized inputs and recomputed statistics:

| file | what |
|---|---|
| `figures/fig3c_probes_predict_disagreement.pdf` / `.png` | standalone panel C, on the exact footprint it occupies in Figure 3 |
| `figures/fig3c_panelC_source.csv` | normalized plot input: tier, index, s_hat, D, in-subsample flag |
| `figures/fig3c_panelC_stats.json` | recomputed statistics, provenance, and the 12 trend bins |

`scripts/extract_fig3_panelAB_data.py` (needs PyMuPDF, not in `.venv`) writes
`figures/fig3_panelA_source.csv`, `figures/fig3_panelB_source.csv` and
`figures/fig3_panelAB_recovery.json`. It only has to be re-run if those inputs are lost;
`make_fig3.py` and `make_fig3c_panel.py` need nothing beyond the project venv.

The submitted `pnc_repro/figures/pnc_bridge_Ant-v5.pdf` is deliberately **left in place
unmodified** — it is both the provenance record and the input the panel A/B recovery reads.

## 9. Verification

**Text.** Poppler's `pdftotext` cannot be installed here (no sudo; the pip wheel needs
`libpoppler-cpp-dev`), so the check was run with PyMuPDF's `gettext`, which is the same
kind of extractor. Two caveats found and worked around: `pdfminer.six` extracts **nothing
at all** from the submitted PDF, and `gettext -mode layout` silently drops rotated text —
under either of those a "no Finite" result would be vacuous. `-mode simple` surfaces the
rotated y-labels, and the identical command run on the submitted PDF finds the old string,
so the check demonstrably discriminates:

```
python -m pymupdf gettext -mode simple <pdf>          submitted   regenerated
  "(C) Probes predict disagreement"                       1            1
  "Finite P&C disagreement"                               1            0
  "P&C disagreement"                                      1            1
  "(A) Correction frontier"                               1            1
  "(B) Disagreement vs distance"                          1            1
```

**Structure.** Both files are pure vector: 0 raster images on the page, no imported-page
XObjects, same page box, same three axes rectangles.

**Pixels**, rasterized at 300 dpi against the submitted figure:

| region | pixels differing >8/255 | >64/255 | mean abs diff |
|---|---|---|---|
| panel A | 0.55% | 0.09% | 0.22/255 |
| panel B | 0.90% | 0.46% | 0.91/255 |
| panel C | 2.15% | 1.31% | 2.77/255 |
| panel B excluding its stats box | 0.13% | — | — |
| panel C excluding y-label and stats box | 0.16% | — | — |

The residual 0.1–0.2% is antialiasing on marker and glyph edges (e.g. the "ps=" callout
crop differs by at most 13/255 and by 0% above 64/255). What is left is two things:

1. **Panel C's y-axis label** — the intended change.
2. **The stats boxes in panels B and C** — the printed values are identical
   (`+0.97`, `+1.03±0.01`; `+0.73`, `+0.83±0.01`, `+117.7`) but the submitted figure's
   mathtext put ~2.25 pt more space around `=` and after the sign, so each box is 2.25 pt
   wider there. This is a matplotlib mathtext-spacing difference between versions, not a
   content difference, and was left alone rather than faked with manual spacing macros.

## 10. Manuscript integration — open item

The revision's LaTeX source is **not in this repository**. `neurips_draft/*.tex` (and the
identical copy at `~/Desktop/2026_neurips_perturb_and_correct/`) is the April draft; it
contains no `\includegraphics` and no Figure 3. So no `\includegraphics` path could be
updated here, and no manuscript prose was touched.

To adopt the regenerated figure, point Figure 3 at `figures/fig3_pnc_bridge_Ant-v5.pdf`
(or copy it over the file the manuscript already references).
