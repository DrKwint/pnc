# Figure 3(C) regeneration audit

Panel (C) of Figure 3, "Probes predict disagreement" (Ant-v5), was re-rendered so its
y-axis reads **P&C disagreement _D(x)_** instead of the submitted **Finite P&C
disagreement _D(x)_**. Nothing else about the panel changed: the plotted data, the
statistics and the drawing are reproduced from the canonical source.

## 1. What Figure 3 actually is

Figure 3 in the submitted manuscript is a **single three-panel composite**:

```
pnc_repro/figures/pnc_bridge_Ant-v5.pdf      (516.65 x 202.16 pt, vector)
  (A) Correction frontier
  (B) Disagreement vs distance
  (C) Probes predict disagreement      <- the panel regenerated here
```

Identified by extracting the text layer of that PDF and matching it against the text
layer of the submitted manuscript PDF: the page-7 figure block and the Figure 3 caption
("Mechanism-level diagnostics for P&C on Ant-v5") reproduce this file panel for panel,
label for label, including `Finite P&C disagreement D(x)` and `rho_s = +0.73`,
`beta_1 = +0.83±0.01`, `t = +117.7`.

`figures/ant_bridge_q123.pdf` (from `scripts/plot_ant_bridge_q123.py`) is a **different**
figure — its panel C is "(C) Random beats Low on Far AUROC" — and is not Figure 3.

## 2. Canonical data source

```
pnc_repro/artifacts/panel_c_diagnostic_Ant-v5_seed0.csv    4000 rows
```

Confirmed canonical, not merely similar: recomputing the panel statistics from this CSV
reproduces all three printed annotations exactly (§5). Supporting artifacts:

| file | role |
|---|---|
| `pnc_repro/artifacts/pnc_bridge_panel_c_Ant-v5_seed0_lreg0.1_bf0.3_ps32_J16_epsfrac0.01.npz` | same `s_hat` values, per tier (x-axis only) |
| `pnc_repro/figures/notes.txt` | prose definition of `s_hat`, J, eps, and of the rho/beta_1/t annotations |
| `pnc_repro/figures/pnc_bridge_Ant-v5.pdf` | the submitted figure; source of the recovered drawing recipe (§6) and of panels A/B (§7) |

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

All recomputed from the canonical CSV by `scripts/make_fig3c_panel.py`; full output in
`figures/fig3c_panelC_stats.json`.

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
filename and is an unrelated import shim. Rather than eyeball the panel, the recipe was
read back out of the submitted **vector** PDF (colours, opacities, line widths, type
sizes, box and axes rectangles in points) and then verified against the data:

* **Scatter** — 500 examples per tier; tab10 green/blue/purple/red
  (`#2ca02c`/`#1f77b4`/`#9467bd`/`#d62728`), markersize 1.4, alpha 0.18. The subsample is
  a single `np.random.default_rng(0)` drawn `choice(1000, 500, replace=False)` in tier
  order id, near, mid, far — recovered by matching the 500 marker centres per tier in the
  PDF back to CSV rows. Nearest-neighbour matching resolves 500/500 distinct rows for
  Near, 499 for Mid, 498 for Far and 494 for ID (the shortfall is coincident markers that
  collapse onto one row). Of those, every recovered row for Near and Far lies in the RNG
  draw, as do 496/499 for Mid and 488/494 for ID — the handful of misses are the same
  collision artifact.
* **Trend line** — 12 quantile bins on the pooled `s_hat` over all 4000 points; x =
  geometric centre of the bin edges, y = **mean** D(x) in the bin, `color="0.1"`, lw 1.5,
  with a ±SEM band (`color="0.25"`, alpha 0.18). Confirmed: the recomputed bin means
  reproduce the 12 vertices of the plotted polyline to ≤0.05%, and the geometric bin
  centres reproduce their x-positions to ≤0.16%; per-bin *medians*, means-of-s_hat and
  arithmetic bin centres all do not.
* **Axes** — log-log, autoscaled from the plotted subsample with matplotlib's default 5%
  log margins (reproduces the submitted limits to <0.1%); major-tick grid
  `#b0b0b0`, alpha 0.22, lw 0.4.
* **Type** — title 8.0 pt, axis labels 7.5 pt, ticks/legend 6.5 pt, stats box 6.2 pt in
  `0.15` grey, DejaVu Sans; labelpad 1.0, titlepad 3.0.

## 7. Outputs and how Figure 3 is assembled

`.venv/bin/python scripts/make_fig3c_panel.py` writes:

| file | what |
|---|---|
| `figures/fig3c_probes_predict_disagreement.pdf` / `.png` | **standalone panel C**, vector, on the exact footprint it occupies in Figure 3 |
| `figures/fig3_pnc_bridge_Ant-v5.pdf` / `.png` | **full Figure 3**, drop-in replacement for `pnc_repro/figures/pnc_bridge_Ant-v5.pdf` |
| `figures/fig3c_panelC_source.csv` | normalized plot input: tier, index, s_hat, D, and whether the point is in the scatter subsample |
| `figures/fig3c_panelC_stats.json` | recomputed statistics, provenance, and the 12 trend bins |

Because Figure 3 is one composite file, the full figure is rebuilt so the paper can swap a
single file. **Panels (A) and (B) are carried over from the submitted PDF as unmodified
vector content** — the page is clipped at the B/C boundary (x = 354.4 pt) and the new
panel C is drawn beside it. Nothing is painted over: the old panel C is clipped away, and
there is no raster step, no image editing and no TeX masking. Verified by rasterizing both
files at 300 dpi:

* A+B region: **0 differing pixels**, max channel difference 0.
* Panel C plot area excluding the legend and stats box: 0.37% of pixels differ, max
  channel difference 43/255 — marker antialiasing only.
* Legend: 0 differing pixels.
* Stats box: differs cosmetically. The three values are identical
  (`+0.73`, `+0.83±0.01`, `+117.7`); the submitted figure's mathtext put slightly wider
  spacing around `=` and after the sign, making its box 2.2 pt wider and 1.2 pt shorter.
  This is a matplotlib mathtext-spacing difference, not a content difference.
* No occurrence of the string "Finite" survives anywhere in the new figure's text layer.

Panels A and B **cannot** be regenerated from data. Their run is a bespoke configuration
(perturbation scales 4/8/16/32, bf 0.3, lambda 0.1, calibration size 1024) that is not
cached under `results/`: the repo has only the 5/10/20/50 grid. A sweep of all 191 cached
Ant-v5 seed-0 sidecars against the cached hidden Mahalanobis distances gets no closer to
panel B's printed rho_s = 0.97 / beta_1 = 1.03±0.01 than rho_s = 0.88 / beta_1 = 0.98, on a
y-scale 3–17x too large. Re-running would produce different numbers and would require
changing the manuscript text, so the submitted panels are preserved as-is.

## 8. Manuscript integration — open item

The revision's LaTeX source is **not in this repository**. `neurips_draft/*.tex` (and the
identical copy at `~/Desktop/2026_neurips_perturb_and_correct/`) is the April draft; it
contains no `\includegraphics` and no Figure 3. So no `\includegraphics` path could be
updated here, and no manuscript prose was touched.

To adopt the regenerated figure, point Figure 3 at `figures/fig3_pnc_bridge_Ant-v5.pdf`
(or copy it over the file the manuscript already references). The submitted
`pnc_repro/figures/pnc_bridge_Ant-v5.pdf` is deliberately **left in place unmodified** —
it is both the provenance record and the input from which panels A/B are carried over.

The composite step needs PyMuPDF, which is not in `.venv`; without it the script still
writes the standalone panel and says so. The composite committed here was produced with
PyMuPDF 1.28.2 installed outside the project environment.
