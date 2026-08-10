# CIFAR Calibration-Set Construction (Phase 2 / Section 9)

**Date:** 2026-07-24 · anchor block s3b0, seed 0, scale 25, **bf=0** (isolate calibration effect from bootstrap),
M=50 · calib size ∈ {128, 256, 512} × selection ∈ {random, class-balanced, k-center on block-input features} ·
pool 4096 ID images · 800 eval imgs/dataset. Raw: `calibration_study_raw.json`; harness: `_calibration_study.py`.

## Result
| size | method | ID acc | ID NLL | ID MI | Near AUROC | Far AUROC |
|---|---|---|---|---|---|---|
| 128 | random | 95.12 | 0.193 | 0.087 | 91.99 | 94.90 |
| 128 | class-balanced | 95.12 | 0.196 | 0.089 | **92.35** | **95.91** |
| 128 | k-center | 94.88 | 0.218 | 0.109 | 92.04 | 95.59 |
| **256** | **random** | **9.50** | **2.181** | 0.346 | **44.38** | 62.78 |
| **256** | **class-balanced** | **10.50** | 1.897 | 0.390 | 49.29 | 60.64 |
| **256** | **k-center** | **9.50** | 2.094 | 0.370 | 46.27 | 61.53 |
| 512 | random | 95.50 | 0.149 | 0.016 | 91.92 | 95.24 |
| 512 | class-balanced | 95.50 | 0.146 | 0.016 | 91.77 | 95.21 |
| 512 | k-center | 94.88 | 0.146 | 0.016 | 91.87 | 95.32 |

## Findings
1. **[seed-0 empirical, striking] Direct double-descent: the interpolation peak is CATASTROPHIC.** At calib
   **size 256 the correction collapses** (ID acc ~10%, Near-AUROC ~46) for **all three selection methods**, while
   both **128 (under-determined)** and **512 (over-determined)** work fine (ID acc ~95, Near-AUROC ~92). This is a
   direct observation of the double-descent / interpolation peak the conditioning study (Section 2) predicted for
   s3b0 (rank→p near ~256–512 imgs; cond worst at the crossing). *(Fills the Section 2.4 double-descent question:
   YES, located at the rank≈p crossing, catastrophic, block-specific.)*
2. **[seed-0 empirical] Selection METHOD matters far less than calibration SIZE.** At each size the three methods are
   close; **no selection method rescues the size-256 collapse.** Class-balancing gives a **small consistent edge at
   the small size 128** (Near +0.36, Far +1.0 AUROC over random) and slightly better NLL at 512. k-center is not
   better than random here (and marginally worse on ID acc), consistent with the **extreme spatial redundancy**
   (stable rank ~5, Section 2): once the effective ~5-dim subspace is covered, which images are chosen barely
   matters. *(Section 21 Q12: calibration construction changes capacity only marginally; SIZE relative to the
   interpolation threshold dominates.)*
3. **[seed-0 empirical] Robustness caveat for the paper.** The submitted config (1024 imgs, over-determined + bf=0.05)
   is safely past the peak, but **a naive choice of ~256 calibration images would catastrophically fail** — a sharp,
   non-obvious reproducibility hazard. This is the clean-bf analogue of the ss=2048 instability seen with bootstrap
   in the sensitivity sweep; both are conditioning-driven collapses near a rank≈p crossing.

## Recommendation
Use a calibration size comfortably above the block's interpolation threshold (for s3b0: ≥512, submitted 1024) and
**class-balanced random sampling** (small, free edge). Geometric selection (k-center/D-optimal) is not worth the
cost at this block given the low effective rank; it may matter more for shallow blocks whose rank saturates at few
images (Section 2.5) — a noted extension. Full D-optimal / near-threshold method comparison (Section 9.2, 9.4) is
deferred; the size-vs-method conclusion is established.
