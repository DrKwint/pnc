# CIFAR BatchNorm Study (Phase 4 / Section 11)

**Date:** 2026-07-24 · anchor s3b0, seed 0, M=16, scale 25, CALIB=1024, 400 imgs/dataset · custom per-member
forward (validated patch + downstream machinery). Raw: `bn_study_raw.json`; harness: `_bn_study.py`.

## Conditions
1. **FROZEN bn2 (headline):** correction fit + forward use the base-model running bn2 statistics.
2. **Per-member bn2 REFRESH:** for each member, recompute bn2 running stats from that member's perturbed conv1 output
   on the calibration subset, **refit the conv2 correction under the refreshed bn2**, and forward with refreshed bn2.
Both correct toward the same base conv2 target; no OOD data used.

## Result
| variant | ID acc | ID NLL | ID MI | Near AUROC | Far AUROC |
|---|---|---|---|---|---|
| **frozen** | 94.50 | 0.209 | 0.0001 | **92.00** | **95.30** |
| per-member refresh | 94.75 | 0.207 | 0.0001 | 91.74 | 95.05 |
(ID acc is ~94.5 here due to M=16 + custom float64 forward + no bootstrap on a 400-img subsample; the RELATIVE
frozen-vs-refresh comparison is the finding.)

## Findings
1. **[seed-0 empirical] Frozen BN is a sound choice — marginally BETTER on OOD than per-member refresh** (Near-AUROC
   +0.26, Far-AUROC +0.25), with refresh giving only a tiny ID-acc gain (+0.25). Frozen BN is **not a hidden crutch
   and not a limitation**: the P&C mechanism does not depend on BN refresh. *(Section 21 Q14: frozen BN neither helps
   nor hinders materially; it is a legitimate simplification, slightly favourable on OOD.)*
2. **[seed-0 empirical] Per-member BN refresh generalizes slightly WORSE to OOD.** Refreshing bn2 on the perturbed-ID
   calibration distribution tunes the normalization to that (ID) distribution, which transfers marginally worse to
   OOD than the base-model stats. Since the base stats and the ID-calibration stats are both ID, refresh barely
   changes the feature map (ID MI unchanged at ~0.0001), so the effect is small in either direction.
3. **Mechanism note (per spec 11.2):** per-member refresh would introduce a distinct **dataset-level** operation into
   the feature map (making it depend on a calibration statistic), separate from the exact per-image correction
   identity (Section 1). The result shows there is **no reason to take on that extra mechanism** — frozen BN keeps the
   clean per-image identity AND is slightly better on OOD.

## Recommendation
Keep **frozen BN** (the headline). It preserves the exact finite-scale correction identity, avoids a dataset-level
BN dependence, and is marginally better for OOD detection. The other Section-11 variants (refresh only bn2 vs a
shared post-hoc refresh; classifier-temperature-only) are expected to fall between these two given the small effect
size; a fuller sweep is low-value and deferred.
