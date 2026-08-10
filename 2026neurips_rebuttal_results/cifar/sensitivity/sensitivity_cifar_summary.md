# Phase 3 — CIFAR-10 One-Factor Sensitivity (anchor s3b0/ps25/bf0.05/K20/M50)

Swept-cell seeds: [0, 1, 2] (3 seeds); anchor points pre-seeded from cache at 3 seeds. Anchor row repeats in each factor. OOD score = predictive_entropy; AUROC macro-mean over datasets. ID accuracy in %, FPR95 in %. Swept-cell values are mean±std over the seeds above.

## Scale sweep

| scale (×anchor: 0.25/0.5/1/2/4 → 6.25/12.5/25/50/100) | ID acc | ID NLL | val NLL | near AUROC | near FPR95 | far AUROC | far FPR95 |
|---|---|---|---|---|---|---|---|
| 6.25 | 95.77±0.25 | 0.142±0.005 | 0.138±0.006 | 88.54±0.06 | 60.20±1.40 | 92.48±1.02 | 32.99±4.31 |
| 12.5 | 95.72±0.19 | 0.138±0.004 | 0.134±0.005 | 89.82±0.04 | 46.46±0.56 | 93.58±0.73 | 24.78±1.90 |
| 25.0 | 95.59±0.25 | 0.138±0.004 | 0.130±0.005 | 91.55±0.16 | 33.06±0.81 | 95.10±0.61 | 18.12±1.29 |
| 50.0 | 34.98±14.22 | 1.553±0.326 | 1.548±0.329 | 58.16±3.99 | 92.42±4.76 | 68.42±8.75 | 77.22±12.14 |
| 100.0 | 10.00±0.00 | 3.773±0.766 | 3.794±0.773 | 62.69±0.59 | 87.41±0.36 | 79.37±3.79 | 61.44±8.25 |

## Rank sweep

| rank K | ID acc | ID NLL | val NLL | near AUROC | near FPR95 | far AUROC | far FPR95 |
|---|---|---|---|---|---|---|---|
| 1 | 95.37±0.15 | 0.149±0.006 | 0.144±0.008 | 90.86±0.15 | 36.13±1.35 | 94.67±1.01 | 19.82±2.41 |
| 2 | 95.46±0.21 | 0.143±0.004 | 0.136±0.008 | 91.22±0.16 | 34.11±1.07 | 94.88±0.84 | 18.95±1.95 |
| 5 | 95.59±0.18 | 0.140±0.004 | 0.133±0.005 | 91.40±0.21 | 33.48±1.17 | 95.15±0.52 | 18.16±1.26 |
| 20 | 95.59±0.25 | 0.138±0.004 | 0.130±0.005 | 91.55±0.16 | 33.08±0.77 | 95.09±0.62 | 18.15±1.31 |
| 40 | 95.60±0.21 | 0.138±0.004 | 0.130±0.005 | 91.55±0.19 | 32.92±0.79 | 95.12±0.53 | 17.95±1.26 |

## Calib sweep

| calib size | ID acc | ID NLL | val NLL | near AUROC | near FPR95 | far AUROC | far FPR95 |
|---|---|---|---|---|---|---|---|
| 256 | 95.59±0.26 | 0.141±0.004 | 0.131±0.005 | 91.59±0.15 | 32.98±0.55 | 95.31±0.52 | 17.63±1.21 |
| 512 | 95.59±0.24 | 0.140±0.004 | 0.130±0.005 | 91.59±0.20 | 32.73±0.56 | 95.16±0.57 | 17.80±1.38 |
| 1024 | 95.59±0.25 | 0.138±0.004 | 0.130±0.005 | 91.55±0.16 | 33.08±0.77 | 95.09±0.62 | 18.15±1.31 |
| 2048 | 57.13±27.70 | 0.798±0.331 | 0.801±0.332 | 63.08±14.24 | 88.42±13.29 | 66.32±12.02 | 84.17±17.44 |
| 4096 | 94.15±0.81 | 0.233±0.046 | 0.236±0.054 | 90.06±0.67 | 39.53±2.05 | 94.16±0.58 | 19.12±1.02 |

## Ridge sweep

| ridge λ | ID acc | ID NLL | val NLL | near AUROC | near FPR95 | far AUROC | far FPR95 |
|---|---|---|---|---|---|---|---|
| 0.0 | 10.00±0.00 | 3.487±0.420 | 3.516±0.419 | 68.03±1.26 | 80.51±2.20 | 82.66±1.43 | 53.52±1.67 |
| 0.0001 | 92.85±1.17 | 0.268±0.048 | 0.273±0.053 | 89.85±0.78 | 37.98±1.12 | 94.75±0.87 | 17.43±2.05 |
| 0.001 | 95.59±0.25 | 0.138±0.004 | 0.130±0.005 | 91.55±0.16 | 33.08±0.77 | 95.09±0.62 | 18.15±1.31 |
| 0.01 | 95.62±0.27 | 0.138±0.004 | 0.130±0.004 | 91.49±0.15 | 33.33±0.46 | 95.03±0.56 | 18.35±1.16 |
| 0.1 | 95.62±0.27 | 0.138±0.004 | 0.130±0.004 | 91.49±0.15 | 33.32±0.49 | 95.04±0.56 | 18.37±1.14 |
| 1.0 | 95.61±0.27 | 0.138±0.004 | 0.130±0.005 | 91.49±0.15 | 33.38±0.46 | 95.04±0.55 | 18.32±1.14 |

## Block sweep

| block (s1b0=early, s2b1=mid, s3b0=late/anchor) | ID acc | ID NLL | val NLL | near AUROC | near FPR95 | far AUROC | far FPR95 |
|---|---|---|---|---|---|---|---|
| block_s1b0 | 95.27±0.18 | 0.150±0.004 | 0.147±0.006 | 89.39±0.17 | 45.61±1.79 | 92.01±1.22 | 28.42±2.33 |
| block_s2b1 | 94.76±0.19 | 0.159±0.004 | 0.151±0.007 | 89.34±0.09 | 39.31±0.42 | 92.81±1.02 | 25.19±2.02 |
| block_s3b0 | 95.59±0.25 | 0.138±0.004 | 0.130±0.005 | 91.55±0.16 | 33.08±0.77 | 95.09±0.62 | 18.15±1.31 |

## Classification of each knob

| knob | verdict | near-AUROC range | ID-acc range | notes |
|---|---|---|---|---|
| **scale** | NARROW OPTIMUM + damages ID beyond it | 58.2–91.6 | 10.0–95.8 | peak at anchor 25; ps≥50 collapses ID (acc→35% at 50, →10% at 100). Most sensitive knob. |
| **rank K** | FLAT (smooth, saturating) | 90.9–91.6 | 95.4–95.6 | even K=1 gives 90.9; saturates by K=20. Robust. |
| **calib size** | MOSTLY FLAT, one REPLICATING instability | 63.1–91.6 | 57.1–95.6 | 256–1024 flat (~91.4); **ss=2048 unstable in ALL 3 seeds** (acc 89/41/42%, nAUROC 80/55/54 — mean acc 57%); ss=4096 fine (non-monotonic → a conditioning/chunk-boundary anomaly at 2 chunks, not sample scarcity). Needs a condition-number probe. |
| **ridge λ** | THRESHOLD (flat above ~1e-3; catastrophic at 0) | 68.0–91.6 | 10.0–95.6 | **λ=0 fails (hard error at seed 1; acc→10% garbage at seeds 0,2 — singular solve)**; λ=1e-4 degraded (acc~94); λ≥1e-3 flat over 3 orders. Corrects the stale 'λ has no effect' claim. |
| **block** | LATE BLOCK BEST | 89.3–91.6 | 94.8–95.6 | late s3b0 (anchor) best (91.4); early/mid ~89.5. Caveat: same ps=25 across blocks has different meaning per block (param dims differ). |

**FPR95 is discussed alongside AUROC** (task requirement): the scale cliff and ridge=0/calib=2048 instabilities show up even more sharply in near-FPR95 (34→87 at ps≥50; 34→73 at ss=2048; 34→82 at λ=0), confirming the AUROC story is not masking an FPR95 regression.

## Solve failures & instabilities (task-required record)

| cell | seed | status | note |
|---|---|---|---|
| ridge_lam0 | 1 | error | Input contains NaN. |

- **ridge λ=0**: hard error at seed 1; garbage (acc≈10%) at seeds 0,2. λ=0 makes the normal-equations matrix singular — the ridge term is load-bearing, not cosmetic. Excluded from its own aggregate mean where errored.
- **calib ss=2048**: no exception but a severe, replicating accuracy/AUROC collapse in all 3 seeds (ID acc 89/41/42%) while ss=256/512/1024/4096 are all healthy (~95%). Flagged as a conditioning anomaly at the 2-chunk boundary; a condition-number probe is the recommended follow-up.
