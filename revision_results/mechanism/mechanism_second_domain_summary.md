# Priority 3 — distance–disagreement mechanism beyond Ant-v5

Replicates the submitted Ant-v5 diagnostic (hidden-space regularized Mahalanobis distance to
the calibration distribution vs P&C ensemble disagreement = sqrt of total predictive variance)
on **HalfCheetah-v5 and Hopper-v5**, using the canonical P&C config (multi-block, random,
bf=0.1, k=20, M=50; size selected by ID val-NLL). Hidden layer = the P&C calibration layer
(layer_idx=2). n = 40,000 points (10k each: ID/Near/Mid/Far). Source:
`mechanism_<env>_seed0.{csv,json}`.

## Result — the mechanism generalizes to a second (and third) domain

| env | size | pooled Spearman ρ | ρ(ID) | ρ(Near) | ρ(Mid) | ρ(Far) | regime-controlled OLS slope (disagreement ~ log10 d) |
|---|:---:|---:|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 5.0 | 0.904 | 0.337 | 0.837 | 0.860 | 0.816 | 1.968±0.008 |
| Hopper-v5 | 5.0 | 0.749 | 0.513 | 0.691 | 0.730 | 0.776 | 0.324±0.002 |

(All pooled p ≈ 0 at n=40,000.)

## Reading
- **HalfCheetah**: strong monotone distance→disagreement link — pooled ρ=0.90, within-regime
  0.82–0.86 on the OOD tiers, and a highly significant regime-controlled log-distance slope of
  1.97 (the effect is not just a regime label — disagreement rises with distance *within* each
  regime). This is a clean second-domain confirmation of the paper's Ant-v5 mechanism.
- **Hopper**: the mechanism is present but weaker — pooled ρ=0.75, all within-regime
  correlations positive and increasing with shift severity (ID 0.51 < Near 0.69 < Mid 0.73 <
  Far 0.78), regime-controlled slope 0.32 (positive, significant). Consistent with the
  manuscript describing Hopper as "the most delicate environment."
- **ID within-regime ρ is lowest on both** (0.34 HC / 0.51 Hopper): ID points cluster near the
  calibration center, so there is little distance dynamic range to correlate — exactly the
  expected behavior, and the reason pooled/across-regime correlation is the headline number.

## Relation to the submitted Ant-v5 diagnostic
This uses the same distance measure and disagreement definition as the submitted Ant-v5
Figure (plot_ant_bridge_q123 Panel B: hidden Mahalanobis vs sqrt(pred_var)), so the three
environments are directly comparable. The CIFAR analogue (the plan's preferred cross-domain
test) is prepared for the separate CIFAR machine (see cifar_other_machine/).
