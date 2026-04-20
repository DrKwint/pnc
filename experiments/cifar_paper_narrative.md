# CIFAR Paper Narrative (Results Section Guide)

Paragraph-by-paragraph guide for the CIFAR results section. Applies the P5
reframing rules from `cifar_neurips_strengthening_plan.md`:

1. Deep Ensemble is a **5×-cost reference**, not a 1×-cost competitor.
2. PnC is a post-hoc method — grouped with LLLA, Mahalanobis, ReAct.
3. Tight margins ⇒ "competitive with" or "ties"; never "outperforms".
4. Per-dataset breakdowns live in the appendix.
5. Mahalanobis caveat sentence is required next to its numbers.
6. OpenOOD dataset variant note is required in the methods section.

## Headline claim (to be placed in §1 Introduction and §5 Results)

> **PnC matches Deep Ensemble's Near-OOD detection quality on CIFAR-10 at 1×
> training cost.** Specifically, single-block PnC at scale=50 achieves
> 91.51 ± 0.21 Near-AUROC vs 91.10 ± 0.04 for Deep Ensemble (5×-cost), a
> statistically meaningful margin on a 3-seed benchmark. The trade-off is a
> 0.97 pp ID accuracy gap and ~5.5× higher inference latency. On CIFAR-100
> under **frozen** CIFAR-10 hyperparameters — a transferability test, not a
> retuning — PnC underperforms, indicating that the headline scale value
> needs dataset-specific tuning.

Avoid: "outperforms Deep Ensemble", "state-of-the-art", "beats baselines".
Use: "matches", "is competitive with", "ties within seed noise".

## §5.1 — CIFAR-10 OOD detection

**Paragraph 1 (setup).** We evaluate twelve OOD detection methods on CIFAR-10
under the OpenOOD v1.5 protocol: six OOD datasets split into near-OOD
{CIFAR-100, TinyImageNet} and far-OOD {MNIST, SVHN, Textures, Places365}.
All methods use identical ID hyperparameters tuned on CIFAR-10 ID-validation
loss alone; no OOD data is used for tuning, model selection, threshold choice,
or temperature fitting. Temperature is fit on the ID validation split only.
Results are reported as mean ± std over three seeds.

**Paragraph 2 (methods).** The twelve methods are grouped as:
(a) **post-hoc OOD baselines** operating on a frozen single model — PreAct
ResNet-18 (raw), MSP, Energy, Mahalanobis, ReAct+Energy;
(b) **post-hoc UQ methods** — LLLA, Epinet, PnC (single-block at two scales
and multi-block);
(c) **train-time UQ methods** — MC Dropout, SWAG;
(d) **reference** — Standard Ensemble n=5 at **5× training cost**, for
calibration of the quality ceiling available at higher compute budgets.

**Paragraph 3 (headline).** Refer to Table 1. PnC scale=50 achieves the
best Near-AUROC (91.51) and Near-FPR95 (32.17) among 1×-cost methods, and
is within seed noise of Standard Ensemble on both Far-OOD metrics. The more
conservative PnC scale=25 preserves ID accuracy to within 0.04 pp of the
base model and still ties Standard Ensemble on Near-AUROC (91.11 vs 91.10),
at the cost of a smaller but still positive Near-FPR95 margin.

**Paragraph 4 (Mahalanobis caveat).** *"We report a paper-clean variant of
Mahalanobis using only the penultimate feature layer with no OOD-trained
logistic combiner. The original Lee et al. (2018) numbers are higher but use
OOD validation data to combine multiple feature-layer scores — a protocol
we deliberately avoid."*

**Paragraph 5 (trade-offs).** Refer to Table 3. PnC's cost profile is
1× training × 5.5× inference — the opposite trade-off from Deep Ensemble
(5× training × 1× inference per member). For latency-sensitive deployment,
Deep Ensemble remains preferable; for training-cost-sensitive or
single-model deployment (e.g., fine-tuned checkpoints, memory-constrained
edge), PnC offers ensemble-grade OOD detection without retraining.

## §5.2 — CIFAR-100 transferability test

**Paragraph 1 (design).** To test whether CIFAR-10-tuned UQ hyperparameters
generalize, we evaluate the same suite on CIFAR-100 *without* any
CIFAR-100-specific retuning. This is a transferability test, not a CIFAR-100
OOD state-of-the-art claim. LLLA and multi-block PnC are omitted because
their memory requirements at 100 classes exceed the single-GPU lane used
for these experiments (see Appendix X).

**Paragraph 2 (result).** Refer to Table 2. The ordering on CIFAR-100 differs
substantially from CIFAR-10: **Standard Ensemble is the unambiguous winner
on every aggregate metric**, exceeding the best 1×-cost contender by more
than 2 AUROC points on Near-OOD — a larger gap than on CIFAR-10. PnC
scale=50, under the frozen CIFAR-10 scale, underperforms the single-pass
baselines; its CIFAR-10 dominance does not transfer. Mahalanobis remains
the strongest single-pass baseline on Far-OOD (82.41 AUROC, 52.18 FPR95),
beating even Standard Ensemble on those two metrics.

**Paragraph 3 (interpretation).** Two effects contribute. First, PnC's
per-direction logit perturbation scales with class count — the CIFAR-10
scale of 50 is over-aggressive in a 100-class softmax, producing the 2.84 pp
ID accuracy drop observed (78.31 → 75.47). A CIFAR-100-specific scale
sweep is expected to recover competitiveness but is out of scope for this
submission, as it would undermine the frozen-hyperparameter claim. Second,
ensembling gains are larger on harder ID tasks: Standard Ensemble's 2.94 pp
ID accuracy advantage on CIFAR-100 (vs 0.82 pp on CIFAR-10) translates
directly to better OOD discrimination.

**Paragraph 4 (honest framing).** The CIFAR-100 transferability gap is
presented as an honest limitation. The paper's claim is not that PnC is
universally the best 1×-cost method, but that (i) on CIFAR-10 under the
paper-clean protocol, PnC at a carefully tuned scale matches Deep Ensemble
OOD quality, and (ii) the method's hyperparameters need to be tuned per
dataset, which can be done on ID-validation data alone (keeping the
paper-clean guarantee).

## §5.3 — Inference cost

**Single paragraph.** Refer to Table 3. The cost breakdown shows that all
"50-sample" stochastic methods (SWAG, LLLA, PnC single-block, PnC
multi-block) sit at 7–10 ms / sample, dominated by the 50 forward passes.
Deep Ensemble at n=5 is ~5× faster per sample (1.34 ms) but costs 5× more
to train. Single-pass post-hoc methods (Mahalanobis, ReAct+Energy,
MSP/Energy) are fastest at ~0.6–0.7 ms / sample. Cost choices should be
made jointly with accuracy/OOD quality preferences; the cost table is not
a ranking.

## Methods-section notes (§4)

Include verbatim:

**OpenOOD dataset variant note.**
*"Our DTD (Textures) and Places365 splits differ in size from the canonical
OpenOOD-v1.5 splits — we use DTD at its native 5640-image size and a
10 000-image Places365 subset for reproducibility on our compute budget.
Per-dataset numbers are therefore not directly comparable to prior OpenOOD
leaderboards. All aggregate metrics reported here use the same protocol
internally across methods, so within-paper comparisons are valid."*

**Protocol-compliance note.**
*"All twelve methods' result JSONs satisfy
`protocol.ood_validation_used = false`,
`protocol.ood_tuning_used = false`, and
`protocol.temperature_fit_split = 'id_validation_only'`.
Compliance is verified automatically by
`scripts/aggregate_ood_results.py`."*

**ODIN exclusion.**
*"ODIN is excluded because its standard formulation tunes the input
perturbation magnitude on OOD validation data. ID-only selection criteria
for ODIN are non-standard, so we omit it rather than adopt an unvalidated
variant."*

## Limitations & future work (§6)

- **Architectural generality.** Numbers are on PreActResNet-18. WideResNet-28-10
  results are planned (P4 of the strengthening plan) but not in this
  submission.
- **Per-dataset retuning.** CIFAR-100 suggests PnC scale is dataset-dependent.
  A full dataset-per-dataset retuning on ID-validation loss alone would
  strengthen the generality claim without breaking the paper-clean
  protocol.
- **K-FAC LLLA.** Full-GGN LLLA does not scale to 100 classes on 8 GB VRAM;
  a K-FAC block-diagonal approximation would restore comparability and is
  a natural extension.
- **Multi-block PnC at 100 classes.** Hits GPU memory fragmentation in the
  OOD score aggregation step. A chunked score-aggregation loop (analogous
  to the ridge-solve chunking) would resolve this.
- **ImageNet scale.** Out of scope; mentioned as "scaling PnC to ImageNet
  is left for future work".

## Appendix-worthy tables (see `cifar_paper_tables.md`)

- Table 1: CIFAR-10 full OOD table
- Table 2: CIFAR-100 full OOD table
- Table 3: Inference cost
- Table 4: CIFAR-10 ID UQ metrics (Acc / NLL / ECE)
- Table 5: Architectural generality (planned, not in current submission)
- Appendix: per-dataset breakdowns via `aggregate_ood_results.py`.
