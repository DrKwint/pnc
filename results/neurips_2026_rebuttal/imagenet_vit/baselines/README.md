# Baselines

MSP, Energy and ReAct+Energy are deterministic post-hoc scores computed on the same
checkpoint, the same preprocessing, the same ID/OOD images and the same evaluator as
P&C, from the same cached CLS residual. Only the score function differs.

ReAct clipping threshold: 0.6918 (p90 of penultimate activations on the ID *training*
temperature pool; no OOD and no validation data).

Per-dataset metrics: `baseline_ood_metrics.csv`. Aggregates and P&C comparison:
`../metrics/ood_results.json` and `../tables/imagenet_vit_main.md`.

MC-Dropout was not run: torchvision's ViT-B/16 ships with every Dropout p=0.0 and
attention dropout 0.0, so it would require altering the architecture (out of scope).
