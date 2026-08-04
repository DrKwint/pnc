# Reproduction — Ant-v5 P&C (canonical cached config)

- Config: `Ant-v5 PJSVD-Multi-LS-random bf0.1 prob k20 n50 ps[5,10,20,50] h[200x4] seed0`
- Producing script: `experiments/scripts/run_ant_bf0.1_fullxsub_all_seeds.sh`
- Cached: `pjsvd_multi_least_squares_random_projected_residual_prob_bf0.1_k20_n50_ps5.0-10.0-20.0-50.0_h200-200-200-200_act-relu_seed0.json`
- Repro wall: 41.3s; base train_time: 13.448949575424194s
- Selected size by min nll_val — repro: 10.0, cached: 10.0

## Max |repro − cached| per field (over all 4 perturbation sizes)

| field | max abs diff |
|---|---:|
| rmse_val | 0.000000 |
| nll_val | 0.000000 |
| rmse_id | 0.000000 |
| nll_id | 0.000000 |
| rmse_ood_near | 0.000000 |
| rmse_ood_mid | 0.000000 |
| rmse_ood_far | 0.000000 |
| nll_ood_near | 0.000000 |
| nll_ood_mid | 0.000000 |
| nll_ood_far | 0.000000 |
| auroc_ood_near | 0.000000 |
| auroc_ood_mid | 0.000000 |
| auroc_ood_far | 0.000000 |

## Full per-size comparison

| ps | field | repro | cached | abs_diff |
|---|---|---:|---:|---:|
| 5.0 | rmse_val | 0.5616 | 0.5616 | 0.0000 |
| 5.0 | nll_val | -1.8164 | -1.8164 | 0.0000 |
| 5.0 | rmse_id | 0.5480 | 0.5480 | 0.0000 |
| 5.0 | nll_id | -1.8050 | -1.8050 | 0.0000 |
| 5.0 | rmse_ood_near | 0.7366 | 0.7366 | 0.0000 |
| 5.0 | rmse_ood_mid | 0.8926 | 0.8926 | 0.0000 |
| 5.0 | rmse_ood_far | 2.0637 | 2.0637 | 0.0000 |
| 5.0 | nll_ood_near | 0.2400 | 0.2400 | 0.0000 |
| 5.0 | nll_ood_mid | 0.0965 | 0.0965 | 0.0000 |
| 5.0 | nll_ood_far | 1.0374 | 1.0374 | 0.0000 |
| 5.0 | auroc_ood_near | 0.7173 | 0.7173 | 0.0000 |
| 5.0 | auroc_ood_mid | 0.7918 | 0.7918 | 0.0000 |
| 5.0 | auroc_ood_far | 0.9979 | 0.9979 | 0.0000 |
| 10.0 | rmse_val | 0.5623 | 0.5623 | 0.0000 |
| 10.0 | nll_val | -1.8452 | -1.8452 | 0.0000 |
| 10.0 | rmse_id | 0.5495 | 0.5495 | 0.0000 |
| 10.0 | nll_id | -1.8049 | -1.8049 | 0.0000 |
| 10.0 | rmse_ood_near | 0.7416 | 0.7416 | 0.0000 |
| 10.0 | rmse_ood_mid | 0.9071 | 0.9071 | 0.0000 |
| 10.0 | rmse_ood_far | 2.1351 | 2.1351 | 0.0000 |
| 10.0 | nll_ood_near | 0.1547 | 0.1547 | 0.0000 |
| 10.0 | nll_ood_mid | 0.0155 | 0.0155 | 0.0000 |
| 10.0 | nll_ood_far | 1.1246 | 1.1246 | 0.0000 |
| 10.0 | auroc_ood_near | 0.7387 | 0.7387 | 0.0000 |
| 10.0 | auroc_ood_mid | 0.7995 | 0.7995 | 0.0000 |
| 10.0 | auroc_ood_far | 0.9981 | 0.9981 | 0.0000 |
| 20.0 | rmse_val | 0.5574 | 0.5574 | 0.0000 |
| 20.0 | nll_val | -1.8373 | -1.8373 | 0.0000 |
| 20.0 | rmse_id | 0.5466 | 0.5466 | 0.0000 |
| 20.0 | nll_id | -1.8127 | -1.8127 | 0.0000 |
| 20.0 | rmse_ood_near | 0.7385 | 0.7385 | 0.0000 |
| 20.0 | rmse_ood_mid | 0.8993 | 0.8993 | 0.0000 |
| 20.0 | rmse_ood_far | 2.0796 | 2.0796 | 0.0000 |
| 20.0 | nll_ood_near | 0.0994 | 0.0994 | 0.0000 |
| 20.0 | nll_ood_mid | 0.0294 | 0.0294 | 0.0000 |
| 20.0 | nll_ood_far | 1.1029 | 1.1029 | 0.0000 |
| 20.0 | auroc_ood_near | 0.7316 | 0.7316 | 0.0000 |
| 20.0 | auroc_ood_mid | 0.7880 | 0.7880 | 0.0000 |
| 20.0 | auroc_ood_far | 0.9982 | 0.9982 | 0.0000 |
| 50.0 | rmse_val | 0.5563 | 0.5563 | 0.0000 |
| 50.0 | nll_val | -1.8411 | -1.8411 | 0.0000 |
| 50.0 | rmse_id | 0.5440 | 0.5440 | 0.0000 |
| 50.0 | nll_id | -1.8136 | -1.8136 | 0.0000 |
| 50.0 | rmse_ood_near | 0.7213 | 0.7213 | 0.0000 |
| 50.0 | rmse_ood_mid | 0.8562 | 0.8562 | 0.0000 |
| 50.0 | rmse_ood_far | 1.6758 | 1.6758 | 0.0000 |
| 50.0 | nll_ood_near | 0.0502 | 0.0502 | 0.0000 |
| 50.0 | nll_ood_mid | 0.0531 | 0.0531 | 0.0000 |
| 50.0 | nll_ood_far | 1.1366 | 1.1366 | 0.0000 |
| 50.0 | auroc_ood_near | 0.7270 | 0.7270 | 0.0000 |
| 50.0 | auroc_ood_mid | 0.7709 | 0.7709 | 0.0000 |
| 50.0 | auroc_ood_far | 0.9982 | 0.9982 | 0.0000 |
