# Gym PnC Member-Radius Ablation

## 2026-04-06: Optional Per-Member Radius Variation for Gym PnC

### Question

The default Gym PnC construction normalizes each member's coefficient vector and then scales every member to the same global perturbation size. This ablation tests whether allowing a small distribution of member radii improves uncertainty quality without hurting in-distribution behavior.

Concretely, the old construction was:

```python
coeffs = z / sigma
coeffs = coeffs / ||coeffs||
coeffs = coeffs * perturbation_scale
```

The new optional construction is:

```python
coeffs = z / sigma
coeffs = coeffs / ||coeffs||
coeffs = coeffs * (perturbation_scale * member_radius_multiplier[i])
```

Direction selection and normalization are unchanged. Only the final scalar radius differs by member.

### Implementation

The following options were added to the shared PnC/PJSVD ensemble code path:

- `member_radius_distribution="fixed"`: exact old behavior
- `member_radius_distribution="lognormal"`: positive member multipliers sampled from a log-normal distribution and renormalized to have sample mean 1
- `member_radius_distribution="two_point"`: implemented but not used in this ablation
- `member_radius_std`: controls log-normal spread
- `member_radius_values`: optional explicit values for `two_point`

The main code changes are in:

- `ensembles.py`
- `gym_tasks.py`
- `pnc.py`

### Main Gym Comparison

I first compared the strongest reported Gym PnC setting on each environment:

- Hopper-v5: `low`, `ps=1`
- HalfCheetah-v5: `low`, `ps=8`
- Ant-v5: `random`, `ps=8`
- Humanoid-v5: `random`, `ps=64`

All runs used:

- `k=40`
- `n=100`
- `seed=200`
- `multi` / `least_squares` / `projected_residual` / `full`
- probabilistic base model
- post-hoc variance calibration enabled
- CPU execution (`--cpu`) because the GPU Lanczos path hit a cuSolver runtime error on this machine

The varying-radius mode used:

- `member_radius_distribution=lognormal`
- `member_radius_std=0.15`

#### Fixed vs Lognormal (`std=0.15`)

| Env | Config | ID RMSE fixed / varying | ID NLL fixed / varying | ID ECE fixed / varying | Near NLL fixed / varying | Mid NLL fixed / varying | Far NLL fixed / varying | Far AUROC fixed / varying |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Hopper-v5 | low, ps=1 | 0.30371 / 0.30371 | -1.15201 / -1.15286 | 0.02125 / 0.02141 | 0.01242 / 0.01533 | 0.34908 / 0.35752 | 0.25980 / 0.26056 | 0.83225 / 0.83046 |
| HalfCheetah-v5 | low, ps=8 | 2.14603 / 2.14655 | 0.56003 / 0.55990 | 0.02752 / 0.02736 | 1.55924 / 1.55904 | 2.30572 / 2.30792 | 1.86955 / 1.87069 | 0.62548 / 0.62785 |
| Ant-v5 | random, ps=8 | 0.68298 / 0.68302 | -2.03473 / -2.03475 | 0.09766 / 0.09762 | 0.76549 / 0.76268 | 0.33575 / 0.33171 | 2.93720 / 2.91731 | 0.45357 / 0.45360 |
| Humanoid-v5 | random, ps=64 | 49.21706 / 49.21701 | 4.49873 / 4.49912 | 0.15557 / 0.15559 | 4.38923 / 4.38901 | 4.33924 / 4.33917 | 4.04624 / 4.04653 | 0.31038 / 0.31035 |

#### Interpretation

- The `std=0.15` log-normal radius change was mostly neutral.
- It did not materially hurt ID behavior.
- It yielded tiny OOD wins on Ant and a small far-AUROC gain on HalfCheetah.
- It yielded small OOD regressions on Hopper.
- Overall this was not strong enough to claim a clear improvement.

### Follow-Up Sweep: Larger Log-Normal Radius Spread

Because `std=0.15` was so narrow, I ran a follow-up sweep with:

- `member_radius_std in {0.3, 0.5, 0.8}`

To keep the follow-up cheap but informative, I used two representative strongest-setting configs:

- Hopper-v5: `low`, `ps=1`
- Ant-v5: `random`, `ps=8`

#### Hopper-v5 (`low`, `ps=1`)

| std | ID RMSE | ID NLL | Near NLL | Mid NLL | Far NLL | Far AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.303713 | -1.152009 | 0.012422 | 0.349080 | 0.259802 | 0.832248 |
| 0.15 | 0.303715 | -1.152863 | 0.015332 | 0.357516 | 0.260557 | 0.830461 |
| 0.3 | 0.303549 | -1.155480 | 0.013339 | 0.364777 | 0.263417 | 0.829839 |
| 0.5 | 0.303038 | -1.161544 | 0.004338 | 0.374876 | 0.273554 | 0.829191 |
| 0.8 | 0.302404 | -1.173450 | -0.001975 | 0.399947 | 0.296108 | 0.826283 |

**Observation:** increasing spread steadily improved ID NLL slightly, but OOD quality worsened overall, especially mid/far NLL and far AUROC. Hopper does not support the “larger spread helps OOD” hypothesis.

#### Ant-v5 (`random`, `ps=8`)

| std | ID RMSE | ID NLL | Near NLL | Mid NLL | Far NLL | Far AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.682980 | -2.034731 | 0.765486 | 0.335746 | 2.937198 | 0.453571 |
| 0.15 | 0.683017 | -2.034747 | 0.762683 | 0.331711 | 2.917311 | 0.453604 |
| 0.3 | 0.683035 | -2.034885 | 0.744423 | 0.331104 | 2.835840 | 0.460888 |
| 0.5 | 0.683152 | -2.035361 | 0.760969 | 0.353137 | 2.761773 | 0.460369 |
| 0.8 | 0.683385 | -2.036296 | 0.777871 | 0.368597 | 2.753806 | 0.443488 |

**Observation:** Ant behaved differently from Hopper. Moderate spread helped far OOD more clearly:

- `std=0.3` improved near, mid, and far OOD NLL, and also improved far AUROC.
- `std=0.5` improved far OOD NLL further, but started hurting mid OOD NLL.
- `std=0.8` pushed too far: far OOD NLL stayed improved, but near/mid OOD NLL and far AUROC degraded.

### Overall Conclusion

This experiment does not support changing the default Gym PnC construction away from a fixed radius.

Main conclusion:

- Narrow member-radius variation (`lognormal`, `std=0.15`) is mostly neutral.
- Larger spread can matter, but the effect is environment-dependent.
- Hopper shows a clear tradeoff: larger spread slightly improves ID NLL but hurts OOD NLL and AUROC.
- Ant shows a possible benefit at moderate spread, especially around `std=0.3`, but that benefit is not robust as spread increases.

Appendix-friendly summary:

> We tested an ablation in which PnC ensemble members were allowed to have different perturbation radii while preserving the original direction normalization logic. A narrow log-normal radius distribution centered at the original perturbation scale was largely neutral on the main Gym configurations. Increasing the radius spread produced mixed results: in some environments it slightly improved OOD uncertainty metrics, but in others it worsened OOD NLL and AUROC. Overall, the effect was not consistently beneficial, so we retained the fixed-radius construction as the default reported setting.

### Recommendation for Paper Reporting

If this is mentioned in the NeurIPS appendix, the most defensible claim is:

- fixed-radius remains the default
- per-member radius variation was explored as an ablation
- mild variation was mostly neutral
- more aggressive variation produced mixed, environment-dependent results
- therefore this did not become a mainline method change

### Exact Commands

Representative commands used for the main fixed-vs-varying comparison:

```bash
/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Hopper-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 1.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family low --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Hopper-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 1.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family low --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --member_radius_distribution lognormal --member_radius_std 0.15

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Ant-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 8.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family random --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --member_radius_distribution lognormal --member_radius_std 0.3
```

