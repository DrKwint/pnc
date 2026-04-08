# Gym PnC Antithetic-Pairing Interaction

## 2026-04-06: Optional Antithetic Pairing for Gym PnC

### Question

The default Gym PnC construction samples each ensemble member's latent coefficient vector independently. This follow-up ablation tests whether antithetic sampling, i.e. pairing coefficient samples as `(z, -z)`, keeps the ensemble more centered around the base predictor and changes the member-radius results.

When `antithetic_pairing=True`, the construction becomes:

```python
base_z = sample_gaussians(n_perturbations // 2)
all_z = concat(base_z, -base_z)
```

If `n_perturbations` is odd, one extra unpaired Gaussian sample is appended at the end. The downstream direction normalization, whitening, and final member-radius scaling are unchanged.

### Implementation

The following option was added to the Gym PJSVD experiment path:

- `antithetic_pairing=False`: exact old behavior
- `antithetic_pairing=True`: sample coefficient vectors in `(z, -z)` pairs

Implementation details:

- pairing is applied before the existing normalization and perturbation-scaling logic
- even `n_perturbations` gives exact pairing
- odd `n_perturbations` adds one extra unpaired sample
- the output filename gets an `_anti` tag so antithetic runs do not overwrite the baseline

The main code changes are in:

- `gym_tasks.py`
- `ensembles.py`

### Experimental Setup

To test the interaction with per-member radius variation, I reran the same representative radius sweep used in the previous note:

- Hopper-v5: `low`, `ps=1`
- Ant-v5: `random`, `ps=8`

Shared settings:

- `k=40`
- `n=100`
- `seed=200`
- `multi` / `least_squares` / `projected_residual` / `full`
- probabilistic base model
- post-hoc variance calibration enabled
- CPU execution (`--cpu`) because the GPU Lanczos path hit a cuSolver runtime error on this machine

Radius settings compared:

- `fixed`
- `lognormal`, `std=0.15`
- `lognormal`, `std=0.3`
- `lognormal`, `std=0.5`
- `lognormal`, `std=0.8`

### Hopper-v5 (`low`, `ps=1`)

Numbers below are `non-antithetic -> antithetic`.

| std | ID RMSE | ID NLL | Near NLL | Mid NLL | Far NLL | Far AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.303713 -> 0.303831 | -1.152009 -> -1.151752 | 0.012422 -> 0.016478 | 0.349080 -> 0.339139 | 0.259802 -> 0.259376 | 0.832248 -> 0.831112 |
| 0.15 | 0.303715 -> 0.303767 | -1.152863 -> -1.153020 | 0.015332 -> 0.016229 | 0.357516 -> 0.346188 | 0.260557 -> 0.256914 | 0.830461 -> 0.830842 |
| 0.3 | 0.303549 -> 0.303579 | -1.155480 -> -1.155672 | 0.013339 -> 0.013682 | 0.364777 -> 0.350366 | 0.263417 -> 0.255538 | 0.829839 -> 0.830246 |
| 0.5 | 0.303038 -> 0.303184 | -1.161544 -> -1.161831 | 0.004338 -> 0.003195 | 0.374876 -> 0.356663 | 0.273554 -> 0.254799 | 0.829191 -> 0.829854 |
| 0.8 | 0.302404 -> 0.302444 | -1.173450 -> -1.173535 | -0.001975 -> -0.002558 | 0.399947 -> 0.395874 | 0.296108 -> 0.264607 | 0.826283 -> 0.828128 |

#### Interpretation

- On Hopper, antithetic pairing leaves ID behavior almost unchanged.
- At small radius spread the effect is mixed.
- Once radius variation is moderate or large, antithetic pairing generally improves mid/far OOD NLL.
- The strongest Hopper effect appears at `std=0.8`, where far OOD NLL improves from `0.296108` to `0.264607` and far AUROC also rises slightly.

So on Hopper, antithetic pairing looks stabilizing when radius spread is already pushing members away from the base predictor.

### Ant-v5 (`random`, `ps=8`)

Numbers below are `non-antithetic -> antithetic`.

| std | ID RMSE | ID NLL | Near NLL | Mid NLL | Far NLL | Far AUROC |
|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.682980 -> 0.682957 | -2.034731 -> -2.034565 | 0.765486 -> 0.772967 | 0.335746 -> 0.359131 | 2.937198 -> 2.968439 | 0.453571 -> 0.455107 |
| 0.15 | 0.683017 -> 0.682960 | -2.034747 -> -2.034691 | 0.762683 -> 0.772929 | 0.331711 -> 0.361742 | 2.917311 -> 2.978206 | 0.453604 -> 0.451667 |
| 0.3 | 0.683035 -> 0.683000 | -2.034885 -> -2.034713 | 0.744423 -> 0.759934 | 0.331104 -> 0.362134 | 2.835840 -> 2.933617 | 0.460888 -> 0.449731 |
| 0.5 | 0.683152 -> 0.683196 | -2.035361 -> -2.035162 | 0.760969 -> 0.751757 | 0.353137 -> 0.366425 | 2.761773 -> 2.897805 | 0.460369 -> 0.444160 |
| 0.8 | 0.683385 -> 0.683478 | -2.036296 -> -2.035958 | 0.777871 -> 0.739046 | 0.368597 -> 0.357989 | 2.753806 -> 2.831077 | 0.443488 -> 0.425595 |

#### Interpretation

- On Ant, antithetic pairing again leaves ID behavior almost unchanged.
- However, it mostly hurts the OOD gains from radius variation.
- Far OOD NLL is worse than the non-antithetic run at every radius setting.
- Far AUROC also drops for every varying-radius setting.
- At `std=0.8`, near and mid OOD NLL improve, but far OOD NLL and far AUROC still deteriorate.

So on Ant, antithetic pairing appears to suppress useful far-OOD diversity rather than improving centering in a helpful way.

### Overall Conclusion

This interaction study does not support making antithetic pairing the new default Gym PnC construction.

Main conclusion:

- antithetic pairing has almost no effect on ID RMSE or ID NLL in these runs
- its OOD effect is environment-dependent
- on Hopper, antithetic pairing can partly offset the OOD degradation from larger radius spread
- on Ant, antithetic pairing mostly removes the modest OOD gains from radius variation

Appendix-friendly summary:

> We tested an antithetic-pairing variant of Gym PnC in which ensemble coefficient samples were generated in `(z, -z)` pairs before the usual normalization and scaling steps. This change had little effect on in-distribution metrics, but its interaction with per-member radius variation was mixed. On Hopper it modestly improved some OOD metrics when the radius spread was large, while on Ant it generally worsened far-OOD NLL and AUROC. Overall, the effect was not consistently beneficial, so we did not adopt antithetic pairing as the default reported setting.

### Recommendation for Paper Reporting

If this appears in the NeurIPS appendix, the cleanest takeaway is:

- antithetic pairing was explored as a centering ablation
- ID behavior was largely unchanged
- OOD effects depended strongly on environment
- because the benefits were not robust, the baseline independent-sampling construction remains the default

### Exact Commands

Commands used for the antithetic interaction sweep:

```bash
/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Hopper-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 1.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family low --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Hopper-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 1.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family low --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true --member_radius_distribution lognormal --member_radius_std 0.15

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Hopper-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 1.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family low --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true --member_radius_distribution lognormal --member_radius_std 0.3

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Hopper-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 1.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family low --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true --member_radius_distribution lognormal --member_radius_std 0.5

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Hopper-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 1.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family low --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true --member_radius_distribution lognormal --member_radius_std 0.8

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Ant-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 8.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family random --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Ant-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 8.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family random --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true --member_radius_distribution lognormal --member_radius_std 0.15

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Ant-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 8.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family random --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true --member_radius_distribution lognormal --member_radius_std 0.3

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Ant-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 8.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family random --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true --member_radius_distribution lognormal --member_radius_std 0.5

/home/equint/github/pnc/.venv/bin/python /home/equint/github/pnc/run_experiment.py --cpu --task GymPJSVD --env Ant-v5 --steps 10000 --subset_size 10000 --n_directions 40 --n_perturbations 100 --perturbation_sizes 8.0 --seed 200 --activation relu --correction_mode least_squares --layer_scope multi --pjsvd_family random --safe_subspace_backend projected_residual --use_full_span true --probabilistic_base_model true --posthoc_calibrate true --antithetic_pairing true --member_radius_distribution lognormal --member_radius_std 0.8
```
