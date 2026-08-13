# ID-stability gate (frozen before any OOD data was read)

Base model top-1 on the 8,192-image ID selection pool: **93.823%**.

A configuration is ID-stable iff **all** of the following hold:

1. ensemble top-1 drop from base <= 0.25 percentage points
2. base/ensemble top-1 agreement >= 99%
3. all metrics finite
4. corrected median logit MSE < uncorrected median logit MSE
5. corrected p99 logit MSE <= uncorrected p99 logit MSE

Conditions 4 and 5 require the affine correction to actually improve member preservation
over the identical perturbation left uncorrected. Calibration residual is deliberately
**not** an acceptance criterion: the preflight showed it is lowest exactly where the
correction is most overfitted, so it is recorded as a diagnostic only.

Selection uses the ID selection pool drawn from ImageNet **training** data. The official
50,000-image validation set is not touched during selection, and no OOD data is read
before `selected_config.json` is frozen.
