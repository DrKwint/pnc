# Part 0 — finite transfer-defect identity

Worst case over members and evaluation points; `literal_*` is the brief's formula as written, `exact_*` is the regime-correct form.

| variant | dtype | n | rank | p | cond_G | literal_max_abs | literal_rel | exact_max_abs | exact_mean_abs | exact_rel | primal_vs_dual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| canonical | float32 | 100 | 200 | 201 | 5.410e+07 | 3.664e-01 | 1.108e-02 | 1.833e-03 | 6.770e-06 | 5.129e-05 | n/a |
| canonical | float64 | 100 | 201 | 201 | 5.410e+07 | 1.161e-09 | 4.109e-11 | 7.208e-10 | 1.760e-12 | 1.604e-11 | n/a |
| shipped default (lam=0, lstsq) | float32 | 100 | 200 | 201 | 6.930e+33 | 4.631e-01 | 1.253e-01 | 9.880e+00 | 1.821e-02 | 4.094e-01 | n/a |
| shipped default (lam=0, lstsq) | float64 | 100 | 201 | 201 | 6.256e+33 | 1.318e-09 | 4.239e-11 | 2.030e-11 | 7.542e-14 | 8.117e-13 | n/a |
| shipped ridge (toward zero) | float32 | 100 | 200 | 201 | 5.410e+07 | 3.662e-01 | 2.182e-01 | 2.316e-03 | 8.720e-06 | 6.171e-05 | n/a |
| shipped ridge (toward zero) | float64 | 100 | 201 | 201 | 5.410e+07 | 3.483e-01 | 2.182e-01 | 8.881e-10 | 1.761e-12 | 1.689e-11 | n/a |
| small calibration (dual form) | float32 | 100 | 200 | 201 | 7.434e+07 | 3.733e+00 | 4.975e-02 | 2.677e-03 | 8.148e-06 | 3.677e-05 | 1.836e+00 |
| small calibration (dual form) | float64 | 100 | 201 | 201 | 7.434e+07 | 1.634e-08 | 1.793e-10 | 4.584e-09 | 7.687e-12 | 8.315e-11 | 4.340e-10 |
