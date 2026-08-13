# ID-only configuration selection

**OOD data accessed before configuration freeze: NO**

Selection used only the 8,192-image ID selection pool drawn from ImageNet
*training* data. The official validation set and every OOD dataset were untouched
at the time this configuration was frozen.

- Stage-B configurations evaluated: 18
- Passing the ID-stability gate: 15
- Tied on NLL (within 0.002): 8

## Rule (frozen before OOD evaluation)

1. apply the ID-stability gate (`ID_STABILITY_RULE.md`)
2. lowest ID-selection NLL
3. ties within 0.002 NLL broken by: larger scale, then n_cal=16,384, then smaller ridge

## Selected

- realized r = 0.3750 (target 0.375)
- lambda = 1.0
- n_cal = 16384
- ID-selection NLL = 0.3260, top-1 = 93.799% (base 93.823%)

## All Stage-B configurations

| r | lambda | n_cal | top-1 | NLL | ECE | agree | median MSE | p99 MSE | gate |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.001 | 16384 | 93.799 | 0.3246 | 0.1207 | 0.9984 | 4.71e-04 | 8.17e-03 | fail |
| 0.25 | 1 | 16384 | 93.787 | 0.3245 | 0.1206 | 0.9982 | 4.60e-04 | 7.61e-03 | fail |
| 0.25 | 100 | 16384 | 93.823 | 0.3248 | 0.1214 | 0.9982 | 5.39e-04 | 5.76e-03 | PASS |
| 0.25 | 0.001 | 32768 | 93.835 | 0.3244 | 0.1210 | 0.9983 | 4.16e-04 | 6.40e-03 | PASS |
| 0.25 | 1 | 32768 | 93.823 | 0.3243 | 0.1208 | 0.9982 | 4.15e-04 | 6.27e-03 | PASS |
| 0.25 | 100 | 32768 | 93.811 | 0.3245 | 0.1210 | 0.9983 | 4.72e-04 | 5.52e-03 | PASS |
| 0.375 | 0.001 | 16384 | 93.811 | 0.3262 | 0.1217 | 0.9977 | 1.03e-03 | 1.77e-02 | fail |
| 0.375 | 1 | 16384 | 93.799 | 0.3260 | 0.1215 | 0.9976 | 9.99e-04 | 1.65e-02 | PASS |
| 0.375 | 100 | 16384 | 93.835 | 0.3265 | 0.1228 | 0.9978 | 1.18e-03 | 1.28e-02 | PASS |
| 0.375 | 0.001 | 32768 | 93.799 | 0.3257 | 0.1212 | 0.9980 | 9.04e-04 | 1.39e-02 | PASS |
| 0.375 | 1 | 32768 | 93.799 | 0.3256 | 0.1212 | 0.9979 | 9.01e-04 | 1.36e-02 | PASS |
| 0.375 | 100 | 32768 | 93.848 | 0.3259 | 0.1223 | 0.9979 | 1.03e-03 | 1.22e-02 | PASS |
| 0.5 | 0.001 | 16384 | 93.738 | 0.3284 | 0.1221 | 0.9963 | 1.75e-03 | 3.03e-02 | PASS |
| 0.5 | 1 | 16384 | 93.738 | 0.3282 | 0.1219 | 0.9966 | 1.71e-03 | 2.83e-02 | PASS |
| 0.5 | 100 | 16384 | 93.848 | 0.3290 | 0.1246 | 0.9976 | 2.02e-03 | 2.22e-02 | PASS |
| 0.5 | 0.001 | 32768 | 93.799 | 0.3275 | 0.1221 | 0.9968 | 1.54e-03 | 2.38e-02 | PASS |
| 0.5 | 1 | 32768 | 93.787 | 0.3275 | 0.1219 | 0.9969 | 1.54e-03 | 2.33e-02 | PASS |
| 0.5 | 100 | 32768 | 93.835 | 0.3279 | 0.1234 | 0.9976 | 1.76e-03 | 2.11e-02 | PASS |
