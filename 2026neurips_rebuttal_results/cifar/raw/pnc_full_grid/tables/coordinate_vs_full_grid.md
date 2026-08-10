# Coordinate-descent vs full-grid selection

- **Global optimum:** s3b0 ps25 bf0.05 (mean val NLL 0.1305 ± 0.0049)
- **Coordinate-descent config:** s3b0 ps25 bf0.05 (mean val NLL 0.1305, **rank 1/54**)
- **Val-NLL regret (CD − global):** 0.0000  (median seed std 0.0067; exceeds seed variability: False)

## Top 10 configs by mean val NLL

| rank | config | mean val NLL | std | val acc |
|---|---|---|---|---|
| 1 | s3b0 ps25 bf0.05 | 0.1305 | 0.0049 | 95.51 |
| 2 | s3b0 ps25 bf0.1 | 0.1357 | 0.0047 | 95.34 |
| 3 | s3b1 ps25 bf0.05 | 0.1363 | 0.0047 | 95.61 |
| 4 | s3b0 ps25 bf0.2 | 0.1452 | 0.0039 | 95.37 |
| 5 | s2b0 ps25 bf0.1 | 0.1468 | 0.0058 | 95.00 |
| 6 | s1b0 ps25 bf0.05 | 0.1473 | 0.0062 | 95.23 |
| 7 | s2b0 ps25 bf0.2 | 0.1473 | 0.0061 | 95.14 |
| 8 | s1b0 ps25 bf0.1 | 0.1479 | 0.0058 | 95.15 |
| 9 | s1b0 ps25 bf0.2 | 0.1480 | 0.0058 | 95.19 |
| 10 | s2b1 ps25 bf0.05 | 0.1515 | 0.0066 | 94.94 |
