# Interaction summary

## Best scale & bootstrap within each block (mean val NLL)

| block | best scale | best bf | best NLL | NLL range across other factors |
|---|---|---|---|---|
| s1b0 | ps25 | bf0.05 | 0.1473 | 0.0124 |
| s1b1 | ps25 | bf0.05 | 0.1546 | 0.0030 |
| s2b0 | ps25 | bf0.1 | 0.1468 | 0.0861 |
| s2b1 | ps25 | bf0.05 | 0.1515 | 0.0187 |
| s3b0 | ps25 | bf0.05 | 0.1305 | 5.5192 |
| s3b1 | ps25 | bf0.05 | 0.1363 | 3.1888 |

## Best bootstrap fraction at each scale (pooled over blocks)

| scale | best bf | mean NLL |
|---|---|---|
| ps25 | bf0.05 | 0.1497 |
| ps50 | bf0.05 | 0.3996 |
| ps100 | bf0.05 | 1.2826 |

## Coordinate-descent path reconstruction (from the completed grid)

- Forward (block→scale→bootstrap, init ps25/bf0.05): → **s3b0 ps25 bf0.05** (NLL 0.1305)
- Reverse (bootstrap→scale→block, init s3b0/ps25): → **s3b0 ps25 bf0.05** (NLL 0.1305)
- Global optimum: **s3b0 ps25 bf0.05** (NLL 0.1305)
- Paths converge to same config: True
