# 11-env Far-OOD sensitivity — median across-env range per factor

Environments: 11 | seeds: 0, 1, 10, 100, 11, 123, 13, 17, 19, 2, 200, 23, 29, 3, 31, 314, 37, 4, 404, 41, 42, 5, 500, 6, 7, 8, 9 | metrics: ID RMSE (relative), Far NLL, Far AUROC, Far Spearman.

| Factor | ID RMSE med(range) | Far NLL med(range) | Far AUROC med(range) | Far Spearman med(range) | Worst env (AUROC) |
|---|---:|---:|---:|---:|---|
| scale | 0.081 | 1.901 | 0.016 | 0.090 | HumanoidStandup-v5 (0.200) |
| rank | 0.028 | 0.167 | 0.002 | 0.020 | HumanoidStandup-v5 (0.022) |
| bootstrap | 0.529 | 1.417 | 0.005 | 0.155 | Humanoid-v5 (0.434) |
| calib | 0.605 | 1.408 | 0.027 | 0.187 | Humanoid-v5 (0.387) |
| ridge | 2.803 | 1.632 | 0.028 | 0.076 | Ant-v5 (0.242) |
| layer | 0.022 | 0.195 | 0.003 | 0.023 | InvertedPendulum-v5 (0.063) |