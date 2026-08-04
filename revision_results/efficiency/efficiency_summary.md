# Priority 4 — efficiency accounting (MuJoCo measured; CIFAR from cached inference_cost.json)

Construction, storage, and inference are reported separately, and marginal (given a pretrained
base) is distinguished from total (including base production). Sources:
`efficiency_construction_mujoco.csv`, `efficiency_storage.csv`,
`efficiency_inference_mujoco.csv`, `efficiency_cifar_inference.csv`,
`results/cifar10/inference_cost.json`.

## 1. Construction cost — P&C's decisive win
Ant-v5, mean over seeds {0,10,42}, `train_time` from the cached result JSONs:

| method | # trained base nets | construction time | vs P&C |
|---|:---:|---:|---:|
| **P&C (bf0.1, M=50)** | **1** | **12.4 s** | 1× |
| Deep Ensemble (×50, paper config) | 50 | 2165.8 s | **175× more** |
| SWAG (n=100) | 1 + SGD trajectory | 148.3 s | 12× more |
| Laplace | 1 + curvature | 24.8 s | 2× more |
| MC Dropout | 1 | 26.0 s | 2× more |

- **Total cost to produce the ensemble:** P&C needs **one** trained base MLP; a 50-member Deep
  Ensemble needs **fifty** independently trained nets. That is the ~175× gap (12.4 s vs 2166 s),
  and it is apples-to-apples with the paper's "Deep Ensemble (×50)".
- **Marginal cost given a pretrained base:** P&C's post-hoc step (direction search + closed-form
  LS correction for all 50 members) is ≈1–2 s (the 12.4 s is dominated by the one base training,
  ≈11 s). A Deep Ensemble has **no** cheap marginal path — every member is a new training run.
- SWAG's 148 s includes SGD-trajectory collection (a separate task, correctly counted here, not
  zeroed).

## 2. Persistent storage
Exact fp32 param counts (Ant-v5, M=50); see `efficiency_storage.csv` for all envs:

| representation | size | vs Deep Ensemble |
|---|---:|---:|
| Deep Ensemble (M=50) / P&C naive full-checkpoints | 37.1 MB | 1× |
| **P&C implemented** (shared base + per-member corrected blocks; what `ensembles.py` stores) | **16.8 MB** | **2.2× smaller** |
| **P&C minimal** (shared base + shared directions + per-member latent coeffs) | **7.15 MB** | **5.2× smaller** |
| SWAG (rank 20 + diagonal) | 16.3 MB | 2.3× smaller |

- The **implemented** shared-base+corrected-block saving (2.2×) is a real code property — the gym
  P&C stores only per-member correction weights + one base, not M full models
  (`ensembles.py:584`). The **minimal** 5.2× figure is the theoretical floor if corrected weights
  are recomputed from the tiny latent coefficients at load time (not currently done), and is
  labeled as such — not claimed as an implemented benefit.
- (MuJoCo Laplace uses KFAC-factored covariance, not a dense GGN; its storage is a small per-layer
  factor pair and is not the headline comparison, so it is omitted rather than mis-estimated.)

## 3. Inference cost — P&C is NOT cheaper (reported honestly)
An M-member P&C ensemble does **M member forward passes**, the same count as a Deep Ensemble of
size M. It is not a single-pass method.
- **MuJoCo micro-benchmark** (Ant-v5, GPU-synced, 20 warmup + 100 timed reps; distinct per-member
  input offsets prevent XLA from collapsing the passes): an M=50 forward costs **5.5× a single
  forward at batch 1** and **2.8× at batch 1000** (the 200-wide MLP under-saturates the GPU, so
  50 members cost far less than 50× — but still >1×).
- **CIFAR (measured, `inference_cost.json`, batch 256, ms/sample):** P&C-multi(50) **9.69**,
  P&C-single(50) **7.42**, MC-Dropout(32) 5.18, SWAG(50) 7.33, Laplace(50) 7.68, Deep
  Ensemble(5) **1.34**, single-pass baselines (MSP/Energy/Mahalanobis) ≈0.6–0.7. P&C's 50-member
  inference is proportional to M and is *slower* than a 5-member Deep Ensemble.

## Where P&C is cheaper, and where it is not
- **Cheaper — construction (independent training):** 1 base net vs M. ~175× vs DE-×50 on MuJoCo. ✅
- **Cheaper — marginal post-hoc build given a pretrained model:** ~1–2 s closed-form vs a full
  training run per DE member. ✅
- **Cheaper — storage:** 2.2× implemented, 5.2× at the minimal representation, vs DE. ✅
- **NOT cheaper — inference:** M forward passes, cost ∝ M, comparable to / slower than a Deep
  Ensemble per member. ❌ (Do not imply single-pass inference; only one base was *trained*, but
  M members are *run*.)
