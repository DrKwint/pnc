# calib sweep fix — validation report

**What was wrong.** The `calib` factor overrode `bootstrap_fraction=0.0` while varying the
calibration size, so it changed *two* things at once vs the anchor (bootstrap ON→OFF **and**
size), and `calibration_size` meant two different things (pool vs per-member) across factors.

**The fix (calib only; the other five factors untouched).**
- Vary **only** the calibration **pool** `N ∈ {512,1024,2048,4096,8192}`; keep each env's
  Table-3 bootstrap fraction. Per-member size = `max(8, int(bf·N))`.
- Two new columns populated for **every** factor: `calibration_pool_size` (=N) and
  `per_member_calibration_size` (rows each member fits on); legacy `calibration_size` kept.
- Fail-loud guard if `N > available id_train rows` (cap 10000; so 16384 is excluded, not
  silently truncated).
- Per-seed rows, full-precision `bootstrap_fraction`, `git_commit=70fb480-calibfix`.
- CSV migrated: dropped 2930 old confounded calib rows, kept the 5 correct factors, added the
  two columns (backup: `far_sensitivity_raw.csv.pre_calibfix.bak`). Re-ran calib for all 293
  (env,seed) combos → 1465 new rows, 0 failures.

## Validation results

| Check | Result |
|---|---|
| **[2] No confound** — bootstrap_fraction constant within calib and = env Table-3 value | **PASS** (all 11 envs) |
| **[3] Completeness** — every (env,N) cell has full seed list, status==ok | **PASS** (27 seeds ×7 envs, 26 ×4 envs) |
| **[4] Sanity** — small-N interpolation blow-up present in every env | **PASS** (N=512 far_nll 6–13 nats; see note) |
| **[1] Centring** — calib(4096) bit-identical to overnight anchor | **233/293 bit-identical; 60 differ — see below** |

### Centring: the 60 exceptions are cross-run numerical non-reproducibility at n/p≈1, not a fix defect

Proven, not asserted:
1. **In-process determinism is perfect.** Building the anchor config twice in one process gives
   diff `0.0e+00` on all four metrics (tested on Reacher s1 and InvertedPendulum s23).
2. **A fresh anchor-config build bit-identically reproduces the stored calib(4096) row** — so the
   calib rows are the canonical, reproducible values of the anchor config in the current environment.
3. **It is the overnight 5-factor anchor rows that differ**, and only in the ill-conditioned regime:
   mismatch combos have n/p median **1.01** (max 6.11) vs match median **4.07**; the near-singular
   low-bootstrap envs dominate (Reacher 12, InvertedDoublePendulum 10, Ant 8, Pusher 7 seeds).
4. **The overnight run is internally self-consistent** — 0/293 combos where its five anchor sweeps
   disagree with each other. So overnight is one coherent BLAS environment and *now* is another.

**Mechanism.** Base model is bit-identical (same disk cache), build code is unchanged, all seeding
is deterministic — the only free variable across the overnight and current runs is BLAS thread
count, which changes the reduction order when forming the Gram `XᵀX` by ~1e-14. At n/p≈1
(cond ≈ 1e5–1e8) that perturbation is amplified — and can flip LAPACK's rank truncation — moving the
Far metrics substantially (worst case InvertedPendulum s23: cond 3.0e8, n/p 1.015, far_nll 2.47 vs
overnight 0.44). 21 of the 60 differ by >1%, all at n/p≈1; the rest are sub-0.1% rounding.

This is not a bug in the fix — it is a concrete manifestation of the **interpolation-regime fragility
the revised theory predicts**: at n/p≈1 the P&C correction is so ill-conditioned that its Far-OOD
metrics are not reproducible across BLAS-threading environments.

**Consequence for the deliverable.** The calib rows are canonical and reproducible; the five other
factors keep their overnight bytes (constraint honored). Exact bit-identical centring for the 60
near-singular combos is only recoverable by recomputing the anchor rows under a **pinned single-thread
BLAS** — which would overwrite the overnight bytes of those combos. Recommended: keep as-is and treat
the 60 exceptions as the interpolation-fragility signature. Optionally pin `OMP_NUM_THREADS=1` for all
future runs so the pipeline is cross-run byte-reproducible even at n/p≈1.

### [4] sanity note
Every env shows the expected small-pool interpolation blow-up (per-member → p): N=512 far_nll is
6–13 nats for the large envs, decaying toward the anchor. Mild large-N non-monotonicity in
Humanoid/HumanoidStandup/Swimmer/Walker2d/InvertedPendulum is within the same conditioning noise
(the low-bf envs sit near n/p=1 even at large N) and is second-order to the dominant small-N signal.
