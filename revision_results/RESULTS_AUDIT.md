# RESULTS_AUDIT — rebuttal commitments vs. what exists

Branch `agent/collect-rebuttal-results`, cut from `neurips-2026-rebuttal` @ `70fb480`.
Companion files: `MANIFEST.md` (provenance per group), `STATUS.csv` (89 requested items),
`provenance/artifact_inventory.csv` (146 copied files + SHA-256),
`provenance/large_artifacts.csv` (6.22 GB indexed, not committed).

---

## 1. Completion summary

| status | items | share |
|---|---:|---:|
| `FOUND_VERIFIED` | 55 | 62% |
| `PARTIAL` | 8 | 9% |
| `FOUND_CONFLICTING` | 7 | 8% |
| `FOUND_UNVERIFIED` | 4 | 4% |
| `MISSING` | 15 | 17% |

| result group | verdict in one line |
|---|---|
| **DistilBERT / Banking77 / CLINC-OOS** | Complete. All 15 quoted values reproduce exactly from per-seed raw data. No inconsistency found. |
| **SCOD (MuJoCo)** | Complete and verified 11/11; the exact snapshot that generated the quoted P&C column was identified. |
| **Mechanism (HalfCheetah / Hopper)** | Complete. All 8 quoted values reproduce exactly from 40,000-point raw data. Only a plot is missing. |
| **Efficiency (MuJoCo)** | Complete and verified: 46.8×–61.7× matches the quoted 47–62×. |
| **MuJoCo sensitivity** | Raw data complete and richer than described (27 seeds, 11 envs, 7 factors) — but the **sharp/flat/worse classifier that produced the headline counts was never saved and its counts cannot be reproduced**. |
| **Shift tiers** | Quoted illustrative values verified, but a **newer, larger aggregation disagrees**; and no artifact covers all methods × 11 environments × Near/Mid/Far. |
| **Layer scope (MuJoCo)** | Quoted values verified — but they are a **HalfCheetah-only, 3-seed** average, not an average over environments. |
| **Finite-scale validation** | The lost relative-error number is recovered (~1e-13) and the 0.97–1.00 cosine range is confirmed — but the **raw data behind both is gone**, and the coverage is 3 envs × seed 0, not 11 envs × 3 seeds. |
| **Efficiency (CIFAR)** | Two of four inference numbers verify; the other two conflict; base-training time and all memory figures are absent. |
| **CIFAR sensitivity / CIFAR SCOD / CIFAR layer scope** | Not on this machine. Run elsewhere; nothing was copied back. The promised **cross-product was never designed**, let alone run. |

---

## 2. Quoted rebuttal numbers: reproduced or not

### Reproduced exactly (recomputed from located raw data)

| number | recomputed | source |
|---|---|---|
| SCOD mean Far AUROC 0.512 | 0.5121 | 33 `_metrics.json` files |
| SCOD Ant 0.110 / Hopper 0.028 / Humanoid 0.267 / Pusher 0.995 | 0.1102 / 0.0276 / 0.2667 / 0.9946 | same |
| Far/ID SCOD score ratio Hopper ≈0.32, Ant ≈0.50 | 0.3203, 0.4999 | same |
| P&C beats SCOD on 11/11 environments | 11/11 | both raw sources |
| P&C mean Far AUROC 0.922 | 0.9215 | `far_sensitivity_raw.csv.pre_bootfull_merge.bak` (11,720 rows) |
| P&C ≈47–62× cheaper than a matched 50-member Deep Ensemble | 61.7× / 49.0× / 46.8× | `efficiency_v2_rows.csv` |
| CIFAR SWAG 7.33 and Laplace 7.68 ms/sample | 7.3253, 7.6797 | `inference_cost.json` @ git HEAD |
| DistilBERT P&C 92.5 / 0.301 / 0.031 and uncorrected 92.1 / 1.079 / 0.525 | exact | `metrics/raw.csv` → `banking77_pnc.csv` |
| CLINC Near/Cross/Far ×3 methods (9 values) | exact | same |
| HalfCheetah ρ 0.904, 0.837/0.860/0.816, slope 1.968 ± 0.008 | exact | `mechanism_second_domain.csv` |
| Hopper ρ 0.749, 0.691/0.730/0.776, slope 0.324 ± 0.002 | exact | same |
| Ant/HalfCheetah/Hopper Near/Mid/Far AUROC (9 values) | exact | `priority5_pnc_tiers.csv` |
| Single/multi block AUROC (6 values) | exact to 3 dp | `priority1/sensitivity_mujoco.csv`, `sweep=='layer'` |
| Finite-scale cosine alignment 0.97–1.00 | Ant 1.00, Hopper 0.99–1.00, HalfCheetah 0.97–1.00 | `THEOREM_VALIDATION.md` §1.9/B.1 |

### Recovered (the number the rebuttal said was lost)

**Relative algebraic-identity error ≈ 1e-13 (float64).** The Eq 2/4 test-point residual
identity holds to median 9.4e-14 (Ant), 5.4e-13 (HalfCheetah), 8.2e-13 (Hopper); Ant's
Near/Mid/Far medians are 7.3e-14 / 6.9e-14 / 5.2e-14, max ≤ 2.3e-11. The independent
`identity.csv` distribution over 4 environments × 5 seeds × members × variants gives float64
`exact_rel_fro` median 5.97e-12, max 8.31e-11. Numerator/denominator, the literal-vs-SVD form
distinction, and near-zero-residual handling are all documented in `MANIFEST.md` §7.

### Not reproduced

| number | what the data says | verdict |
|---|---|---|
| Sensitivity cells 0/38/6, 0/44/0, 4/31/9, 0/43/1 (total 4/156/16) | no rule reproduces them; best of six candidates is 122 cells off in L1 | classifier **MISSING** |
| CIFAR P&C 95.69 / 90.99 / 37.39 / 94.83 / 19.52 | closest local analogue 95.70 / 91.11 / 34.99 / **94.04** / **21.10** | off-machine |
| CIFAR SCOD 95.74 / 89.69 / 39.40 / 92.56 / 21.41 | no CIFAR SCOD artifact exists | off-machine |
| CIFAR block ranges: acc 94.76–95.59, Near 89.3–91.6, Far 92.0–95.1 | no block sweep here; the *scale* sweep spans Near 89.3–91.8 and Far 92.2–95.2 but acc 94.89–95.85 | off-machine; **do not** substitute the scale sweep |
| CIFAR inference P&C 7.53, Deep Ensemble 7.26 ms/sample | local file: P&C 7.4186 (single) / 9.6881 (multi); DE n=5 = 1.3380 | **conflicting** |
| CIFAR base training 8,010 s | the string occurs in no artifact | **MISSING** |

---

## 3. Missing artifacts

1. **The sharp / flat / materially-worse classifier** — no script, no output, no intermediate.
2. **Practical tolerance definitions** (AUROC/Spearman 0.02, RMSE 5%, NLL 0.10) — appear only
   in the rebuttal prose; no config or script in the repository defines them.
3. **Original seed-paired differences and 95% CIs** — derived here from raw data, but no
   original artifact exists.
4. **Near-tier and Mid-tier Spearman correlations** — never computed by the harness for any
   method; only `far_spearman` exists.
5. **A calibration-pool × bootstrap grid at fixed effective correction size** — the confound was
   found and fixed (`CALIB_FIX_VALIDATION.md`), but the corrected sweep varies N at fixed bf,
   so bf·N is not held constant. The 2-D grid was never run.
6. **CIFAR one-factor sensitivity** (all five axes) — other machine.
7. **CIFAR target block × scale × bootstrap fraction cross-product** — never designed. The
   prepared script is coordinate descent. Marked `MISSING`, not approximated.
8. **CIFAR SCOD comparison** (three checkpoints) — other machine.
9. **CIFAR early/middle/late block comparison**, correction dimensions, per-block compute cost.
10. **CIFAR efficiency**: base training time, P&C construction time and its upper-bound status,
    Deep Ensemble / SWAG / Laplace construction timing, persistent disk storage.
11. **Materialized and peak GPU memory** — on *both* benchmarks. Never instrumented anywhere.
12. **MuJoCo reconstruction time** — not separable from `build_s` in any artifact.
13. **`artifacts/pnc_theory/bridge/` and `artifacts/pnc_theory/round1/`** — the raw JSONs behind
    the cosine-alignment and identity-error tables. Absent from disk *and* never committed.
14. **A plot / plotting script for the HalfCheetah–Hopper mechanism replication.**
15. **MuJoCo base checkpoints for the sensitivity study** — retrained per run, never archived;
    only the regenerated SCOD batch and the `pnc_theory` base models survive.

---

## 4. Conflicting result versions

**C1 — Two snapshots of the sensitivity raw data (resolved).** `far_sensitivity_raw.csv` has
grown from 11,720 to 14,520 rows since the rebuttal numbers were computed.

| P&C source | mean Far AUROC | HumanoidStandup | InvertedPendulum | Humanoid |
|---|---|---|---|---|
| current (14,520 rows) | 0.9197 | 0.9134 | 0.9130 | 0.3427 |
| **`.pre_bootfull_merge.bak` (11,720 rows)** | **0.9215** | **0.9277** | **0.9169** | **0.3443** |
| quoted | 0.922 | 0.928 | 0.919 | 0.344 |

**The 11,720-row snapshot generated the rebuttal numbers.** Both are preserved.

**C2 — Two Near/Mid/Far tier tables.** `priority5_pnc_tiers.csv` (3 envs, 5 seeds, bf=0.1/λ=0/
pool 10000) reproduces the quoted values exactly. The later 11-env, 27-seed corrected-anchor
study gives Far AUROC Ant 0.998, HalfCheetah 0.987, **Hopper 0.980 vs the quoted 0.951**.
Different anchor, seed set and checkpoints — not a pipeline fault.

**C3 — Layer-scope scope mismatch.** Quoted "average AUROC" = HalfCheetah-v5, 3 seeds. The
11-env, 27-seed sweep gives single 0.764/0.815/0.904 vs multi 0.799/0.843/0.922. The ordering
(multi ≥ single on every tier) is stable across both.

**C4 — Ridge centring.** `anchors.json` and `MUJOCO_FAR_SENSITIVITY_RESULTS.md` say the ridge
shrinks **toward original**; all 14,520 executed rows record `ridge_center = 'zero'`. The
headline sensitivity study did not use the convention its own anchor file specifies. This also
matters for §7: the exact identity is unconditional only under toward-original ridge or λ=0.

**C5 — HumanoidStandup-v5 selected hyperparameters.** `appendix_selected_hparams_paper.txt`
says λ=1e-4, bf=0.3, ps=32.0; `anchors.json` and every executed row say λ=1e-2, bf=0.20, ps=8.0.
Ten of eleven environments agree.

**C6 — SCOD report internal contradiction.** §5 prose gives Ant's Far/ID Fisher-energy ratio as
"0.22×"; its own §3 table gives 0.50. Raw data gives **0.4999**; the table is right.

**C7 — CIFAR inference latency.** Quoted P&C 7.53 and DE 7.26 vs local P&C 7.4186/9.6881 and
DE(n=5) 1.3380. A matched M=50 CIFAR Deep Ensemble was never benchmarked here; the quoted pair
implies one was, on the other machine.

**C8 — Correction-pool size is not uniform.** The protocol states 4,096; the `bootfull`
double-descent sweep ran at 10,000, and the multi-seed wrapper defaults to 10,000.

**C9 — Mid-tier availability.** Five environments lack a Mid tier by dataset design
(Swimmer, Reacher, Pusher, InvertedPendulum, InvertedDoublePendulum) — but **seven** have empty
Mid columns, because HumanoidStandup and Walker2d have Mid available and were not run with it.

**C10 — Seed-count mismatches.** The brief says 20 seeds; the data has 27 (no 20-seed snapshot
exists). SCOD uses 3 seeds against a 27-seed P&C column; a seed-matched comparison is provided
and moves the mean by 0.0002. The finite-scale cosine evidence is single-seed.

---

## 5. Recommended canonical source per paper table / figure

| paper element | canonical source | why |
|---|---|---|
| MuJoCo hyperparameter-sensitivity claim | `mujoco_sensitivity/far_sensitivity_raw.csv` (current, 14,520 rows) | largest and latest; 11 envs × 27 seeds; zero failures. Re-derive the headline classification with a **saved, documented** rule. |
| Any quoted P&C Far AUROC carried over from the rebuttal | `far_sensitivity_raw_11720rows_CANONICAL_FOR_QUOTED_NUMBERS.csv` | the only snapshot that reproduces the quoted values; use it *or* restate the numbers from the current file — do not mix. |
| SCOD vs P&C (MuJoCo) | `scod/scod_vs_pnc_mujoco_by_environment.csv` | seed-matched and 27-seed columns side by side, plus per-env score ratios. |
| Efficiency, MuJoCo | `efficiency/mujoco_construction_normalized.csv` | measured, matched-M, per-method, with the DE/P&C ratio recomputed. |
| Efficiency, CIFAR inference | `efficiency/cifar10_inference_cost_RECOVERED_FROM_GIT_HEAD.json` | the only measured CIFAR latency artifact here — but restate P&C and DE from it, or import the other machine's newer benchmark. |
| DistilBERT table | `distilbert/banking77_pnc.csv` (+ `metrics_raw_per_seed.csv`) | verified end to end from per-seed raw data. |
| Distance–disagreement mechanism | `mechanism/mechanism_second_domain.csv` | 40,000 point-level rows per environment. |
| Random vs Low | `mechanism/random_vs_low_evidence.md` | the only broadening beyond Ant — but re-check the scale axis before citing it as a Figure 3(C) extension. |
| Finite-scale identity | `finite_scale_validation/finite_identity_mujoco_tier1_v2.csv` for the error *distribution*; `THEOREM_VALIDATION.md` for the cosine bridge | the CSV is raw; the cosine table is transcribed and needs regeneration. |
| MuJoCo Near/Mid/Far main table | **currently none is complete.** Build from `far_sensitivity_raw.csv` anchor rows (P&C, 11 envs) + `priority5_full_mujoco_tables.csv` (44 methods, 3 envs) | state the coverage explicitly rather than implying 11 envs × all methods. |
| Layer scope, MuJoCo | `layer_scope/sourceB_11env_27seed_layer_by_environment.csv` | larger and later; if the quoted numbers are kept, label them "HalfCheetah-v5, 3 seeds". |
| Protocol / configuration appendix | `provenance/protocol_configuration_record.csv` | every row carries a verified `file:line`. |
| All CIFAR tables | **retrieve from the CIFAR machine** | nothing here is rebuttal-era. |

---

## 6. Experiments that genuinely need rerunning

Ordered by cost. Nothing below was rerun — no costly experiment was executed during this collection.

**Cheap, on this machine (minutes):**

1. **Re-derive the sensitivity cell classification** with a saved script and an explicitly
   documented rule. `provenance/verify_mujoco_sensitivity.py` gives a working scaffold;
   `classification_rule_sweep.csv` shows six rules and how far each lands from the quoted
   counts. Either publish the new counts or recover the original rule — the current counts
   are unsupported by any artifact.
2. **Regenerate `artifacts/pnc_theory/bridge/`** via
   `finite_scale_validation/script_validate_bridge.py`. The base models are present
   (`artifacts/pnc_theory/base_models/`, 41 files, checksummed). This restores raw backing for
   the cosine-alignment claim, which is currently transcription-only.
3. **Plot the HalfCheetah/Hopper mechanism replication** from the existing 40,000-point CSVs.

**Moderate, on this machine (hours, sequential GPU):**

4. **Extend the finite-scale validation to the claimed coverage** — 11 environments × 3 seeds
   × 4 regimes × both correction stages — or narrow the claim to what exists (3 envs, seed 0).
   Engines exist: `script_finite_identity.py`, `script_multilayer.py`.
5. **Run the calibration-pool × bootstrap grid at fixed effective correction size.** The raw
   CSV already carries `per_member_calibration_size`, `unique_calibration_rows` and
   `nominal_n_over_p`, so the design is a small extension of `script_run_sensitivity.py`.
6. **Add Near/Mid Spearman** to the evaluation path (`pnc_core/util.py` computes only Far) and
   re-emit the tier tables. No retraining needed if per-point arrays were retained.
7. **Fill the Mid tier for HumanoidStandup and Walker2d**, which have `simple` Minari data.
8. **Resolve C4**: rerun the anchor rows with `ridge_toward_orig=True` to match `anchors.json`,
   or correct the anchor file and the report text to say toward-zero.

**Requires the CIFAR machine (do not rerun here — no checkpoints, no OpenOOD data):**

9. CIFAR one-factor sensitivity, all five axes, seeds 0/1/2.
10. **CIFAR target block × scale × bootstrap fraction cross-product** — this one is a genuinely
    new experiment, not a retrieval; the promised design was never written down.
11. CIFAR SCOD on the same three checkpoints.
12. CIFAR efficiency accounting: base training time, P&C construction time, matched-M Deep
    Ensemble / SWAG / Laplace construction, disk storage, and inference latency at matched M.
13. CIFAR early/middle/late block comparison with correction dimensions and compute cost.

**Both benchmarks:**

14. **GPU memory instrumentation** (materialized and peak) — never measured anywhere.

---

## 7. Large artifacts intentionally omitted from git

6.22 GB across 1,141 files, indexed in `provenance/large_artifacts.csv` with a per-tree
SHA-256 manifest under `provenance/large_artifact_checksums/`. Verify a whole tree by
re-hashing its `.sha256` manifest.

| original path | files | MB | manifest SHA-256 |
|---|---:|---:|---|
| `results/posthoc_mujoco/scod/sketches` | 33 | 4400.46 | `b3683e59d8da6030ec061796b375d1873c6d59492175a953b03e548272767b2d` |
| `results/banking77_distilbert_pnc/members` | 16 | 876.03 | `6da54b5d3cf396fdf08f9da08fdacaa2227a16ff20bc916ed07d91a8f3bb4481` |
| `results/posthoc_mujoco/scod/predictions` | 213 | 590.23 | `b08f1fe4ee257f34d011a5b168d486ba48024fe310d93aa6ec8b6a97c6725193` |
| `results/banking77_distilbert_pnc/checkpoint` | 10 | 269.02 | `7a4dd31cdcdd5f5e38d09261e88d634f867072c506a326072a029d5398fbbb34` |
| `artifacts/pnc_theory/base_models` | 41 | 28.97 | `2cfd5ad8cdbae9fb8530b2bcfb3b5ae8d79310dc01db3c9346e1d24837850e35` |
| `artifacts/finite_transfer/mujoco_tier1_v2` (parquet) | 2 | 27.78 | `618d66f2af775d5db6cfa861e71bb2193e51e27730456aca3002352d4f321a2e` |
| `archive/artifacts_backup` (SCOD canonical base models) | 2 | 20.98 | `f83cc596f8ea6f903235ea0f73818a59920dadccebf7dc73d44ab8dc4a0d892a` |
| `results/neurips_2026_rebuttal/mujoco_sensitivity/raw` | 573 | 6.19 | `b15f29d511421ba105d98845ea8c96cd2ef9509531e9a37e5f907cfa389c3eb5` |
| `results/neurips_2026_rebuttal/priority3/pnc` | 10 | 2.88 | `6851c0a791416f15d5bea7e112140067e8edba6484aab53a363e56d037db8305` |
| `results/banking77_distilbert_pnc/predictions` | 175 | 2.40 | `af2e02e0126b916601dd0eb33257d95d99d6ca092582ed083429b4a3911642ba` |
| `results/posthoc_mujoco/scod/configs` | 33 | 0.02 | `dacdcdbea9f43256b5a2066315f8d6916287d6202d79043359bcd15bdd00bf9a` |
| `results/posthoc_mujoco/scod/timing` | 33 | 0.01 | `2f9009407055adc3eee87bc94827043b71863ca6294c9204d219a904faae1ae6` |

Two compact-but-not-tiny CSVs **were** committed because they are the primary evidence for the
headline claims: `far_sensitivity_raw.csv` (6.3 MB) and the 11,720-row snapshot (5.0 MB), plus
the 40,000-point mechanism CSVs (9.2 MB total).

---

## 8. Two process notes

* **The paper repository was unreachable.** `EtherealEq/perturb_and_correct` returns
  "Repository not found" over SSH and fails on credentials over HTTPS; `gh` is not installed.
  `revision_results/` was created in `/home/elean/pnc`, which holds the experiment code and an
  April manuscript copy. **No draft PR could be opened**, and no `.tex` file was modified.
* **Nothing was rerun and no source artifact was modified.** Every recovery of a deleted
  `results/cifar10` file used `git show HEAD:<path>` into a new destination; the working-tree
  deletions are untouched.
