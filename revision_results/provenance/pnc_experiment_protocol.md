# P&C experiment protocol

The default conventions for all **new** P&C experiments. Historical runs keep their own
semantics and remain reproducible unchanged.

---

## 1. Required declarations

A new experiment config must declare all of:

| field | why |
|---|---|
| `objective_reduction` | `mean` for new work; `sum_legacy` only to reproduce history |
| `center` | ridge centre, normally `original` |
| `correction_row_policy` | see `pnc_correction_rows.md` |
| `selection_policy` | normally `weakest_within_id_budget` |
| `id_preservation_budget` | task-appropriate, saved **before** evaluating candidates |
| calibration + validation split identifiers | reproducibility |
| OOD-isolation declaration | which stage may read OOD |

Validation **fails** when any is missing. Validation **warns or fails** when the config:

* uses `sum_legacy`;
* passes a bare raw ridge value (e.g. `lambda_reg: 1000`) without declaring legacy
  semantics;
* selects correction strength using OOD data;
* uses CLS-only rows before downstream token mixing;
* does not save correction-row indices;
* has no frozen-selection stage.

Example:

```yaml
correction:
  objective_reduction: mean
  center: original
  lambda_parameterization: gram_normalized   # or: mean
  lambda_grid: [1.0e-4, 3.0e-4, 1.0e-3, ..., 1.0e2]
  lambda_mode: memberwise_gram_normalized_lambda
  row_policy: sampled_valid_tokens
selection:
  policy: weakest_within_id_budget
  budget: classification_default
  development_seeds: [0, 10, 42]
  held_out_seeds: [123, 2026]
isolation:
  stage_a_may_read: [calibration, id_validation]
  stage_b_may_read: [id_test, ood_near, ood_cross, ood_far]
```

---

## 2. Two-stage isolation (mandatory)

### Stage A — ID-only construction and selection

May read: correction/calibration pool, ID validation, base checkpoint, saved perturbation
randomness.

Must **not** read: ID test (unless the protocol explicitly defines it as validation),
Near/Cross/Far OOD, or any OOD-derived summary.

Writes, then **terminates**:

```
selection/selected_config.json
selection/selected_config.sha256
selection/FROZEN
selection/selection_data_manifest.json
selection/withheld_data_manifest.json
```

Implement the restriction as a **whitelist loader** that logs what it withheld, not as a
convention. Banking77 v2 prints:

```
exposed:  Z_cal, Z_id_val, h0_cal, h0_id_val, y_cal, y_id_val
withheld: h0_cross, h0_far, h0_id_eval, h0_near, y_id_eval
```

### Stage B — final evaluation

Refuses to run without `FROZEN`; verifies the config hash; loads the frozen configuration
unmodified; only then loads ID test and OOD; never writes back a different selection.

---

## 3. Selection

Default: **weakest admissible correction within the ID-preservation budget**
(`experiments/pnc_protocol/selection.py`). Not minimum ID-validation NLL — that remains
available as `min_id_nll` for analysis and must be labelled wherever reported.

Selecting the *strongest* correction is a valid outcome when nothing weaker is admissible;
it is flagged in the result. Selecting nothing is also valid — the selector fails loudly
rather than picking the least-bad candidate.

---

## 4. Standard artifact layout

```
results/<experiment>/
├── MANIFEST.json
├── protocol.json
├── correction/
│   ├── objective.json          objective_reduction, center, grid, solver dtype
│   ├── row_policy.json         policy, downstream mixing, justification, row hashes
│   ├── row_indices.npz
│   └── diagnostics.csv         per member x strength (see §4.4 of the brief)
├── selection/
│   ├── budget.json
│   ├── candidate_metrics.csv
│   ├── selected_config.json
│   ├── selected_config.sha256
│   └── FROZEN
├── final_evaluation/
├── predictions/
├── logs/
└── REPORT.md
```

`REPORT.md` must state: whether the objective is mean or legacy sum; the ridge value in
**both** parameterizations; the correction-row semantics; the ID budget; the selected
normalized strength; whether weaker correction was admissible; and whether the result was
selected without OOD access.

---

## 5. Per-family defaults

| family | rows | budget | notes |
|---|---|---|---|
| Transformer, final block | `cls_only` | classification template | CLS-only valid; no downstream mixing |
| Transformer, earlier block | `sampled_valid_tokens` | classification template | CLS-only rejected by the preflight gate |
| CNN / conv block | `all_spatial_positions` or sampled | classification template | `pooled_only` needs justification |
| MLP regression (MuJoCo) | one row per example | **task-specific**, must be defined | classification template must not be inherited |

---

## 6. Migration status

* **Done**: shared solver modes, conversion helpers, normalized diagnostics, ID-preservation
  selector, Banking77 protocol v2, tests.
* **Not yet migrated**: MuJoCo (`ensembles.py`), CIFAR (`pnc.py` conv path), experiment
  templates, ImageNet. These keep `sum_legacy` behaviour and remain reproducible; see
  `pnc_correction_migration_report.md` for the remaining call sites.
* **Never migrated**: historical result artifacts. They are read-only.
