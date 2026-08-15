"""Config validation for new P&C experiments (brief §10.2).

Makes it hard for a future script to silently revert to raw, unnormalized ridge tuning:
required declarations are hard failures, and the risky-but-legal choices are warnings that
name the document explaining why.
"""
from __future__ import annotations

from dataclasses import dataclass, field

REQUIRED = {
    "correction.objective_reduction": "docs/pnc_correction_strength.md §1",
    "correction.center": "docs/pnc_correction_strength.md §1",
    "correction.row_policy": "docs/pnc_correction_rows.md",
    "selection.policy": "docs/pnc_correction_strength.md §4",
    "selection.budget": "docs/pnc_correction_strength.md §4",
    "data.calibration_split_id": "docs/pnc_experiment_protocol.md §1",
    "data.id_validation_split_id": "docs/pnc_experiment_protocol.md §1",
    "isolation.stage_a_may_read": "docs/pnc_experiment_protocol.md §2",
}
_TOKEN_MIXING_POLICIES = {"sampled_valid_tokens", "all_valid_tokens"}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self):
        if self.errors:
            raise ValueError("invalid P&C experiment config:\n  - " +
                             "\n  - ".join(self.errors))


def _get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def validate_config(cfg: dict, *, strict_legacy: bool = False) -> ValidationResult:
    """Validate a new-experiment config. ``strict_legacy`` turns legacy warnings into errors."""
    r = ValidationResult()

    for path, doc in REQUIRED.items():
        if _get(cfg, path) is None:
            r.errors.append(f"missing required field '{path}' (see {doc})")

    red = _get(cfg, "correction.objective_reduction")
    if red is not None and red not in ("mean", "sum_legacy"):
        r.errors.append(f"correction.objective_reduction must be 'mean' or 'sum_legacy', "
                        f"got {red!r}")
    if red == "sum_legacy":
        msg = ("correction.objective_reduction='sum_legacy': raw lambda is not comparable "
               "across row counts; new experiments should use 'mean' "
               "(docs/pnc_correction_strength.md §1)")
        (r.errors if strict_legacy else r.warnings).append(msg)

    # a bare raw ridge value without declared legacy semantics
    for k in ("lambda_reg", "ridge", "lambda"):
        if k in cfg or k in (_get(cfg, "correction") or {}):
            if red != "sum_legacy":
                r.errors.append(
                    f"bare raw ridge field '{k}' present without "
                    "correction.objective_reduction='sum_legacy'; use "
                    "RidgeSpecification / correction.lambda_grid instead "
                    "(docs/pnc_correction_strength.md §1)")

    pol = _get(cfg, "selection.policy")
    if pol == "min_id_nll":
        r.warnings.append("selection.policy='min_id_nll' is retained for analysis only and "
                          "must be labelled in any table; the default is "
                          "'weakest_within_id_budget' (docs/pnc_correction_strength.md §4)")
    elif pol not in (None, "weakest_within_id_budget"):
        r.errors.append(f"unknown selection.policy {pol!r}")

    if _get(cfg, "selection.uses_ood") or _get(cfg, "selection.ood_metric"):
        r.errors.append("selection must not use OOD data (docs/pnc_experiment_protocol.md §2)")

    # row policy vs downstream mixing
    row = _get(cfg, "correction.row_policy")
    mixing = _get(cfg, "correction.downstream_mixing_description")
    if row == "cls_only":
        if mixing is None:
            r.errors.append("correction.row_policy='cls_only' requires "
                            "correction.downstream_mixing_description stating that no "
                            "downstream token mixing remains (docs/pnc_correction_rows.md §2)")
        elif "none" not in str(mixing).lower():
            if _get(cfg, "correction.override_row_policy_safety"):
                r.warnings.append("cls_only used before downstream token mixing with an "
                                  "explicit override: results are DIAGNOSTIC ONLY and must "
                                  "be stamped protocol_valid=false")
            else:
                r.errors.append(
                    "correction.row_policy='cls_only' is rejected when downstream token "
                    f"mixing remains ({mixing!r}); use one of {sorted(_TOKEN_MIXING_POLICIES)} "
                    "(docs/pnc_correction_rows.md §2)")
    if row == "pooled_only":
        r.warnings.append("correction.row_policy='pooled_only' is only justified when no "
                          "spatial mixing remains between the target and the pooling "
                          "operation (docs/pnc_correction_rows.md §3)")

    if _get(cfg, "correction.save_row_indices") is not True:
        r.errors.append("correction.save_row_indices must be true — a design cannot be "
                        "audited afterwards without it (docs/pnc_correction_rows.md §5)")

    if _get(cfg, "isolation.stage_a_may_read") is not None:
        bad = [s for s in _get(cfg, "isolation.stage_a_may_read")
               if any(t in str(s).lower() for t in ("ood", "test"))]
        if bad:
            r.errors.append(f"isolation.stage_a_may_read includes forbidden splits {bad}")
    if _get(cfg, "selection.frozen_stage", True) is False:
        r.errors.append("a frozen-selection stage is mandatory "
                        "(docs/pnc_experiment_protocol.md §2)")

    task = _get(cfg, "experiment.task_type")
    if task and task != "classification" and _get(cfg, "selection.budget") == "classification_default":
        r.errors.append(f"task_type={task!r} must define its own ID-preservation budget; the "
                        "classification template must not be inherited "
                        "(docs/pnc_correction_strength.md §4)")
    return r
