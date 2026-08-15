"""Config-validation tests (brief §10.2, §11 row-policy section)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.pnc_protocol.config import validate_config  # noqa: E402

TEMPLATE = ROOT / "experiments/templates/pnc_posthoc_experiment/config.yaml"


def _good() -> dict:
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["correction"]["downstream_mixing_description"] = \
        "later self-attention mixes tokens after the corrected FFN"
    return cfg


def test_template_is_valid_once_filled():
    r = validate_config(_good())
    assert r.ok, r.errors


def test_missing_required_fields_fail():
    cfg = _good(); del cfg["correction"]["objective_reduction"]
    r = validate_config(cfg)
    assert not r.ok and any("objective_reduction" in e for e in r.errors)


def test_sum_legacy_warns_and_can_be_strict():
    cfg = _good(); cfg["correction"]["objective_reduction"] = "sum_legacy"
    assert validate_config(cfg).ok
    assert any("sum_legacy" in w for w in validate_config(cfg).warnings)
    assert not validate_config(cfg, strict_legacy=True).ok


def test_bare_raw_ridge_value_is_rejected():
    cfg = _good(); cfg["correction"]["lambda_reg"] = 1000
    r = validate_config(cfg)
    assert not r.ok and any("bare raw ridge" in e for e in r.errors)


def test_bare_raw_ridge_allowed_only_with_declared_legacy_semantics():
    cfg = _good()
    cfg["correction"]["lambda_reg"] = 1000
    cfg["correction"]["objective_reduction"] = "sum_legacy"
    assert validate_config(cfg).ok


def test_ood_selection_is_rejected():
    cfg = _good(); cfg["selection"]["uses_ood"] = True
    assert not validate_config(cfg).ok


def test_cls_only_rejected_before_token_mixing():
    cfg = _good()
    cfg["correction"]["row_policy"] = "cls_only"
    cfg["correction"]["downstream_mixing_description"] = \
        "layers 2-5 self-attention still mix tokens"
    r = validate_config(cfg)
    assert not r.ok and any("cls_only" in e for e in r.errors)


def test_cls_only_allowed_at_final_block():
    cfg = _good()
    cfg["correction"]["row_policy"] = "cls_only"
    cfg["correction"]["downstream_mixing_description"] = \
        "none after the corrected final-block FFN"
    assert validate_config(cfg).ok


def test_cls_only_override_is_diagnostic_only():
    cfg = _good()
    cfg["correction"]["row_policy"] = "cls_only"
    cfg["correction"]["downstream_mixing_description"] = "later attention mixes tokens"
    cfg["correction"]["override_row_policy_safety"] = True
    r = validate_config(cfg)
    assert r.ok and any("DIAGNOSTIC ONLY" in w for w in r.warnings)


def test_row_indices_must_be_saved():
    cfg = _good(); cfg["correction"]["save_row_indices"] = False
    assert not validate_config(cfg).ok


def test_regression_task_cannot_inherit_classification_budget():
    cfg = _good()
    cfg["experiment"]["task_type"] = "regression"
    r = validate_config(cfg)
    assert not r.ok and any("own ID-preservation budget" in e for e in r.errors)


def test_stage_a_cannot_read_ood_or_test():
    cfg = _good(); cfg["isolation"]["stage_a_may_read"] = ["calibration", "ood_far"]
    assert not validate_config(cfg).ok


def test_min_id_nll_is_allowed_but_warns():
    cfg = _good(); cfg["selection"]["policy"] = "min_id_nll"
    r = validate_config(cfg)
    assert r.ok and any("analysis only" in w for w in r.warnings)
