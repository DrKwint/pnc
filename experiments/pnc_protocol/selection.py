"""ID-only weakest-admissible correction-strength selection.

Replaces "minimize ID-validation NLL" as the default. For ridge centred on the original
parameters, LARGER lambda means WEAKER correction, so among candidates that stay inside a
predeclared ID-preservation budget we take the largest correction-strength coordinate.

Never sees OOD data: the selector consumes candidate metrics that the caller must have
produced from correction and ID-validation data only, and the caller is responsible for
the process separation (Stage A writes FROZEN and exits; Stage B refuses to run without it).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal, Sequence

import numpy as np

Aggregation = Literal["mean", "all_seeds", "mean_and_guard"]

__all__ = ["PreservationConstraint", "IDPreservationBudget", "Candidate",
           "SelectionResult", "select_weakest_admissible",
           "CLASSIFICATION_DEFAULT_BUDGET"]


@dataclass(frozen=True)
class PreservationConstraint:
    """One ID-preservation requirement.

    ``max_degradation`` bounds (base - candidate) for metrics where higher is better, or
    (candidate - base) where lower is better — the caller supplies the already-signed
    degradation under ``metric``. ``min_value`` bounds the raw metric from below
    (e.g. base-prediction agreement >= 0.99).
    """
    metric: str
    max_degradation: float | None = None
    min_value: float | None = None
    aggregation: Aggregation = "mean_and_guard"

    def __post_init__(self):
        if self.max_degradation is None and self.min_value is None:
            raise ValueError(f"constraint on {self.metric!r} bounds nothing")


@dataclass(frozen=True)
class IDPreservationBudget:
    constraints: tuple[PreservationConstraint, ...]
    per_seed_guard_multiplier: float = 2.0
    task: str = "unspecified"

    def to_dict(self) -> dict:
        return {"task": self.task,
                "per_seed_guard_multiplier": self.per_seed_guard_multiplier,
                "constraints": [asdict(c) for c in self.constraints]}


# Template matching the Banking77 study. A starting point, NOT a universal theorem —
# every experiment must save its own budget before evaluating candidates.
CLASSIFICATION_DEFAULT_BUDGET = IDPreservationBudget(
    task="classification",
    constraints=(
        PreservationConstraint("accuracy_drop_pp", max_degradation=0.25),
        PreservationConstraint("nll_increase", max_degradation=0.01),
        PreservationConstraint("ece_increase", max_degradation=0.01),
        PreservationConstraint("base_agreement", min_value=0.99),
    ),
)


@dataclass
class Candidate:
    """One correction strength, with its per-seed ID-validation metrics.

    ``strength`` is the ordering coordinate (larger = weaker correction): the
    Gram-normalized lambda when available, else lambda_mean.
    """
    key: str
    strength: float
    per_seed: dict[int, dict[str, float]]
    extra: dict = field(default_factory=dict)

    def mean_metric(self, m: str) -> float:
        return float(np.mean([s[m] for s in self.per_seed.values()]))


@dataclass
class SelectionResult:
    selected: Candidate | None
    admissible: list[Candidate]
    rejected: dict[str, list[str]]
    status: Literal["ok", "no_admissible_candidate"]
    message: str
    fell_back_to_strongest: bool
    budget: dict

    def to_dict(self) -> dict:
        return {"status": self.status, "message": self.message,
                "fell_back_to_strongest": self.fell_back_to_strongest,
                "selected": (None if self.selected is None else
                             {"key": self.selected.key, "strength": self.selected.strength,
                              "mean_metrics": {m: self.selected.mean_metric(m)
                                               for m in next(iter(
                                                   self.selected.per_seed.values()))},
                              **self.selected.extra}),
                "n_admissible": len(self.admissible),
                "admissible_keys": [c.key for c in self.admissible],
                "rejected": self.rejected, "budget": self.budget}


def _violations(cand: Candidate, budget: IDPreservationBudget) -> list[str]:
    out = []
    for c in budget.constraints:
        vals = [s[c.metric] for s in cand.per_seed.values()]
        mean = float(np.mean(vals))
        g = budget.per_seed_guard_multiplier
        if c.max_degradation is not None:
            if mean > c.max_degradation:
                out.append(f"{c.metric}: mean {mean:.5g} > {c.max_degradation:g}")
            if c.aggregation == "mean_and_guard" and max(vals) > g * c.max_degradation:
                out.append(f"{c.metric}: a seed exceeds {g}x budget "
                           f"({max(vals):.5g} > {g * c.max_degradation:g})")
            if c.aggregation == "all_seeds" and max(vals) > c.max_degradation:
                out.append(f"{c.metric}: a seed exceeds budget ({max(vals):.5g})")
        if c.min_value is not None:
            if mean < c.min_value:
                out.append(f"{c.metric}: mean {mean:.5g} < {c.min_value:g}")
            slack = (1.0 - c.min_value) * g
            if c.aggregation == "mean_and_guard" and min(vals) < c.min_value - slack:
                out.append(f"{c.metric}: a seed below guard "
                           f"({min(vals):.5g} < {c.min_value - slack:.5g})")
            if c.aggregation == "all_seeds" and min(vals) < c.min_value:
                out.append(f"{c.metric}: a seed below budget ({min(vals):.5g})")
    return out


def select_weakest_admissible(candidates: Sequence[Candidate],
                              budget: IDPreservationBudget) -> SelectionResult:
    """Largest admissible ``strength``; deterministic tie-break on ``key``.

    Returns ``status='no_admissible_candidate'`` rather than silently choosing the
    least-bad option — the brief forbids that fallback. Selecting the STRONGEST
    correction when it is the only admissible point is a valid, expected outcome.
    """
    if not candidates:
        raise ValueError("no candidates supplied")
    required = {c.metric for c in budget.constraints}
    for cand in candidates:
        for seed, s in cand.per_seed.items():
            missing = required - set(s)
            if missing:
                raise ValueError(f"candidate {cand.key!r} seed {seed} missing {sorted(missing)}")

    rejected: dict[str, list[str]] = {}
    admissible = []
    for cand in candidates:
        v = _violations(cand, budget)
        if v:
            rejected[cand.key] = v
        else:
            admissible.append(cand)

    if not admissible:
        return SelectionResult(
            None, [], rejected, "no_admissible_candidate",
            "No candidate satisfied the ID-preservation budget. Do not pick the least-bad "
            "option: extend the grid toward STRONGER correction (smaller lambda) or reduce "
            "the perturbation scale, then re-run selection.",
            False, budget.to_dict())

    strongest = min(c.strength for c in candidates)
    best = max(admissible, key=lambda c: (c.strength, c.key))
    fell_back = np.isclose(best.strength, strongest)
    msg = (f"selected weakest admissible correction: strength {best.strength:.6g} "
           f"({len(admissible)}/{len(candidates)} admissible)")
    if fell_back:
        msg += ("; this is the STRONGEST correction on the grid — no weakening was "
                "admissible, which is a valid outcome (cf. MuJoCo in the large-lambda study)")
    return SelectionResult(best, admissible, rejected, "ok", msg, bool(fell_back),
                           budget.to_dict())
