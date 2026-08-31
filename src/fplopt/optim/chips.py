"""Evaluate the best gameweek to play each chip over a horizon.

For each chip, re-solves the horizon MILP once per candidate gameweek with
that chip forced active there (and nowhere else), and compares the
resulting objective value against a baseline solve with no chips played.
The candidate gameweek with the largest improvement is the recommendation;
a chip whose best candidate does not beat baseline is a chip not worth
playing within this horizon yet.

This is deliberately a search over re-solves rather than a single MILP that
picks its own chip timing endogenously: chip usage is scarce (one of each
per half-season, two total for the whole season) and each choice interacts
with the whole rest of the plan in a way that is much easier to reason about,
and to explain to a user, as "gameweek 7 gains +14.2 points versus not
playing Bench Boost at all" than as an opaque binary decision variable
buried in a much larger model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .multiweek import MultiweekPlan, solve_multiweek
from .state import CHIP_NAMES, ManagerState

__all__ = ["ChipEvaluation", "evaluate_chip", "evaluate_all_chips"]


@dataclass(frozen=True)
class ChipEvaluation:
    """The result of testing one chip across a set of candidate gameweeks."""

    chip: str
    baseline_objective: float
    candidates: tuple[tuple[int, float], ...]  # (gw, objective_value), one per candidate
    best_gw: int
    best_gain: float

    @property
    def ranked(self) -> tuple[tuple[int, float], ...]:
        """Candidates sorted by descending gain over the no-chip baseline."""
        return tuple(sorted(self.candidates, key=lambda t: t[1], reverse=True))

    @property
    def worth_playing(self) -> bool:
        return self.best_gain > 0


def evaluate_chip(
    state: ManagerState,
    projections: pd.DataFrame,
    horizon_gws: Sequence[int],
    chip: str,
    candidate_gws: Sequence[int] | None = None,
    discount: float | Mapping[int, float] = 0.90,
    pool_per_position: int = 20,
    value_col: str = "ep",
    time_limit: float | None = 60.0,
    baseline: MultiweekPlan | None = None,
) -> ChipEvaluation:
    """Find the best gameweek in the horizon to play a single chip.

    Parameters mirror :func:`fplopt.optim.multiweek.solve_multiweek`.
    ``candidate_gws`` restricts which weeks are tried (default: every week in
    the horizon); pass a shorter list to cut the number of re-solves.
    ``baseline`` lets a caller supply an already-computed no-chip solve (see
    :func:`evaluate_all_chips`, which shares one across every chip).
    """
    if chip not in CHIP_NAMES:
        raise ValueError(f"Unknown chip {chip!r}; must be one of {CHIP_NAMES}")
    if state.chips_remaining.get(chip, 0) <= 0:
        raise ValueError(f"No {chip!r} remaining for this manager")

    sorted_gws = sorted(set(horizon_gws))
    candidates = sorted(set(candidate_gws)) if candidate_gws is not None else sorted_gws
    unknown = set(candidates) - set(sorted_gws)
    if unknown:
        raise ValueError(f"candidate_gws outside the horizon: {sorted(unknown)}")

    if baseline is None:
        baseline = solve_multiweek(
            state,
            projections,
            sorted_gws,
            discount=discount,
            chip_plan=None,
            pool_per_position=pool_per_position,
            value_col=value_col,
            time_limit=time_limit,
        )
    if not baseline.is_optimal:
        raise RuntimeError(f"Baseline (no-chip) solve did not reach optimality: {baseline.status}")

    results: list[tuple[int, float]] = []
    for g in candidates:
        plan = solve_multiweek(
            state,
            projections,
            sorted_gws,
            discount=discount,
            chip_plan={g: chip},
            pool_per_position=pool_per_position,
            value_col=value_col,
            time_limit=time_limit,
        )
        objective = plan.objective_value if plan.is_optimal else float("-inf")
        results.append((g, objective))

    best_gw, best_objective = max(results, key=lambda t: t[1])
    return ChipEvaluation(
        chip=chip,
        baseline_objective=baseline.objective_value,
        candidates=tuple(results),
        best_gw=best_gw,
        best_gain=best_objective - baseline.objective_value,
    )


def evaluate_all_chips(
    state: ManagerState,
    projections: pd.DataFrame,
    horizon_gws: Sequence[int],
    chips: Sequence[str] | None = None,
    candidate_gws: Sequence[int] | None = None,
    discount: float | Mapping[int, float] = 0.90,
    pool_per_position: int = 20,
    value_col: str = "ep",
    time_limit: float | None = 60.0,
) -> dict[str, ChipEvaluation]:
    """Run :func:`evaluate_chip` for every chip the manager still has.

    Shares a single no-chip baseline solve across every chip tested. Chips
    with zero uses remaining (``state.chips_remaining``) are skipped rather
    than raising, so this can safely be called with the full ``CHIP_NAMES``
    default regardless of what the manager has already used.
    """
    sorted_gws = sorted(set(horizon_gws))
    chip_list = list(chips) if chips is not None else list(CHIP_NAMES)
    baseline = solve_multiweek(
        state,
        projections,
        sorted_gws,
        discount=discount,
        chip_plan=None,
        pool_per_position=pool_per_position,
        value_col=value_col,
        time_limit=time_limit,
    )
    if not baseline.is_optimal:
        raise RuntimeError(f"Baseline (no-chip) solve did not reach optimality: {baseline.status}")

    out: dict[str, ChipEvaluation] = {}
    for chip in chip_list:
        if state.chips_remaining.get(chip, 0) <= 0:
            continue
        out[chip] = evaluate_chip(
            state,
            projections,
            sorted_gws,
            chip,
            candidate_gws=candidate_gws,
            discount=discount,
            pool_per_position=pool_per_position,
            value_col=value_col,
            time_limit=time_limit,
            baseline=baseline,
        )
    return out
