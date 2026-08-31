"""Pluggable objectives.

Every solver in this package (``squad``, ``multiweek``) maximises a single
named column (``value_col``, default ``"ep"``) on the projections frame. The
functions here compute that column so the choice of objective never has to
leak into the MILP-building code.

Two objectives are provided:

* :func:`plain_expected_points` -- maximise raw expected points. The default,
  and the right choice for a manager who just wants the highest expected
  score.
* :func:`rank_utility` -- a risk-adjusted objective for a manager trailing in
  a mini-league, who needs variance (differentials), not just expectation, to
  have a realistic chance of closing a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

__all__ = [
    "MiniLeagueContext",
    "plain_expected_points",
    "rank_utility",
    "recommend_lambda",
]


@dataclass(frozen=True)
class MiniLeagueContext:
    """Where the manager stands in the mini-league they actually care about.

    This is descriptive context for tuning ``lambda`` in :func:`rank_utility`
    (via :func:`recommend_lambda`) and for narrating the trade-off in
    :mod:`fplopt.optim.explain` -- it is not consumed directly by the solver.
    """

    rank: int
    n_managers: int
    points_behind_leader: int
    gws_remaining: int

    @property
    def is_trailing(self) -> bool:
        return self.rank > 1 and self.points_behind_leader > 0

    @property
    def urgency(self) -> float:
        """Points-per-gameweek the manager must find just to draw level.

        A crude but legible measure of how much risk is rational: needing to
        find 88 points over 36 gameweeks (2.4/gw) is a very different
        situation from needing 88 points over 3 gameweeks (29/gw).
        """
        if self.gws_remaining <= 0:
            return float("inf") if self.points_behind_leader > 0 else 0.0
        return self.points_behind_leader / self.gws_remaining


def plain_expected_points(projections: pd.DataFrame, out_col: str = "objective_score") -> pd.DataFrame:
    """Return ``projections`` with ``out_col`` set to raw expected points (``ep``).

    This is the objective for a manager who only cares about maximising their
    own total score, independent of any league context.
    """
    out = projections.copy()
    out[out_col] = out["ep"]
    return out


def recommend_lambda(context: MiniLeagueContext, base: float = 0.15, urgency_scale: float = 0.08) -> float:
    """A documented, tunable heuristic for the ``lambda`` in :func:`rank_utility`.

    ``lambda = base`` when not trailing at all, growing linearly with
    ``context.urgency`` (points needed per remaining gameweek) up to a cap of
    ``base + 1.0`` so a single bad gameweek's arithmetic can't send it to
    infinity. This is intentionally simple and meant to be overridden --
    pass an explicit ``lam`` to :func:`rank_utility` to bypass it entirely.

    With the default constants: a manager 88 points behind with 36 GWs left
    (urgency ~2.4/gw) gets ``lambda ~= 0.15 + 0.08*2.4 = 0.34``. The same gap
    with only 3 GWs left (urgency ~29.3/gw) saturates the cap at ``1.15``,
    reflecting that only large, achievable-by-luck differentials matter when
    time is nearly out.
    """
    if not context.is_trailing:
        return base
    return min(base + urgency_scale * context.urgency, base + 1.0)


def rank_utility(
    projections: pd.DataFrame,
    effective_ownership: Mapping[int, float],
    lam: float,
    out_col: str = "objective_score",
    default_eo: float = 0.0,
) -> pd.DataFrame:
    """Risk-adjusted objective: reward expectation plus low-ownership variance.

    ``objective_score = ep + lam * (1 - EO) * sd``

    where ``EO`` is the player's *league* effective ownership (fraction of
    the mini-league's total captained+owned exposure, in ``[0, 1]``, supplied
    by the caller -- this is a league-specific quantity, not something the
    projections model can know) and ``sd`` is the projection's own standard
    deviation of points, i.e. how much variance this pick could realistically
    swing a gameweek by.

    Why this shape: a low-EO player (a "differential") who does well moves
    the manager up the mini-league table relative to nearly everyone, while a
    high-EO player doing well is matched by most rivals and moves the table
    much less. Scaling the variance bonus by ``(1 - EO)`` captures exactly
    that -- it does nothing to a 100%-owned template player (``1-EO=0``,
    objective reduces to plain ``ep``) and adds up to ``lam * sd`` for a
    completely unowned one. ``sd`` (rather than a flat bonus) means the boost
    only applies to genuinely volatile players, not merely unpopular ones a
    projection model is simply confident will score little.

    This is a documented, linear, single-knob trade-off, not a black box:
    ``lam = 0`` recovers :func:`plain_expected_points` exactly, and
    :func:`recommend_lambda` gives a starting point for ``lam`` given a
    :class:`MiniLeagueContext`, but the caller should feel free to sweep it.

    Parameters
    ----------
    effective_ownership:
        ``{player_id: EO}`` with EO in ``[0, 1]``. Players missing from this
        mapping get ``default_eo`` (0.0 by default, i.e. treated as a maximal
        differential -- deliberately optimistic so an unmodelled player isn't
        silently penalised to look like a template pick).
    lam:
        The risk-appetite knob. 0 recovers plain expected points; larger
        values chase variance more aggressively. Typically obtained from
        :func:`recommend_lambda`, but always overridable.
    """
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    missing_sd = [c for c in ("ep", "sd") if c not in projections.columns]
    if missing_sd:
        raise ValueError(f"projections is missing required columns: {missing_sd}")

    out = projections.copy()
    eo = out["player_id"].map(lambda pid: effective_ownership.get(pid, default_eo))
    out["_eo"] = eo.clip(lower=0.0, upper=1.0)
    out[out_col] = out["ep"] + lam * (1.0 - out["_eo"]) * out["sd"]
    out = out.drop(columns="_eo")
    return out
