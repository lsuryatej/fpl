"""Single-gameweek squad optimizer: MILP over one gameweek's projections.

Picks the best legal 15, the best legal starting XI within that 15, a
captain and vice-captain, subject to budget, per-club and formation limits.
This is the building block :mod:`fplopt.optim.multiweek` composes over a
horizon; it is also useful standalone (e.g. "what's the optimal squad to
build from scratch this week").

Bench order simplification
---------------------------
The MILP decides *which* 4 players are benched, the captain and vice. It
does not model bench order as a decision variable: auto-substitution only
matters in the rare case a starter doesn't play, and correctly valuing bench
order would require modelling the joint probability of each starter's
non-appearance and each bench player's eligibility to replace them position-
for-position -- a second-order effect next to who starts and who captains.
Bench order is instead assigned by a simple, documented heuristic after the
solve: outfield bench players are ordered by descending expected points
(``value_col``), and the bench goalkeeper is reported separately since its
"order" is fixed (it can only ever cover for the starting keeper).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import pulp

from .state import (
    BENCH_SIZE,
    BUDGET_TENTHS,
    DEF,
    FWD,
    GKP,
    MID,
    MIN_STARTING_DEF,
    MIN_STARTING_FWD,
    MIN_STARTING_GKP,
    POSITIONS,
    SQUAD_COMPOSITION,
    SQUAD_SIZE,
    SQUAD_TEAM_LIMIT,
    STARTING_XI_SIZE,
)

__all__ = ["SquadResult", "REQUIRED_COLUMNS", "optimize_squad"]

REQUIRED_COLUMNS = {"player_id", "position", "team", "price", "ep"}


@dataclass(frozen=True)
class SquadResult:
    """The outcome of a single-gameweek squad solve."""

    status: str
    objective_value: float
    squad_ids: tuple[int, ...]
    starting_ids: tuple[int, ...]
    bench_outfield_ids: tuple[int, ...]  # ordered, highest sub-priority first
    bench_gk_id: int
    captain_id: int
    vice_captain_id: int
    total_cost: int
    solver_seconds: float

    @property
    def bench_ids(self) -> tuple[int, ...]:
        """All 4 bench players, outfield sub-priority order then the GK."""
        return self.bench_outfield_ids + (self.bench_gk_id,)

    @property
    def is_optimal(self) -> bool:
        return self.status == "Optimal"


def _validate_projections(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"projections is missing required columns: {sorted(missing)}")
    if df["player_id"].duplicated().any():
        dupes = df.loc[df["player_id"].duplicated(), "player_id"].tolist()
        raise ValueError(f"projections has duplicate player_id rows: {dupes}")


def optimize_squad(
    projections: pd.DataFrame,
    budget: int = BUDGET_TENTHS,
    locked_ids: Iterable[int] = (),
    banned_ids: Iterable[int] = (),
    value_col: str = "ep",
    time_limit: float | None = 30.0,
) -> SquadResult:
    """Solve for the best 15/XI/captain for one gameweek.

    Parameters
    ----------
    projections:
        One row per candidate player for a single gameweek, with at least
        the columns in :data:`REQUIRED_COLUMNS`.
    budget:
        Total squad value cap, tenths of a million (default the full 100.0m).
    locked_ids:
        Players that must be in the final 15 (e.g. players already owned
        that should not be considered for sale in this solve).
    banned_ids:
        Players excluded from selection entirely.
    value_col:
        Column to maximise. Defaults to ``ep``; pass the output column of an
        :mod:`fplopt.optim.objective` transform to optimise something else.
    time_limit:
        Seconds given to the CBC solver. ``None`` for unlimited.
    """
    _validate_projections(projections)
    locked_ids = set(locked_ids)
    banned_ids = set(banned_ids)
    overlap = locked_ids & banned_ids
    if overlap:
        raise ValueError(f"Players cannot be both locked and banned: {sorted(overlap)}")

    df = projections[~projections["player_id"].isin(banned_ids)].reset_index(drop=True)
    missing_locks = locked_ids - set(df["player_id"])
    if missing_locks:
        raise ValueError(f"locked_ids not present in projections (or banned): {sorted(missing_locks)}")

    players = df["player_id"].tolist()
    position = dict(zip(df["player_id"], df["position"]))
    team = dict(zip(df["player_id"], df["team"]))
    price = dict(zip(df["player_id"], df["price"]))
    value = dict(zip(df["player_id"], df[value_col]))
    teams = sorted(set(team.values()))

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

    squad = {i: pulp.LpVariable(f"squad_{i}", cat="Binary") for i in players}
    start = {i: pulp.LpVariable(f"start_{i}", cat="Binary") for i in players}
    captain = {i: pulp.LpVariable(f"captain_{i}", cat="Binary") for i in players}
    vice = {i: pulp.LpVariable(f"vice_{i}", cat="Binary") for i in players}

    # Squad composition.
    prob += pulp.lpSum(squad.values()) == SQUAD_SIZE
    for pos in POSITIONS:
        pos_ids = [i for i in players if position[i] == pos]
        prob += pulp.lpSum(squad[i] for i in pos_ids) == SQUAD_COMPOSITION[pos]

    # Budget.
    prob += pulp.lpSum(price[i] * squad[i] for i in players) <= budget

    # Club limit.
    for t in teams:
        team_ids = [i for i in players if team[i] == t]
        prob += pulp.lpSum(squad[i] for i in team_ids) <= SQUAD_TEAM_LIMIT

    # Locks / bans (bans already excluded from `players`; locks forced in).
    for i in locked_ids:
        prob += squad[i] == 1

    # Starting XI.
    for i in players:
        prob += start[i] <= squad[i]
    prob += pulp.lpSum(start.values()) == STARTING_XI_SIZE
    gkp_ids = [i for i in players if position[i] == GKP]
    def_ids = [i for i in players if position[i] == DEF]
    fwd_ids = [i for i in players if position[i] == FWD]
    prob += pulp.lpSum(start[i] for i in gkp_ids) == MIN_STARTING_GKP
    prob += pulp.lpSum(start[i] for i in def_ids) >= MIN_STARTING_DEF
    prob += pulp.lpSum(start[i] for i in fwd_ids) >= MIN_STARTING_FWD

    # Captain / vice-captain: distinct starters.
    for i in players:
        prob += captain[i] <= start[i]
        prob += vice[i] <= start[i]
        prob += captain[i] + vice[i] <= 1
    prob += pulp.lpSum(captain.values()) == 1
    prob += pulp.lpSum(vice.values()) == 1

    # Objective: starters score once, the captain scores an extra time (2x total).
    prob += pulp.lpSum(value[i] * (start[i] + captain[i]) for i in players)

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    t0 = time.monotonic()
    prob.solve(solver)
    elapsed = time.monotonic() - t0

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return SquadResult(
            status=status,
            objective_value=float("nan"),
            squad_ids=(),
            starting_ids=(),
            bench_outfield_ids=(),
            bench_gk_id=-1,
            captain_id=-1,
            vice_captain_id=-1,
            total_cost=0,
            solver_seconds=elapsed,
        )

    squad_ids = tuple(i for i in players if squad[i].value() > 0.5)
    starting_ids = tuple(i for i in players if start[i].value() > 0.5)
    captain_id = next(i for i in players if captain[i].value() > 0.5)
    vice_id = next(i for i in players if vice[i].value() > 0.5)
    bench_ids = [i for i in squad_ids if i not in starting_ids]
    bench_gk_id = next(i for i in bench_ids if position[i] == GKP)
    bench_outfield = sorted(
        (i for i in bench_ids if i != bench_gk_id),
        key=lambda i: value[i],
        reverse=True,
    )
    total_cost = sum(price[i] for i in squad_ids)

    return SquadResult(
        status=status,
        objective_value=pulp.value(prob.objective),
        squad_ids=squad_ids,
        starting_ids=starting_ids,
        bench_outfield_ids=tuple(bench_outfield),
        bench_gk_id=bench_gk_id,
        captain_id=captain_id,
        vice_captain_id=vice_id,
        total_cost=total_cost,
        solver_seconds=elapsed,
    )
