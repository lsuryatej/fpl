"""The core solver: a single MILP over a rolling horizon of gameweeks.

One MILP jointly decides, for every gameweek in the horizon at once: which
players to buy and sell, whether to take a hit or bank a free transfer,
starting XI, and captaincy. This is what lets it find plans a week-by-week
("greedy") planner cannot: e.g. holding a transfer this week specifically so
two are available next week for a move that would otherwise cost a hit.

Key simplifications (each is a deliberate trade-off for tractability and
correctness, not an oversight -- see the sections below):

Price is constant over the horizon
-----------------------------------
Each player's price is taken once, from the earliest gameweek in the
horizon, and held constant for the rest of it. Real FPL prices do drift
(typically +/-1-3 tenths over 8 weeks for most players), a second-order
effect next to which players to buy and when to hit. The payoff is large:
under a constant price, a player bought and later sold *within* the horizon
always has exactly zero profit (bought and sold at the same price), so the
50%-of-profit sell-on fee never has to be modelled as MILP variables -- only
as a plain per-player constant computed once in Python via
``fplopt.optim.state.selling_price``. Without this assumption, exact fee
tracking needs a cost-basis variable and a ``max(0, profit)`` linearisation
per (player, gameweek), multiplying the model's variable count for a
second-order gain. The one place this matters is a player held *before* the
horizon started: their true purchase price (from ``ManagerState``) can
differ substantially from the current price, and that fee is still applied
exactly, because it too is just a constant lookup, not a variable.

Free-transfer and bank "state" variables carry only upper-bound constraints
-----------------------------------------------------------------------------
``ft[g]`` (free transfers available at gw g) and ``free_used[g]`` (free
transfers actually spent) are given only ``<=`` constraints (``ft[g] <= 5``,
``ft[g] <= previous_ft - previous_free_used + 1``, and similarly for
``free_used[g] <= ft[g]``, ``free_used[g] <= transfers_made[g]``), never a
forced equality via a min() linearisation with an auxiliary binary. This is
sound, not sloppy: both quantities only ever appear as upper bounds on
something the objective wants to maximise (more banked transfers can only
relax a future hit-avoidance constraint, never hurt it), so at any optimum
the solver has no incentive to report either below its true value -- if a
higher value would unlock a better plan, the solver is free to choose it,
because that value is within the feasible region already. The one wrinkle is
that CBC can return an arbitrary value within available slack when it truly
doesn't affect the objective (e.g. reporting 1 banked transfer when 3 were
technically available but never spent that path), so a tiny epsilon reward
on ``ft`` and ``free_used`` in the objective (``_TIE_BREAK_EPS``, many
orders of magnitude below a real fractional point) breaks that tie toward
the truthful maximum -- purely so ``explain`` can narrate an accurate
transfer-bank trajectory, without perturbing the real decision.

Bench order is not modelled
-----------------------------
As in :mod:`fplopt.optim.squad`, bench order is a post-solve heuristic
(descending expected points), not a MILP decision -- see that module's
docstring for the reasoning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pulp

from .state import (
    CHIP_NAMES,
    DEF,
    FWD,
    GKP,
    MAX_FREE_TRANSFERS_BANKED,
    MAX_TRANSFERS_PER_GW,
    MIN_STARTING_DEF,
    MIN_STARTING_FWD,
    MIN_STARTING_GKP,
    POSITIONS,
    SQUAD_COMPOSITION,
    SQUAD_SIZE,
    SQUAD_TEAM_LIMIT,
    STARTING_XI_SIZE,
    TRANSFER_HIT_COST,
    ManagerState,
    PlayerHolding,
    selling_price,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "GwPlan",
    "MultiweekPlan",
    "discount_factors",
    "filter_candidate_pool",
    "solve_multiweek",
    "greedy_baseline",
]

REQUIRED_COLUMNS = {"player_id", "gw", "position", "team", "price", "ep"}

# Negligible next to any real fractional expected-points value; exists only
# to break ties toward the truthful (tightest) free-transfer trajectory. See
# the module docstring.
_TIE_BREAK_EPS = 1e-6


def _val(x: object) -> float:
    """Return a plain float from either a constant or a solved PuLP variable."""
    if isinstance(x, (int, float)):
        return float(x)
    return float(x.value())


def _ival(x: object) -> int:
    return int(round(_val(x)))


@dataclass(frozen=True)
class GwPlan:
    """The decided plan for one gameweek within a horizon solve."""

    gw: int
    chip: str | None
    squad_ids: tuple[int, ...]
    starting_ids: tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    transfers_in: tuple[int, ...]
    transfers_out: tuple[int, ...]
    free_transfers_before: int
    free_transfers_used: int
    hits: int
    hit_cost: int
    bank_after: int
    raw_points: float  # undiscounted: this gw's points minus this gw's hit cost
    discounted_points: float


@dataclass(frozen=True)
class MultiweekPlan:
    """The outcome of a horizon solve (:func:`solve_multiweek` or :func:`greedy_baseline`)."""

    status: str
    objective_value: float
    gws: tuple[GwPlan, ...]
    solver_seconds: float
    pool_size: int
    n_binary_vars: int

    @property
    def is_optimal(self) -> bool:
        return self.status == "Optimal"

    @property
    def total_raw_points(self) -> float:
        """Sum of undiscounted per-gameweek points across the whole horizon, hits included."""
        return sum(p.raw_points for p in self.gws)

    def plan_for(self, gw: int) -> GwPlan:
        for p in self.gws:
            if p.gw == gw:
                return p
        raise KeyError(f"No plan for gameweek {gw} in this horizon")


def discount_factors(gws: Sequence[int], gamma: float) -> dict[int, float]:
    """Return ``{gw: gamma ** i}`` for ``gw`` at position ``i`` in sorted order.

    ``gamma < 1`` values later gameweeks less, reflecting that projections
    further out are less certain.
    """
    sorted_gws = sorted(gws)
    return {g: gamma**i for i, g in enumerate(sorted_gws)}


def filter_candidate_pool(
    projections: pd.DataFrame,
    current_squad_ids: Iterable[int],
    horizon_gws: Sequence[int],
    per_position: int = 20,
    value_col: str = "ep",
) -> pd.DataFrame:
    """Shrink the candidate pool so the horizon MILP stays tractable.

    Every currently-owned player is always kept (the solver must be able to
    represent keeping *or* selling them). Everyone else is filtered per
    position in two stages:

    1. **Dominance filter**: a player is dropped only if another same-position
       player is no more expensive AND matches or beats their expected points
       in *every single gameweek* of the horizon -- a true per-week Pareto
       dominance on ``(price, ep_week_1, ..., ep_week_H)``. Comparing by
       *summed* horizon EP alone is not a valid dominance test and was an
       earlier bug in this function: a player who is excellent in some weeks
       and merely average in others can have a lower total than a steadier
       rival while still being the better pick for the specific weeks they
       are strong in -- exactly the kind of timing a multi-week model is
       supposed to exploit. Excluding them on a total-EP shortcut measurably
       cost the solver points (see the multiweek-vs-greedy test), which is
       why this is a full per-week check, not a cheaper approximation.
    2. **Top-N cap**: of the (correctly) non-dominated survivors, keep only
       the ``per_position`` best by total horizon expected points, to bound
       pool size even when the true non-dominated frontier is itself large.
       This is the one place real information can still be discarded -- a
       spiky player ranked, say, 25th by total EP with ``per_position=20``
       is dropped even though nothing dominates them -- so treat
       ``per_position`` as the tractability/completeness dial and widen it
       if solve time allows.
    """
    horizon_gws = sorted(set(horizon_gws))
    df = projections[projections["gw"].isin(horizon_gws)]
    if df.empty:
        raise ValueError(f"No projections rows for horizon gameweeks {horizon_gws}")

    price_by_player = df.sort_values("gw").groupby("player_id")["price"].first().dropna()
    df = df[df["player_id"].isin(price_by_player.index)]
    total_value_by_player = df.groupby("player_id")[value_col].sum()
    position_by_player = df.drop_duplicates("player_id").set_index("player_id")["position"]

    current_squad_ids = set(current_squad_ids)
    missing_current = current_squad_ids - set(position_by_player.index)
    if missing_current:
        raise ValueError(
            f"Currently-owned players missing from projections for horizon {horizon_gws}: {sorted(missing_current)}"
        )

    keep_ids: set[int] = set(current_squad_ids)

    for _pos, pos_ids_index in position_by_player.groupby(position_by_player).groups.items():
        pos_ids = list(pos_ids_index)
        pivot = (
            df[df["player_id"].isin(pos_ids)]
            .pivot_table(index="player_id", columns="gw", values=value_col, fill_value=0.0)
            .reindex(columns=horizon_gws, fill_value=0.0)
        )
        ids = pivot.index.to_numpy()
        ep_mat = pivot.to_numpy(dtype=float)
        prices = price_by_player.reindex(ids).to_numpy(dtype=float)

        n = len(ids)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            weakly_dominates = (prices <= prices[i]) & (ep_mat >= ep_mat[i]).all(axis=1)
            weakly_dominates[i] = False
            if not weakly_dominates.any():
                continue
            strictly_better = (prices < prices[i]) | (ep_mat > ep_mat[i]).any(axis=1)
            if (weakly_dominates & strictly_better).any():
                dominated[i] = True

        survivor_ids = ids[~dominated]
        ranked = total_value_by_player.reindex(survivor_ids).sort_values(ascending=False)
        keep_ids.update(int(pid) for pid in ranked.head(per_position).index)

    return df[df["player_id"].isin(keep_ids)].copy()


def _validate_chip_plan(chip_plan: Mapping[int, str], sorted_gws: list[int], state: ManagerState) -> None:
    for g, chip in chip_plan.items():
        if g not in sorted_gws:
            raise ValueError(f"chip_plan references gw {g}, outside the horizon {sorted_gws}")
        if chip not in CHIP_NAMES:
            raise ValueError(f"Unknown chip {chip!r}; must be one of {CHIP_NAMES}")
    used_counts: dict[str, int] = {}
    for chip in chip_plan.values():
        used_counts[chip] = used_counts.get(chip, 0) + 1
    for chip, count in used_counts.items():
        remaining = state.chips_remaining.get(chip, 0)
        if count > remaining:
            raise ValueError(f"chip_plan uses {chip!r} {count} time(s) but only {remaining} remaining")


def solve_multiweek(
    state: ManagerState,
    projections: pd.DataFrame,
    horizon_gws: Sequence[int],
    discount: float | Mapping[int, float] = 0.90,
    chip_plan: Mapping[int, str] | None = None,
    pool_per_position: int = 20,
    value_col: str = "ep",
    time_limit: float | None = 120.0,
    locked_ids: Iterable[int] = (),
    banned_ids: Iterable[int] = (),
) -> MultiweekPlan:
    """Jointly optimise transfers, hits, banking, XI and captaincy over a horizon.

    Parameters
    ----------
    state:
        The manager's real situation as of the first gameweek in the horizon.
    projections:
        Rows for every candidate player at every gameweek in the horizon,
        with at least the columns in :data:`REQUIRED_COLUMNS`.
    horizon_gws:
        The gameweeks to plan for, e.g. ``range(3, 11)`` for an 8-GW horizon
        starting at gw3.
    discount:
        Either a constant per-step discount factor (see
        :func:`discount_factors`) or an explicit ``{gw: factor}`` mapping.
    chip_plan:
        ``{gw: chip_name}`` forcing a specific chip to be active in specific
        gameweeks (at most one chip per gameweek). Leave empty to solve with
        no chips played -- see :mod:`fplopt.optim.chips` for searching over
        candidate chip weeks.
    pool_per_position:
        Passed to :func:`filter_candidate_pool`.
    value_col:
        Column to maximise; swap in an :mod:`fplopt.optim.objective` output
        column to change what "best" means.
    locked_ids, banned_ids:
        Players forced into, or excluded from, every gameweek's squad.
    """
    chip_plan = dict(chip_plan or {})
    sorted_gws = sorted(set(horizon_gws))
    if not sorted_gws:
        raise ValueError("horizon_gws must be non-empty")
    if sorted_gws[0] < state.as_of_gw:
        raise ValueError(f"horizon starts at gw {sorted_gws[0]}, before the manager's state gw {state.as_of_gw}")
    _validate_chip_plan(chip_plan, sorted_gws, state)

    if isinstance(discount, Mapping):
        discount_map = dict(discount)
        missing_d = set(sorted_gws) - set(discount_map)
        if missing_d:
            raise ValueError(f"discount mapping missing gameweeks: {sorted(missing_d)}")
    else:
        discount_map = discount_factors(sorted_gws, float(discount))

    locked_ids = set(locked_ids)
    banned_ids = set(banned_ids)
    overlap = locked_ids & banned_ids
    if overlap:
        raise ValueError(f"Players cannot be both locked and banned: {sorted(overlap)}")

    current_squad_ids = set(state.squad_ids)
    pool_df = filter_candidate_pool(
        projections, current_squad_ids, sorted_gws, per_position=pool_per_position, value_col=value_col
    )
    pool_df = pool_df[~pool_df["player_id"].isin(banned_ids)]
    players = sorted(pool_df["player_id"].unique().tolist())
    missing_locks = locked_ids - set(players)
    if missing_locks:
        raise ValueError(f"locked_ids not present in the (filtered, un-banned) pool: {sorted(missing_locks)}")
    missing_squad = current_squad_ids - set(players) - banned_ids
    if missing_squad:
        raise ValueError(f"Currently-owned players fell out of the candidate pool: {sorted(missing_squad)}")

    position: dict[int, int] = {}
    team: dict[int, int] = {}
    price: dict[int, int] = {}
    ep: dict[tuple[int, int], float] = {}
    for row in pool_df.itertuples(index=False):
        pid = int(row.player_id)
        position[pid] = int(row.position)
        team[pid] = int(row.team)
        ep[(pid, int(row.gw))] = float(getattr(row, value_col))
        if pid not in price:
            price[pid] = int(row.price)
    for pid in players:
        for g in sorted_gws:
            ep.setdefault((pid, g), 0.0)

    holdings_by_id = {h.player_id: h for h in state.squad}
    purchase_price = {pid: (holdings_by_id[pid].purchase_price if pid in holdings_by_id else price[pid]) for pid in players}
    sell_value = {pid: selling_price(purchase_price[pid], price[pid]) for pid in players}
    prev_squad = {pid: (1 if pid in holdings_by_id else 0) for pid in players}
    teams_present = sorted(set(team.values()))

    prob = pulp.LpProblem("fpl_multiweek", pulp.LpMaximize)

    squad = {(pid, g): pulp.LpVariable(f"squad_{pid}_{g}", cat="Binary") for pid in players for g in sorted_gws}
    start = {(pid, g): pulp.LpVariable(f"start_{pid}_{g}", cat="Binary") for pid in players for g in sorted_gws}
    captain = {(pid, g): pulp.LpVariable(f"captain_{pid}_{g}", cat="Binary") for pid in players for g in sorted_gws}
    vice = {(pid, g): pulp.LpVariable(f"vice_{pid}_{g}", cat="Binary") for pid in players for g in sorted_gws}

    buy: dict[tuple[int, int], pulp.LpVariable] = {}
    sell: dict[tuple[int, int], pulp.LpVariable] = {}
    for g in sorted_gws:
        if chip_plan.get(g) == "free_hit":
            continue
        for pid in players:
            buy[(pid, g)] = pulp.LpVariable(f"buy_{pid}_{g}", cat="Binary")
            sell[(pid, g)] = pulp.LpVariable(f"sell_{pid}_{g}", cat="Binary")

    bank: dict[int, pulp.LpVariable] = {}
    for g in sorted_gws:
        if chip_plan.get(g) == "free_hit":
            continue
        bank[g] = pulp.LpVariable(f"bank_{g}", lowBound=0)

    ft: dict[int, pulp.LpVariable] = {
        g: pulp.LpVariable(f"ft_{g}", lowBound=0, upBound=MAX_FREE_TRANSFERS_BANKED, cat="Integer")
        for g in sorted_gws[1:]
    }
    free_used: dict[int, pulp.LpVariable] = {}
    hits: dict[int, pulp.LpVariable] = {}
    for g in sorted_gws:
        if chip_plan.get(g) in ("wildcard", "free_hit"):
            continue
        free_used[g] = pulp.LpVariable(f"free_used_{g}", lowBound=0, upBound=MAX_FREE_TRANSFERS_BANKED, cat="Integer")
        hits[g] = pulp.LpVariable(f"hits_{g}", lowBound=0, upBound=MAX_TRANSFERS_PER_GW, cat="Integer")

    def ft_value(g: int):
        return state.free_transfers if g == sorted_gws[0] else ft[g]

    # ---- formation & squad-composition constraints, every gameweek ----
    for g in sorted_gws:
        prob += pulp.lpSum(squad[(pid, g)] for pid in players) == SQUAD_SIZE
        for pos in POSITIONS:
            pos_ids = [pid for pid in players if position[pid] == pos]
            prob += pulp.lpSum(squad[(pid, g)] for pid in pos_ids) == SQUAD_COMPOSITION[pos]
        for t in teams_present:
            t_ids = [pid for pid in players if team[pid] == t]
            prob += pulp.lpSum(squad[(pid, g)] for pid in t_ids) <= SQUAD_TEAM_LIMIT
        for pid in players:
            prob += start[(pid, g)] <= squad[(pid, g)]
        prob += pulp.lpSum(start[(pid, g)] for pid in players) == STARTING_XI_SIZE
        gkp_ids = [pid for pid in players if position[pid] == GKP]
        def_ids = [pid for pid in players if position[pid] == DEF]
        fwd_ids = [pid for pid in players if position[pid] == FWD]
        prob += pulp.lpSum(start[(pid, g)] for pid in gkp_ids) == MIN_STARTING_GKP
        prob += pulp.lpSum(start[(pid, g)] for pid in def_ids) >= MIN_STARTING_DEF
        prob += pulp.lpSum(start[(pid, g)] for pid in fwd_ids) >= MIN_STARTING_FWD

        is_bench_boost = chip_plan.get(g) == "bench_boost"
        for pid in players:
            cap_upper = squad[(pid, g)] if is_bench_boost else start[(pid, g)]
            prob += captain[(pid, g)] <= cap_upper
            prob += vice[(pid, g)] <= start[(pid, g)]
            prob += captain[(pid, g)] + vice[(pid, g)] <= 1
        prob += pulp.lpSum(captain[(pid, g)] for pid in players) == 1
        prob += pulp.lpSum(vice[(pid, g)] for pid in players) == 1

    for pid in locked_ids:
        for g in sorted_gws:
            prob += squad[(pid, g)] == 1

    # ---- persistence chain: squad continuity + bank recursion ----
    # Free-hit weeks are excluded from the chain -- the squad picked that
    # week is a one-off, and the gameweek after it continues from whatever
    # came *before* the free hit, exactly as the real chip reverts.
    persistent_pred: dict[int, int | None] = {}
    last_chain_gw: int | None = None
    for g in sorted_gws:
        persistent_pred[g] = last_chain_gw
        if chip_plan.get(g) != "free_hit":
            last_chain_gw = g

    def squad_at(pid: int, pred_gw: int | None):
        return prev_squad[pid] if pred_gw is None else squad[(pid, pred_gw)]

    def bank_at(pred_gw: int | None):
        return state.bank if pred_gw is None else bank[pred_gw]

    for g in sorted_gws:
        pred = persistent_pred[g]
        if chip_plan.get(g) == "free_hit":
            available = bank_at(pred) + pulp.lpSum(sell_value[pid] * squad_at(pid, pred) for pid in players)
            prob += pulp.lpSum(price[pid] * squad[(pid, g)] for pid in players) <= available
            continue
        for pid in players:
            prob += squad[(pid, g)] == squad_at(pid, pred) + buy[(pid, g)] - sell[(pid, g)]
            prob += buy[(pid, g)] <= 1 - squad_at(pid, pred)
            prob += sell[(pid, g)] <= squad_at(pid, pred)
        proceeds = pulp.lpSum(sell_value[pid] * sell[(pid, g)] for pid in players)
        costs = pulp.lpSum(price[pid] * buy[(pid, g)] for pid in players)
        prob += bank[g] == bank_at(pred) + proceeds - costs

    # ---- transfers / hits / free-transfer accounting ----
    for g in sorted_gws:
        chip = chip_plan.get(g)
        if chip == "free_hit":
            continue
        transfers_expr = pulp.lpSum(buy[(pid, g)] for pid in players)
        prob += transfers_expr <= MAX_TRANSFERS_PER_GW
        if chip == "wildcard":
            continue  # unlimited, free, and does not touch the FT bank
        prob += free_used[g] <= ft_value(g)
        prob += free_used[g] <= transfers_expr
        prob += hits[g] == transfers_expr - free_used[g]

    # ---- free-transfer bank roll-forward ----
    for g_prev, g_cur in zip(sorted_gws, sorted_gws[1:]):
        chip_prev = chip_plan.get(g_prev)
        free_used_prev = 0 if chip_prev in ("wildcard", "free_hit") else free_used[g_prev]
        raw = ft_value(g_prev) - free_used_prev + 1
        prob += ft[g_cur] <= MAX_FREE_TRANSFERS_BANKED
        prob += ft[g_cur] <= raw

    # ---- objective ----
    total_terms = []
    tie_break = []
    for g in sorted_gws:
        chip = chip_plan.get(g)
        if chip == "triple_captain":
            gw_points = pulp.lpSum(ep[(pid, g)] * start[(pid, g)] for pid in players) + 2 * pulp.lpSum(
                ep[(pid, g)] * captain[(pid, g)] for pid in players
            )
        elif chip == "bench_boost":
            gw_points = pulp.lpSum(ep[(pid, g)] * squad[(pid, g)] for pid in players) + pulp.lpSum(
                ep[(pid, g)] * captain[(pid, g)] for pid in players
            )
        else:
            gw_points = pulp.lpSum(ep[(pid, g)] * (start[(pid, g)] + captain[(pid, g)]) for pid in players)
        hit_cost = 0 if chip in ("wildcard", "free_hit") else TRANSFER_HIT_COST * hits[g]
        total_terms.append(discount_map[g] * (gw_points - hit_cost))
        if g in free_used:
            tie_break.append(free_used[g])
        if g in ft:
            tie_break.append(ft[g])
    prob += pulp.lpSum(total_terms) + _TIE_BREAK_EPS * pulp.lpSum(tie_break)

    # PuLP normalises `cat="Binary"` variables to `cat="Integer", lowBound=0,
    # upBound=1` internally, so introspecting `.cat` after the fact can't
    # distinguish them from other integer variables -- count directly from
    # the dicts that were built as binary instead.
    n_binaries = len(squad) + len(start) + len(captain) + len(vice) + len(buy) + len(sell)

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    t0 = time.monotonic()
    prob.solve(solver)
    elapsed = time.monotonic() - t0
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        return MultiweekPlan(
            status=status,
            objective_value=float("nan"),
            gws=(),
            solver_seconds=elapsed,
            pool_size=len(players),
            n_binary_vars=n_binaries,
        )

    gw_plans: list[GwPlan] = []
    for g in sorted_gws:
        pred = persistent_pred[g]
        chip = chip_plan.get(g)
        if pred is None:
            pred_squad_ids = current_squad_ids
        else:
            pred_squad_ids = {pid for pid in players if squad[(pid, pred)].value() > 0.5}
        cur_squad_ids = tuple(sorted(pid for pid in players if squad[(pid, g)].value() > 0.5))
        cur_start_ids = tuple(sorted(pid for pid in players if start[(pid, g)].value() > 0.5))
        cap_id = next(pid for pid in players if captain[(pid, g)].value() > 0.5)
        vice_id = next(pid for pid in players if vice[(pid, g)].value() > 0.5)

        if chip == "free_hit":
            transfers_in = tuple(sorted(set(cur_squad_ids) - pred_squad_ids))
            transfers_out = tuple(sorted(pred_squad_ids - set(cur_squad_ids)))
            ft_before = _ival(ft_value(g))
            ft_used = 0
            hit_count = 0
            hit_cost_val = 0
            bank_after = _ival(bank_at(pred))
        else:
            transfers_in = tuple(sorted(pid for pid in players if buy[(pid, g)].value() > 0.5))
            transfers_out = tuple(sorted(pid for pid in players if sell[(pid, g)].value() > 0.5))
            ft_before = _ival(ft_value(g))
            if chip == "wildcard":
                ft_used = 0
                hit_count = 0
                hit_cost_val = 0
            else:
                ft_used = _ival(free_used[g])
                hit_count = _ival(hits[g])
                hit_cost_val = TRANSFER_HIT_COST * hit_count
            bank_after = _ival(bank[g])

        if chip == "triple_captain":
            raw_points = sum(ep[(pid, g)] for pid in cur_start_ids) + 2 * ep[(cap_id, g)]
        elif chip == "bench_boost":
            raw_points = sum(ep[(pid, g)] for pid in cur_squad_ids) + ep[(cap_id, g)]
        else:
            raw_points = sum(ep[(pid, g)] for pid in cur_start_ids) + ep[(cap_id, g)]
        raw_points -= hit_cost_val

        gw_plans.append(
            GwPlan(
                gw=g,
                chip=chip,
                squad_ids=cur_squad_ids,
                starting_ids=cur_start_ids,
                captain_id=cap_id,
                vice_captain_id=vice_id,
                transfers_in=transfers_in,
                transfers_out=transfers_out,
                free_transfers_before=ft_before,
                free_transfers_used=ft_used,
                hits=hit_count,
                hit_cost=hit_cost_val,
                bank_after=bank_after,
                raw_points=float(raw_points),
                discounted_points=float(discount_map[g] * raw_points),
            )
        )

    return MultiweekPlan(
        status=status,
        objective_value=pulp.value(prob.objective),
        gws=tuple(gw_plans),
        solver_seconds=elapsed,
        pool_size=len(players),
        n_binary_vars=n_binaries,
    )


def _advance_state(cur_state: ManagerState, projections: pd.DataFrame, plan: GwPlan, gw: int) -> ManagerState:
    """Build the next :class:`ManagerState` after realising one gameweek's plan.

    Used by :func:`greedy_baseline` to chain single-gameweek solves; kept
    private since a caller doing a real multi-week solve never needs to
    "advance" -- the whole horizon is decided in one solve.
    """
    rows = projections[(projections["gw"] == gw) & (projections["player_id"].isin(plan.squad_ids))]
    info = rows.set_index("player_id")
    prior_by_id = {h.player_id: h for h in cur_state.squad}
    holdings = []
    for pid in plan.squad_ids:
        row = info.loc[pid]
        price_now = int(row["price"])
        prior = prior_by_id.get(pid)
        purchase_price = prior.purchase_price if (prior is not None and pid not in plan.transfers_in) else price_now
        holdings.append(
            PlayerHolding(
                player_id=int(pid),
                position=int(row["position"]),
                team=int(row["team"]),
                purchase_price=purchase_price,
                current_price=price_now,
                is_captain=(pid == plan.captain_id),
                is_vice_captain=(pid == plan.vice_captain_id),
                multiplier=2 if pid == plan.captain_id else (1 if pid in plan.starting_ids else 0),
            )
        )
    ft_next = min(MAX_FREE_TRANSFERS_BANKED, cur_state.free_transfers - plan.free_transfers_used + 1)
    return ManagerState(
        entry_id=cur_state.entry_id,
        as_of_gw=gw + 1,
        bank=plan.bank_after,
        free_transfers=max(1, ft_next),
        chips_remaining=dict(cur_state.chips_remaining),
        squad=tuple(holdings),
    )


def greedy_baseline(
    state: ManagerState,
    projections: pd.DataFrame,
    horizon_gws: Sequence[int],
    value_col: str = "ep",
    pool_per_position: int = 20,
    time_limit: float | None = 30.0,
) -> MultiweekPlan:
    """A one-week-at-a-time baseline: solve each gameweek in isolation.

    At every gameweek, transfers are chosen to maximise *that gameweek's*
    score alone (points minus any hit taken this week), with no knowledge of
    -- or credit for -- how banking a free transfer might pay off later.
    This is the standard "naive" comparator multi-week planning has to beat:
    it always takes an immediately-positive transfer even when holding it
    would have funded a much better move (or avoided a hit) further out.

    Implemented by calling :func:`solve_multiweek` with a single-gameweek
    horizon at each step, each time starting from the real state left behind
    by the previous week's realised decision. No chips are considered.
    """
    sorted_gws = sorted(set(horizon_gws))
    cur_state = state
    gw_plans: list[GwPlan] = []
    total_solver_seconds = 0.0
    max_pool_size = 0
    for g in sorted_gws:
        step = solve_multiweek(
            cur_state,
            projections,
            [g],
            discount=1.0,
            chip_plan=None,
            pool_per_position=pool_per_position,
            value_col=value_col,
            time_limit=time_limit,
        )
        total_solver_seconds += step.solver_seconds
        max_pool_size = max(max_pool_size, step.pool_size)
        if step.status != "Optimal":
            return MultiweekPlan(
                status=step.status,
                objective_value=float("nan"),
                gws=tuple(gw_plans),
                solver_seconds=total_solver_seconds,
                pool_size=max_pool_size,
                n_binary_vars=0,
            )
        plan = step.gws[0]
        gw_plans.append(plan)
        cur_state = _advance_state(cur_state, projections, plan, g)
    return MultiweekPlan(
        status="Optimal",
        objective_value=sum(p.raw_points for p in gw_plans),
        gws=tuple(gw_plans),
        solver_seconds=total_solver_seconds,
        pool_size=max_pool_size,
        n_binary_vars=0,
    )
