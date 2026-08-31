"""Reasoning traces for recommended moves.

Every transfer or captaincy call this package produces can be handed to this
module, which explains it the way a manager actually reasons about a
decision: how many points it is worth, what the next-best alternative was
and by how much it lost out, which game-rule constraint was actually
limiting the choice, and what it changes for the gameweeks that follow.

Nothing here re-solves the whole problem from scratch except
``downstream_effect``, which needs one full re-solve with the specific
transfer forbidden to isolate its effect on later gameweeks -- everything
else is a small, targeted, cheap computation against the same projections
used to produce the plan being explained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .multiweek import GwPlan, MultiweekPlan, solve_multiweek
from .state import SQUAD_TEAM_LIMIT, ManagerState, selling_price

__all__ = ["TransferExplanation", "CaptainExplanation", "explain_transfer", "explain_captain"]


@dataclass(frozen=True)
class TransferExplanation:
    """A full reasoning trace for one transfer within a solved plan."""

    gw: int
    player_out: int
    player_in: int
    ep_delta: float
    """Expected-points difference this gameweek: ep(player_in) - ep(player_out)."""
    next_best_alternative: int | None
    """The best other same-position, affordable player not otherwise in the squad."""
    next_best_margin: float
    """ep(player_in) - ep(next_best_alternative) this gameweek. Positive means the chosen
    player really was the better pick; near zero means the decision was close; ``inf``
    means no affordable alternative existed at all."""
    binding_constraints: tuple[str, ...]
    """Which game rules were actually limiting this gameweek's choice (budget, club
    limit, free-transfer exhaustion), in plain language."""
    downstream_effect: dict[int, float]
    """{gw: points difference} between the actual plan and a re-solve with this specific
    transfer forbidden, for every gameweek in the horizon. Shows how much of the move's
    value is "this week" versus "unlocked later gameweeks"."""

    def summary(self) -> str:
        lines = [
            f"GW{self.gw}: {self.player_out} -> {self.player_in} "
            f"({'+' if self.ep_delta >= 0 else ''}{self.ep_delta:.2f} EP this week)"
        ]
        if self.next_best_alternative is not None:
            lines.append(
                f"  next best alternative: player {self.next_best_alternative} "
                f"(margin {self.next_best_margin:+.2f} EP in favour of the recommended player)"
            )
        else:
            lines.append("  no affordable alternative existed at this position/price point")
        for note in self.binding_constraints:
            lines.append(f"  binding: {note}")
        if self.downstream_effect:
            total = sum(v for v in self.downstream_effect.values() if v == v)  # skip NaN
            lines.append(f"  downstream effect across the horizon: {total:+.2f} EP versus not making this move")
        return "\n".join(lines)


@dataclass(frozen=True)
class CaptainExplanation:
    """A reasoning trace for one gameweek's captaincy choice."""

    gw: int
    captain_id: int
    vice_captain_id: int
    ep_as_captain: float
    """The extra expected points captaincy adds this gameweek (equal to the captain's own ep)."""
    next_best_alternative: int | None
    next_best_margin: float
    """ep(captain) - ep(next_best starter) this gameweek."""

    def summary(self) -> str:
        text = f"GW{self.gw}: captain {self.captain_id} (+{self.ep_as_captain:.2f} EP from the armband)"
        if self.next_best_alternative is not None:
            text += f"; next best was {self.next_best_alternative} (margin {self.next_best_margin:+.2f} EP)"
        return text


def _prior_bank(state: ManagerState, plan: MultiweekPlan, gw: int) -> int:
    sorted_plan_gws = sorted(p.gw for p in plan.gws)
    idx = sorted_plan_gws.index(gw)
    if idx == 0:
        return state.bank
    return plan.plan_for(sorted_plan_gws[idx - 1]).bank_after


def _binding_constraints(projections: pd.DataFrame, gw: int, gw_plan: GwPlan) -> tuple[str, ...]:
    notes: list[str] = []
    if gw_plan.bank_after <= 0:
        notes.append("budget: bank fully spent (GBP0.0m remaining after this gameweek's transfers)")

    gw_rows = projections[projections["gw"] == gw].set_index("player_id")
    team_of = gw_rows["team"].to_dict()
    counts: dict[int, int] = {}
    for pid in gw_plan.squad_ids:
        t = team_of.get(pid)
        if t is not None:
            counts[t] = counts.get(t, 0) + 1
    maxed_teams = sorted(t for t, c in counts.items() if c >= SQUAD_TEAM_LIMIT)
    if maxed_teams:
        notes.append(f"club limit: at the {SQUAD_TEAM_LIMIT}-player cap for team id(s) {maxed_teams}")

    n_transfers = len(gw_plan.transfers_in)
    if gw_plan.hits > 0:
        notes.append(
            f"free transfers: took a hit ({gw_plan.hits} x -4 = -{gw_plan.hit_cost}pts) -- "
            f"had {gw_plan.free_transfers_before} free, made {n_transfers} transfer(s)"
        )
    elif n_transfers > 0 and gw_plan.free_transfers_used >= gw_plan.free_transfers_before:
        notes.append(
            f"free transfers: used all {gw_plan.free_transfers_before} available, none rolled to next gameweek"
        )
    return tuple(notes)


def explain_transfer(
    plan: MultiweekPlan,
    state: ManagerState,
    projections: pd.DataFrame,
    gw: int,
    player_out: int,
    player_in: int,
    value_col: str = "ep",
    pool_per_position: int = 20,
    discount: float | Mapping[int, float] = 0.90,
    time_limit: float | None = 60.0,
) -> TransferExplanation:
    """Produce a full reasoning trace for one transfer recommended in ``plan``.

    ``plan`` must be the output of a :func:`fplopt.optim.multiweek.solve_multiweek`
    call against this same ``state``/``projections``, with ``player_in`` in
    ``plan.plan_for(gw).transfers_in`` and ``player_out`` in ``transfers_out``.
    """
    gw_plan = plan.plan_for(gw)
    if player_in not in gw_plan.transfers_in or player_out not in gw_plan.transfers_out:
        raise ValueError(f"gw {gw}: {player_in} in / {player_out} out is not part of this plan")

    gw_rows = projections[projections["gw"] == gw].set_index("player_id")
    ep_in = float(gw_rows.loc[player_in, value_col])
    ep_out = float(gw_rows.loc[player_out, value_col])
    ep_delta = ep_in - ep_out

    out_position = int(gw_rows.loc[player_out, "position"])
    price_out_now = int(gw_rows.loc[player_out, "price"])
    holding_out = state.holding(player_out)
    purchase_price_out = holding_out.purchase_price if holding_out is not None else price_out_now
    money_from_sale = selling_price(purchase_price_out, price_out_now)
    budget_freed = money_from_sale + _prior_bank(state, plan, gw)

    squad_after = set(gw_plan.squad_ids)
    same_pos = gw_rows[gw_rows["position"] == out_position]
    candidates = same_pos[(~same_pos.index.isin(squad_after)) & (same_pos["price"] <= budget_freed)]
    if not candidates.empty:
        best_alt_id = candidates[value_col].idxmax()
        best_alt_ep = float(candidates.loc[best_alt_id, value_col])
        next_best_alternative: int | None = int(best_alt_id)
        next_best_margin = ep_in - best_alt_ep
    else:
        next_best_alternative = None
        next_best_margin = float("inf")

    binding = _binding_constraints(projections, gw, gw_plan)

    horizon = [p.gw for p in plan.gws]
    chip_plan = {p.gw: p.chip for p in plan.gws if p.chip} or None
    forbidden = solve_multiweek(
        state,
        projections,
        horizon,
        discount=discount,
        chip_plan=chip_plan,
        pool_per_position=pool_per_position,
        value_col=value_col,
        time_limit=time_limit,
        banned_ids=[player_in],
    )
    downstream: dict[int, float] = {}
    if forbidden.is_optimal:
        forbidden_by_gw = {p.gw: p.raw_points for p in forbidden.gws}
        for p in plan.gws:
            if p.gw in forbidden_by_gw:
                downstream[p.gw] = p.raw_points - forbidden_by_gw[p.gw]

    return TransferExplanation(
        gw=gw,
        player_out=player_out,
        player_in=player_in,
        ep_delta=ep_delta,
        next_best_alternative=next_best_alternative,
        next_best_margin=next_best_margin,
        binding_constraints=binding,
        downstream_effect=downstream,
    )


def explain_captain(gw_plan: GwPlan, projections: pd.DataFrame, value_col: str = "ep") -> CaptainExplanation:
    """Produce a reasoning trace for one gameweek's captaincy choice."""
    gw_rows = projections[projections["gw"] == gw_plan.gw].set_index("player_id")
    ep_captain = float(gw_rows.loc[gw_plan.captain_id, value_col])
    starters = gw_rows.loc[list(gw_plan.starting_ids)]
    others = starters[starters.index != gw_plan.captain_id]
    if not others.empty:
        alt_id = others[value_col].idxmax()
        alt_ep = float(others.loc[alt_id, value_col])
        return CaptainExplanation(
            gw=gw_plan.gw,
            captain_id=gw_plan.captain_id,
            vice_captain_id=gw_plan.vice_captain_id,
            ep_as_captain=ep_captain,
            next_best_alternative=int(alt_id),
            next_best_margin=ep_captain - alt_ep,
        )
    return CaptainExplanation(
        gw=gw_plan.gw,
        captain_id=gw_plan.captain_id,
        vice_captain_id=gw_plan.vice_captain_id,
        ep_as_captain=ep_captain,
        next_best_alternative=None,
        next_best_margin=float("inf"),
    )
