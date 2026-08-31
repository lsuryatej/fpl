"""Tests for fplopt.optim.squad: the single-gameweek MILP.

Each constraint is tested twice: once verifying the solver finds a legal
optimum under normal conditions, and once by making the constraint
impossible to satisfy and asserting the solver reports infeasibility rather
than silently returning an illegal squad.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fplopt.optim.squad import optimize_squad
from fplopt.optim.state import (
    BUDGET_TENTHS,
    DEF,
    FWD,
    GKP,
    MID,
    MIN_STARTING_DEF,
    MIN_STARTING_FWD,
    MIN_STARTING_GKP,
    SQUAD_TEAM_LIMIT,
    STARTING_XI_SIZE,
)


def test_finds_a_legal_optimal_squad(projections):
    gw1 = projections[projections["gw"] == 1]
    result = optimize_squad(gw1)
    assert result.is_optimal
    assert len(result.squad_ids) == 15
    assert len(result.starting_ids) == STARTING_XI_SIZE
    assert len(result.bench_ids) == 4
    assert result.captain_id in result.starting_ids
    assert result.vice_captain_id in result.starting_ids
    assert result.captain_id != result.vice_captain_id
    assert result.total_cost <= BUDGET_TENTHS


def test_squad_composition_is_exact(projections):
    gw1 = projections[projections["gw"] == 1]
    result = optimize_squad(gw1)
    pos_by_id = dict(zip(gw1["player_id"], gw1["position"]))
    counts = {GKP: 0, DEF: 0, MID: 0, FWD: 0}
    for pid in result.squad_ids:
        counts[pos_by_id[pid]] += 1
    assert counts == {GKP: 2, DEF: 5, MID: 5, FWD: 3}


def test_starting_xi_formation_is_legal(projections):
    gw1 = projections[projections["gw"] == 1]
    result = optimize_squad(gw1)
    pos_by_id = dict(zip(gw1["player_id"], gw1["position"]))
    start_counts = {GKP: 0, DEF: 0, MID: 0, FWD: 0}
    for pid in result.starting_ids:
        start_counts[pos_by_id[pid]] += 1
    assert start_counts[GKP] == MIN_STARTING_GKP
    assert start_counts[DEF] >= MIN_STARTING_DEF
    assert start_counts[FWD] >= MIN_STARTING_FWD
    assert sum(start_counts.values()) == STARTING_XI_SIZE


def test_no_more_than_three_per_club(projections):
    gw1 = projections[projections["gw"] == 1]
    result = optimize_squad(gw1)
    team_by_id = dict(zip(gw1["player_id"], gw1["team"]))
    counts: dict[int, int] = {}
    for pid in result.squad_ids:
        counts[team_by_id[pid]] = counts.get(team_by_id[pid], 0) + 1
    assert max(counts.values()) <= SQUAD_TEAM_LIMIT


def test_captain_is_the_highest_ep_starter_when_unconstrained(projections):
    # A sufficiently generous budget should let the solver captain its own
    # highest-ep starter (captaincy is a free doubling of whoever is best).
    gw1 = projections[projections["gw"] == 1]
    result = optimize_squad(gw1)
    ep_by_id = dict(zip(gw1["player_id"], gw1["ep"]))
    best_starter = max(result.starting_ids, key=lambda pid: ep_by_id[pid])
    assert result.captain_id == best_starter


def test_locked_ids_are_forced_into_the_squad(projections):
    gw1 = projections[projections["gw"] == 1]
    # Lock in a deliberately mediocre goalkeeper and check the solver obeys.
    gkp_ids = gw1[gw1["position"] == GKP]["player_id"]
    mediocre_gk = gw1[gw1["player_id"].isin(gkp_ids)].sort_values("ep").iloc[0]["player_id"]
    result = optimize_squad(gw1, locked_ids=[int(mediocre_gk)])
    assert result.is_optimal
    assert int(mediocre_gk) in result.squad_ids


def test_banned_ids_are_excluded(projections):
    gw1 = projections[projections["gw"] == 1]
    best_overall = gw1.sort_values("ep", ascending=False).iloc[0]["player_id"]
    result = optimize_squad(gw1, banned_ids=[int(best_overall)])
    assert result.is_optimal
    assert int(best_overall) not in result.squad_ids


def test_locked_and_banned_overlap_raises(projections):
    gw1 = projections[projections["gw"] == 1]
    pid = int(gw1.iloc[0]["player_id"])
    with pytest.raises(ValueError, match="both locked and banned"):
        optimize_squad(gw1, locked_ids=[pid], banned_ids=[pid])


# --------------------------------------------------------------------------
# Constraint-violation tests: force infeasibility, assert the solver refuses.
# --------------------------------------------------------------------------
def test_budget_infeasible_when_cheapest_legal_squad_exceeds_it(projections):
    gw1 = projections[projections["gw"] == 1]
    # The true cheapest legal squad costs at least the 15 cheapest-by-position
    # players' combined price; asking for a tiny fraction of that must fail.
    result = optimize_squad(gw1, budget=100)  # 10.0m for 15 players is impossible
    assert not result.is_optimal


def test_infeasible_when_too_few_clubs_to_satisfy_club_limit(projections):
    # Only 4 clubs present but the squad needs 15 players at <=3 per club:
    # 4*3=12 < 15, so no legal squad exists.
    gw1 = projections[(projections["gw"] == 1) & (projections["team"].isin([1, 2, 3, 4]))]
    result = optimize_squad(gw1)
    assert not result.is_optimal


def test_infeasible_when_a_position_has_too_few_candidates(projections):
    gw1 = projections[projections["gw"] == 1]
    # Keep only 1 goalkeeper in the pool; the squad needs exactly 2.
    gkp_ids = gw1[gw1["position"] == GKP]["player_id"].tolist()
    drop_all_but_one = gkp_ids[1:]
    trimmed = gw1[~gw1["player_id"].isin(drop_all_but_one)]
    result = optimize_squad(trimmed)
    assert not result.is_optimal


def test_infeasible_when_locks_violate_club_limit(projections):
    gw1 = projections[projections["gw"] == 1]
    # Lock in 4 players from the same club: violates the 3-per-club cap by construction.
    same_club_ids = gw1[gw1["team"] == 1]["player_id"].tolist()
    assert len(same_club_ids) >= 4, "fixture assumption: at least 4 players for team 1"
    result = optimize_squad(gw1, locked_ids=[int(p) for p in same_club_ids[:4]])
    assert not result.is_optimal


def test_infeasible_when_locks_violate_squad_composition(projections):
    gw1 = projections[projections["gw"] == 1]
    # Lock in 3 goalkeepers: the squad only has room for exactly 2.
    gkp_ids = gw1[gw1["position"] == GKP]["player_id"].tolist()
    assert len(gkp_ids) >= 3
    result = optimize_squad(gw1, locked_ids=[int(p) for p in gkp_ids[:3]])
    assert not result.is_optimal


def test_missing_required_column_raises(projections):
    gw1 = projections[projections["gw"] == 1].drop(columns=["price"])
    with pytest.raises(ValueError, match="missing required columns"):
        optimize_squad(gw1)


def test_duplicate_player_rows_raise(projections):
    gw1 = projections[projections["gw"] == 1]
    duped = pd.concat([gw1, gw1.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        optimize_squad(duped)
