"""Gap, overlap and multiplier-settlement tests against the toy league."""

from __future__ import annotations

import copy

import pytest

from fplopt.league.fetch import effective_multipliers, gw_score, verify_totals
from fplopt.league.gap import catchup_plan, gap_targets, overlap_table, squad_overlap

HAALAND, SALAH, SAKA, PALMER, WATKINS, ISAK, RAYA, SELS = 1, 2, 3, 4, 5, 6, 7, 8


# ------------------------------------------------------------------ gap maths


def test_gap_targets(toy_league):
    t = {g.label: g for g in gap_targets(toy_league, remaining_gws=36)}
    # User is on 17, leader on 29.
    assert t["1st"].total == 29
    assert t["1st"].gap == 12
    assert t["1st"].per_gw == pytest.approx(12 / 36)
    # Median of [29, 24, 21, 17] is 22.5 -> int() -> 22.
    assert t["median"].gap == 22 - 17
    # One place up is rank 3 (103, on 21).
    assert t["3th (one place up)"].gap == 4


def test_rungs_beyond_the_league_size_clamp_to_last_place(toy_league):
    t = {g.label: g for g in gap_targets(toy_league, remaining_gws=36)}
    # Only 4 entries, so "top-5" and "top-10" clamp to 4th (the user).
    assert t["top-5 (5th)"].total == 17
    assert t["top-10 (10th)"].total == 17


def test_per_gw_edge_scales_with_horizon(toy_league):
    a = {g.label: g for g in gap_targets(toy_league, remaining_gws=36)}["1st"]
    b = {g.label: g for g in gap_targets(toy_league, remaining_gws=18)}["1st"]
    assert b.per_gw == pytest.approx(2 * a.per_gw)


def test_catchup_plan_rates(toy_league):
    cp = catchup_plan(toy_league, remaining_gws=36)
    assert cp.user_total == 17
    assert cp.user_rank == 4
    assert cp.user_ppg == pytest.approx(17.0)  # one gameweek played
    assert cp.leader_ppg == pytest.approx(29.0)
    assert cp.league_ppg == pytest.approx((29 + 24 + 21 + 17) / 4)
    assert cp.required_edge_vs_leader == pytest.approx(12 / 36)
    assert cp.required_ppg_for_first == pytest.approx(29 + 12 / 36)


# -------------------------------------------------------------- squad overlap


def test_overlap_counts_shared_players(toy_league):
    # 101 {1,4,5,8,6} vs 102 {1,2,3,7,6} -> shared {Haaland, Isak}
    o = squad_overlap(toy_league, 102, remaining_gws=36)
    assert o.shared_squad == 2
    assert o.shared_names == ["Haaland", "Isak"]
    assert o.shared_pct == pytest.approx(40.0)
    # Only Haaland is in both starting XIs.
    assert o.shared_xi == 1
    assert o.live_edge_players == 3


def test_dead_weight_uses_the_minimum_multiplier(toy_league):
    # vs 102: user mass 2+1+1+1+0 = 5; overlap mass min(2,1) = 1 -> 20%.
    o102 = squad_overlap(toy_league, 102, remaining_gws=36)
    assert o102.dead_weight_pct == pytest.approx(20.0)
    # vs 103: same 2 shared players, but 103 also captains Haaland, so the
    # neutralised mass doubles to min(2,2) = 2 -> 40%.
    o103 = squad_overlap(toy_league, 103, remaining_gws=36)
    assert o103.shared_squad == 2
    assert o103.dead_weight_pct == pytest.approx(40.0)


def test_captaincy_divergence_creates_separation_on_a_shared_player(toy_league):
    """Two managers owning the same player still separate if only one captains."""
    a = squad_overlap(toy_league, 102, remaining_gws=36)  # 102 does not captain Haaland
    b = squad_overlap(toy_league, 103, remaining_gws=36)  # 103 does
    assert a.dead_weight_pct < b.dead_weight_pct


def test_a_perfectly_shared_squad_is_100_percent_dead_weight(toy_league):
    data = copy.deepcopy(toy_league)
    data.picks[102][1] = copy.deepcopy(data.picks[101][1])
    o = squad_overlap(data, 102, remaining_gws=36)
    assert o.shared_squad == 5
    assert o.dead_weight_pct == pytest.approx(100.0)
    assert o.live_edge_players == 0


def test_a_disjoint_squad_is_zero_dead_weight(toy_league):
    o = squad_overlap(toy_league, 104, remaining_gws=36)
    # 101 {1,4,5,8,6} vs 104 {2,3,5,7,8} -> shared {Watkins, Sels}
    assert o.shared_names == ["Sels", "Watkins"]
    # 104 benches Sels, so only Watkins is neutralised: min(1,1)=1 of mass 5.
    assert o.dead_weight_pct == pytest.approx(20.0)


def test_overlap_table_covers_every_rival(toy_league):
    rows = overlap_table(toy_league, remaining_gws=36)
    assert {r.entry for r in rows} == {102, 103, 104}
    assert all(r.entry != toy_league.user_entry for r in rows)


# ------------------------------------------------- multiplier settlement rules


def _blank_captain(data):
    """Make 101's captain (Haaland) a zero-minute blank."""
    data.live[1][HAALAND] = {"total_points": 0, "minutes": 0}
    return data


def test_vice_captain_takes_over_when_the_captain_blanks(toy_league):
    data = _blank_captain(copy.deepcopy(toy_league))
    m = effective_multipliers(data, 101, 1)
    assert m[HAALAND] == 1  # armband removed
    assert m[PALMER] == 2  # vice-captain promoted
    # 0 + 2*2 + 8 + 1 = 13
    assert gw_score(data, 101, 1) == 13


def test_no_vice_promotion_while_the_gameweek_is_still_running(toy_league):
    data = _blank_captain(copy.deepcopy(toy_league))
    data.fixtures[1] = [
        {"team_h": 1, "team_a": 2, "finished": False, "finished_provisional": False}
    ]
    assert data.is_gw_final(1) is False
    m = effective_multipliers(data, 101, 1)
    # Deadline multipliers stand: a zero-minute player has simply not kicked off.
    assert m[HAALAND] == 2
    assert m[PALMER] == 1


def test_automatic_subs_are_applied(toy_league):
    data = copy.deepcopy(toy_league)
    # Watkins does not play; Isak comes off the bench.
    data.live[1][WATKINS] = {"total_points": 0, "minutes": 0}
    data.live[1][ISAK] = {"total_points": 7, "minutes": 90}
    data.picks[101][1]["automatic_subs"] = [{"element_out": WATKINS, "element_in": ISAK}]
    m = effective_multipliers(data, 101, 1)
    assert m[WATKINS] == 0
    assert m[ISAK] == 1
    # 2*3 + 2 + 0 + 1 + 7 = 16
    assert gw_score(data, 101, 1) == 16


def test_verify_totals_reports_a_mismatch_when_the_data_disagrees(toy_league):
    data = copy.deepcopy(toy_league)
    data.standings[0]["total"] += 5
    v = verify_totals(data)
    assert v["n_match"] == 3
    assert v["match_rate"] == pytest.approx(0.75)
    assert v["mismatches"][0]["delta"] == -5
