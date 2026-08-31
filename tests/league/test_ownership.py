"""Hand-checkable tests for the ownership / EO / swing maths.

Every expected number in here is worked out on paper in ``conftest.py``'s
docstring. Nothing is copied from a program run.
"""

from __future__ import annotations

import math

import pytest

from fplopt.league.fetch import effective_multipliers, gw_score, recomputed_total, verify_totals
from fplopt.league.ownership import (
    captaincy_table,
    ownership_table,
    player_swings,
    threat_weights,
    user_position,
)

HAALAND, SALAH, SAKA, PALMER, WATKINS, ISAK, RAYA, SELS = 1, 2, 3, 4, 5, 6, 7, 8


def rows_by_element(data, **kw):
    return {r.element: r for r in ownership_table(data, **kw)}


# ----------------------------------------------------------------- ownership


def test_league_ownership_counts(toy_league):
    r = rows_by_element(toy_league)
    # Haaland is in 3 of the 4 squads; Palmer in 2.
    assert r[HAALAND].owners == 3
    assert r[HAALAND].owned_pct == pytest.approx(75.0)
    assert r[SALAH].owned_pct == pytest.approx(75.0)
    assert r[SAKA].owned_pct == pytest.approx(75.0)
    assert r[RAYA].owned_pct == pytest.approx(75.0)
    for el in (PALMER, WATKINS, ISAK, SELS):
        assert r[el].owned_pct == pytest.approx(50.0), el


def test_starting_vs_squad_ownership(toy_league):
    r = rows_by_element(toy_league)
    # Palmer is owned by 2 but started by 1 (103 benches him).
    assert r[PALMER].owned_pct == pytest.approx(50.0)
    assert r[PALMER].start_pct == pytest.approx(25.0)
    # Isak is owned by 2 and started by nobody.
    assert r[ISAK].owned_pct == pytest.approx(50.0)
    assert r[ISAK].start_pct == pytest.approx(0.0)


def test_effective_ownership_counts_the_captain_twice(toy_league):
    r = rows_by_element(toy_league)
    # 2 of 4 captain Haaland -> 50% captaincy on top of 75% ownership.
    assert r[HAALAND].captain_pct == pytest.approx(50.0)
    assert r[HAALAND].eo_pct == pytest.approx(125.0)
    assert r[SALAH].captain_pct == pytest.approx(50.0)
    assert r[SALAH].eo_pct == pytest.approx(125.0)
    # Nobody captains Saka, so his EO is just his ownership.
    assert r[SAKA].captain_pct == pytest.approx(0.0)
    assert r[SAKA].eo_pct == pytest.approx(75.0)


def test_multiplier_effective_ownership(toy_league):
    r = rows_by_element(toy_league)
    # sum of multipliers / N: Haaland 2+1+2+0 = 5 over 4 managers.
    assert r[HAALAND].mult_eo_pct == pytest.approx(125.0)
    # Palmer 1+0 = 1 over 4; benching drops him out of the effective figure.
    assert r[PALMER].mult_eo_pct == pytest.approx(25.0)
    assert r[ISAK].mult_eo_pct == pytest.approx(0.0)


def test_eo_never_below_ownership(toy_league):
    for r in ownership_table(toy_league):
        assert r.eo_pct >= r.owned_pct - 1e-9


# ------------------------------------------------------------- threat weights


def test_threat_weights_exclude_the_user_and_sum_to_one(toy_league):
    w = threat_weights(toy_league)
    assert toy_league.user_entry not in w
    assert set(w) == {102, 103, 104}
    assert sum(w.values()) == pytest.approx(1.0)


def test_contention_weights_match_the_closed_form(toy_league):
    # totals 104=29 (leader), 102=24, 103=21; tau = sample sd of those three.
    totals = [29, 24, 21]
    mean = sum(totals) / 3
    tau = math.sqrt(sum((t - mean) ** 2 for t in totals) / 2)
    raw = {104: 1.0, 102: math.exp(-5 / tau), 103: math.exp(-8 / tau)}
    s = sum(raw.values())
    w = threat_weights(toy_league, scheme="contention")
    for eid, v in raw.items():
        assert w[eid] == pytest.approx(v / s)
    # The leader must carry the most weight.
    assert w[104] > w[102] > w[103]


def test_uniform_scheme_reduces_to_plain_league_ownership(toy_league):
    w = threat_weights(toy_league, scheme="uniform", exclude_user=False)
    r = rows_by_element(toy_league, weights=w)
    for row in r.values():
        assert row.w_owned_pct == pytest.approx(row.owned_pct)
        assert row.w_eo_pct == pytest.approx(row.eo_pct)


def test_rival_weighted_eo_reweights_toward_the_leader(toy_league):
    w = threat_weights(toy_league, scheme="contention")
    r = rows_by_element(toy_league, weights=w)
    # Salah is owned by every rival and captained by the leader, so his
    # threat-weighted EO must exceed his flat EO.
    assert r[SALAH].w_eo_pct > r[SALAH].eo_pct
    # 104 and 102 both captain Salah (x2 each); 103 merely owns him (x1).
    expected = 100 * (2 * w[104] + 2 * w[102] + 1 * w[103])
    assert r[SALAH].w_eo_pct == pytest.approx(expected)
    # Haaland is owned only by the two managers *behind* the leader, so his
    # threat-weighted EO collapses well below his flat EO.
    assert r[HAALAND].w_eo_pct < r[HAALAND].eo_pct
    assert r[HAALAND].w_eo_pct == pytest.approx(100 * (1 * w[102] + 2 * w[103]))


def test_gap_scheme_gives_zero_weight_to_anyone_level_with_the_user(toy_league):
    toy_league.standings[-1]["total"] = toy_league.standings[0]["total"]  # user ties the leader
    w = threat_weights(toy_league, scheme="gap")
    assert sum(w.values()) == pytest.approx(1.0)


# --------------------------------------------------------------- point swings


def test_player_swing_is_multiplier_delta_times_points(toy_league):
    s = player_swings(toy_league)
    # Haaland: user multiplier 2, field mean (2+1+2+0)/4 = 1.25, 3 points.
    assert s[HAALAND].swing == pytest.approx((2 - 1.25) * 3)
    # Salah: user 0, field mean (0+2+1+2)/4 = 1.25, 6 points.
    assert s[SALAH].swing == pytest.approx(-1.25 * 6)
    assert s[SAKA].swing == pytest.approx(-0.75 * 4)
    assert s[PALMER].swing == pytest.approx((1 - 0.25) * 2)
    assert s[WATKINS].swing == pytest.approx((1 - 0.5) * 8)
    assert s[RAYA].swing == pytest.approx(-0.75 * 5)
    assert s[SELS].swing == pytest.approx((1 - 0.25) * 1)
    # Isak scored nothing and nobody started him.
    assert s[ISAK].swing == pytest.approx(0.0)


def test_total_swing_equals_user_score_minus_league_mean(toy_league):
    s = player_swings(toy_league)
    net = sum(v.swing for v in s.values())
    scores = [gw_score(toy_league, e, 1) for e in toy_league.entry_ids]
    mean = sum(scores) / len(scores)
    user = gw_score(toy_league, toy_league.user_entry, 1)
    assert net == pytest.approx(user - mean)
    assert net == pytest.approx(17 - 22.75)


def test_weighted_swing_uses_the_weighted_field(toy_league):
    w = threat_weights(toy_league)
    s = player_swings(toy_league, weights=w)
    # Salah: user 0; weighted field multiplier = 2*w104 + 2*w102 + 1*w103.
    wfm = 2 * w[104] + 2 * w[102] + 1 * w[103]
    assert s[SALAH].w_swing == pytest.approx(-wfm * 6)


# ------------------------------------------------------------- user position


def test_user_position_splits_differentials_and_liabilities(toy_league):
    up = user_position(toy_league, diff_threshold=60.0, liability_threshold=25.0)
    diffs = {d.name for d in up.differentials}
    liabs = {l.name for l in up.liabilities}
    # Owned by the user at 50% league ownership -> differentials at a 60% cut.
    assert diffs == {"Palmer", "Watkins", "Sels", "Isak"}
    # Owned by 75% of the league and not by the user.
    assert liabs == {"Salah", "Saka", "Raya"}
    # Haaland is owned by the user at 75% ownership: neither.
    assert "Haaland" not in diffs and "Haaland" not in liabs
    assert up.total_liability_swing == pytest.approx(-7.5 - 3.0 - 3.75)
    assert up.total_differential_swing == pytest.approx(1.5 + 4.0 + 0.75 + 0.0)


def test_liabilities_are_sorted_worst_first(toy_league):
    up = user_position(toy_league, diff_threshold=60.0, liability_threshold=25.0)
    swings = [l.swing for l in up.liabilities]
    assert swings == sorted(swings)


def test_captaincy_table(toy_league):
    caps = dict((n, (c, pct, pts)) for n, c, pct, pts in captaincy_table(toy_league, 1))
    assert caps["Haaland"] == (2, pytest.approx(50.0), 3)
    assert caps["Salah"] == (2, pytest.approx(50.0), 6)


# ------------------------------------------------------- scoring reconstruction


def test_gw_scores_match_the_hand_computation(toy_league):
    assert gw_score(toy_league, 101, 1) == 17
    assert gw_score(toy_league, 102, 1) == 24
    assert gw_score(toy_league, 103, 1) == 21
    assert gw_score(toy_league, 104, 1) == 29


def test_verify_totals_matches_every_entry(toy_league):
    v = verify_totals(toy_league)
    assert v["n_entries"] == 4
    assert v["match_rate"] == 1.0
    assert v["mismatches"] == []


def test_transfer_hits_are_subtracted(toy_league):
    toy_league.histories[101]["current"][0]["event_transfers_cost"] = 4
    assert recomputed_total(toy_league, 101) == 13
