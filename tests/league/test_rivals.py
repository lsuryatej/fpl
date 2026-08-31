"""Rival profiling: chips, transfers, and the template/differential axis."""

from __future__ import annotations

import copy

import pytest

from fplopt.league.rivals import build_profiles, squad_lines, style_summary, transfer_log

HAALAND, SALAH, SAKA, PALMER, WATKINS, ISAK, RAYA, SELS = 1, 2, 3, 4, 5, 6, 7, 8


def test_profiles_cover_every_entry_including_the_user(toy_league):
    p = build_profiles(toy_league)
    assert set(p) == {101, 102, 103, 104}
    assert p[101].rank == 4
    assert p[104].rank == 1


def test_template_score_is_leave_one_out(toy_league):
    p = build_profiles(toy_league)
    # 102 holds Haaland, Salah, Saka, Raya, Isak.
    # Ownership counts across 4 managers: Haaland 3, Salah 3, Saka 3, Raya 3, Isak 2.
    # Leave-one-out over the other 3 managers: 2/3, 2/3, 2/3, 2/3, 1/3.
    expected = 100 * (2 / 3 + 2 / 3 + 2 / 3 + 2 / 3 + 1 / 3) / 5
    assert p[102].template_score == pytest.approx(expected)


def test_a_squad_nobody_else_owns_scores_zero_template(toy_league):
    data = copy.deepcopy(toy_league)
    for i, pk in enumerate(data.picks[101][1]["picks"]):
        pk["element"] = 900 + i  # players held by nobody else
        data.players[900 + i] = data.players[HAALAND]
        data.live[1][900 + i] = {"total_points": 0, "minutes": 90}
    p = build_profiles(data)
    assert p[101].template_score == pytest.approx(0.0)
    assert p[101].style == "differential"


def test_style_classification_is_a_partition(toy_league):
    p = build_profiles(toy_league)
    counts = style_summary(p)
    assert sum(counts.values()) == len(p)
    assert set(counts) <= {"template", "balanced", "differential"}


def test_chips_left_decrements_on_use(toy_league):
    data = copy.deepcopy(toy_league)
    data.histories[102]["chips"] = [{"name": "bboost", "event": 1}]
    p = build_profiles(data)
    # Two of each chip per season (one per half).
    assert p[102].chips_left["bboost"] == 1
    assert p[102].chips_left["3xc"] == 2
    assert p[102].chips_used == [("bboost", 1)]
    assert p[103].chips_left["bboost"] == 2


def test_overlap_with_user_is_recorded(toy_league):
    p = build_profiles(toy_league)
    assert p[102].overlap_with_user == 2  # Haaland, Isak
    assert p[101].overlap_with_user == 5  # the user overlaps himself entirely


def test_gap_to_user_and_threat_weight(toy_league):
    p = build_profiles(toy_league)
    assert p[104].gap_to_user == 29 - 17
    assert p[101].threat_weight == 0.0  # the user is not a threat to himself
    assert p[104].threat_weight > p[103].threat_weight


def test_transfer_log_and_squad_lines(toy_league):
    data = copy.deepcopy(toy_league)
    data.transfers[102] = [{"event": 1, "element_in": PALMER, "element_out": SAKA}]
    assert transfer_log(data, 102) == ["GW1: Saka -> Palmer"]
    lines = squad_lines(data, build_profiles(data)[101])
    assert len(lines) == 5
    # Starters first, then the bench; within each block, sorted by position.
    assert [l.startswith("BENCH") for l in lines] == [False, False, False, False, True]
    assert lines[0].strip().startswith("GKP Sels")  # only goalkeeper in the XI
    assert any("Haaland(C)" in l for l in lines)
    assert "Isak" in lines[-1]


def test_hits_and_bench_points_are_summed(toy_league):
    data = copy.deepcopy(toy_league)
    data.histories[101]["current"][0]["event_transfers_cost"] = 8
    data.histories[101]["current"][0]["points_on_bench"] = 12
    p = build_profiles(data)
    assert p[101].hits_taken == 8
    assert p[101].points_on_bench == 12
