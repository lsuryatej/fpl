"""Risk detection, especially the false positives that would bury real signals."""

from __future__ import annotations

import pytest

from fplopt.news import risk as nrisk

TEAMS = {1: "AVL", 2: "MCI"}


def element(pid, *, pos=4, minutes=180, price=70, status="a", news="", team=1):
    return {
        "id": pid,
        "web_name": f"P{pid}",
        "element_type": pos,
        "minutes": minutes,
        "now_cost": price,
        "status": status,
        "news": news,
        "team": team,
        "chance_of_playing_next_round": None,
    }


def test_zero_minutes_on_expensive_player_is_critical():
    risks = nrisk.scan([element(1, minutes=0, price=79)], TEAMS, gws_played=2)
    zero = [r for r in risks if r.kind == "zero-minutes"]
    assert len(zero) == 1
    assert zero[0].severity == "critical"
    assert zero[0].needs_human_check


def test_unavailable_status_is_critical_and_needs_no_news_check():
    risks = nrisk.scan(
        [element(1, status="u", news="Has joined Internazionale permanently")],
        TEAMS,
        gws_played=2,
    )
    flagged = [r for r in risks if r.kind == "flagged"]
    assert flagged[0].severity == "critical"
    # A permanent departure is settled; there is nothing left to research.
    assert not flagged[0].needs_human_check


def test_backup_keeper_with_zero_minutes_is_not_flagged():
    """The second goalkeeper is meant to play nothing. Flagging him is noise."""
    elements = [
        element(1, pos=1, minutes=180, price=50),  # first choice
        element(2, pos=1, minutes=0, price=40),  # backup
    ]
    picks = [{"element": 1}, {"element": 2}]
    risks = nrisk.squad_risks(picks, elements, TEAMS, gws_played=2)
    assert not [r for r in risks if r.player_id == 2]


def test_outfield_zero_minutes_still_flagged_alongside_backup_keeper():
    elements = [
        element(1, pos=1, minutes=180, price=50),
        element(2, pos=1, minutes=0, price=40),
        element(3, pos=4, minutes=0, price=79),
    ]
    picks = [{"element": 1}, {"element": 2}, {"element": 3}]
    risks = nrisk.squad_risks(picks, elements, TEAMS, gws_played=2)
    flagged_ids = {r.player_id for r in risks}
    assert 3 in flagged_ids
    assert 2 not in flagged_ids


def test_single_gameweek_does_not_trigger_zero_minutes():
    """One missed game is not evidence. Two is."""
    risks = nrisk.scan([element(1, minutes=0)], TEAMS, gws_played=1)
    assert not [r for r in risks if r.kind == "zero-minutes"]


def test_risks_sort_most_severe_first():
    elements = [
        element(1, minutes=90, price=60),  # low-minutes, medium/high
        element(2, status="u"),  # critical
    ]
    risks = nrisk.scan(elements, TEAMS, gws_played=2)
    assert risks[0].severity == "critical"


def test_open_questions_only_cover_unresolved_risks():
    elements = [element(1, minutes=0, price=79), element(2, status="u")]
    risks = nrisk.scan(elements, TEAMS, gws_played=2)
    questions = nrisk.open_questions(risks)
    assert any("zero minutes" in q for q in questions)
    assert len(questions) == sum(1 for r in risks if r.needs_human_check)
