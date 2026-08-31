"""Rendering must survive missing upstream data and must escape user content."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from fplopt.brief.assemble import Brief
from fplopt.brief.render import render


def empty_brief(**overrides):
    base = dict(
        generated_at=dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc),
        target_gw=3,
        deadline=dt.datetime(2026, 9, 4, 17, 30, tzinfo=dt.timezone.utc),
        remaining_gws=36,
        manager={"entry": 1, "name": "A", "team_name": "T", "league": "L", "n_managers": 45},
        standing={"rank": 45, "total": 78, "leader": 166, "median": 131, "net_swing": -54.0},
        squad=[],
        fixtures=[],
        risks=[],
        open_questions=[],
        liabilities=[],
        differentials=[],
        ownership=[],
        gap_targets=[],
        overlap=[],
        price_movers=pd.DataFrame(),
    )
    base.update(overrides)
    return Brief(**base)


def test_renders_with_every_optional_panel_missing():
    html = render(empty_brief())
    assert "<title>ECPP War Room</title>" in html
    assert "Recommendation pending" in html


def test_pending_lists_unfinished_panels():
    brief = empty_brief()
    assert set(brief.pending) == {"projections", "recommendation", "chip plan", "model health"}


def test_countdown_reports_time_remaining():
    assert "3d" in render(empty_brief())


def test_passed_deadline_is_reported_not_negative():
    brief = empty_brief(
        generated_at=dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc)
    )
    assert "passed" in render(brief)


def test_league_name_is_escaped():
    """League and team names are user-supplied and reach the page verbatim."""
    brief = empty_brief(
        manager={
            "entry": 1,
            "name": "A",
            "team_name": "<script>alert(1)</script>",
            "league": "L",
            "n_managers": 45,
        }
    )
    html = render(brief)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_every_colour_token_is_defined_outside_theme_blocks():
    """A token defined only inside a media query renders unreadably in the
    un-stamped default theme, which is the classic broken-artifact bug."""
    html = render(empty_brief())
    root_block = html.split(":root {")[1].split("}")[0]
    for token in ("--ground", "--surface", "--ink", "--line", "--accent", "--critical"):
        assert token in root_block, f"{token} missing from the base :root palette"
