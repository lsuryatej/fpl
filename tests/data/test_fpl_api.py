"""Tests for the FPL API client.

Offline tests run against real payload slices saved under ``fixtures/``.
The single networked test is marked ``network`` and can be deselected with
``-m "not network"``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from fplopt.data import fpl_api
from fplopt.data.http import FetchError


# ----------------------------------------------------------------------
# offline: pure helpers over saved payloads
# ----------------------------------------------------------------------
def test_elements_frame_joins_position_and_team(bootstrap_small: dict[str, Any]) -> None:
    frame = fpl_api.elements_frame(bootstrap_small)
    assert len(frame) == len(bootstrap_small["elements"])
    assert set(frame["position"]) <= {"GKP", "DEF", "MID", "FWD"}
    assert frame["position"].notna().all()
    assert frame["full_name"].str.len().gt(0).all()
    # now_cost is tenths of a million.
    assert (frame["price"] * 10 == frame["now_cost"]).all()


def test_events_frame_parses_deadlines(bootstrap_small: dict[str, Any]) -> None:
    frame = fpl_api.events_frame(bootstrap_small)
    assert pd.api.types.is_datetime64_any_dtype(frame["deadline_time"])
    assert frame["deadline_time"].is_monotonic_increasing


def test_current_and_finished_events(bootstrap_small: dict[str, Any]) -> None:
    assert fpl_api.current_event(bootstrap_small) == 2
    assert fpl_api.next_event(bootstrap_small) == 3
    assert fpl_api.finished_events(bootstrap_small) == [1]
    # GW2 is finished=False, so relaxing data_checked must not add it.
    assert fpl_api.finished_events(bootstrap_small, require_data_checked=False) == [1]


def test_current_event_none_before_season_start() -> None:
    boot = {"events": [{"id": 1, "is_current": False, "is_next": True, "finished": False}]}
    assert fpl_api.current_event(boot) is None
    assert fpl_api.finished_events(boot) == []


def test_fixtures_frame_drops_nested_stats(fixtures_small: list[dict[str, Any]]) -> None:
    frame = fpl_api.fixtures_frame(fixtures_small)
    assert "stats" not in frame.columns
    assert pd.api.types.is_datetime64_any_dtype(frame["kickoff_time"])
    assert {"team_h", "team_a", "team_h_difficulty"} <= set(frame.columns)


def test_fixture_stats_frame_flattens_both_sides(fixtures_small: list[dict[str, Any]]) -> None:
    frame = fpl_api.fixture_stats_frame(fixtures_small)
    assert not frame.empty
    assert set(frame.columns) == {"fixture", "event", "identifier", "side", "element", "value"}
    assert set(frame["side"]) <= {"h", "a"}
    assert frame["element"].notna().all()


def test_live_explain_reconstructs_total_points(live_gw1_small: dict[str, Any]) -> None:
    """The explain breakdown is the ground truth; it must sum to total_points."""
    stats = fpl_api.live_points_frame(live_gw1_small, gw=1)
    explain = fpl_api.live_explain_frame(live_gw1_small, gw=1)

    explain["contrib"] = explain["points"].fillna(0) + explain["points_modification"].fillna(0)
    rebuilt = explain.groupby("element", as_index=False)["contrib"].sum()
    merged = stats[["element", "total_points"]].merge(rebuilt, on="element", how="left")
    merged["contrib"] = merged["contrib"].fillna(0)

    assert (merged["contrib"] == merged["total_points"]).all()


def test_live_points_frame_coerces_string_metrics(live_gw1_small: dict[str, Any]) -> None:
    frame = fpl_api.live_points_frame(live_gw1_small, gw=1)
    for col in ("influence", "creativity", "threat", "ict_index", "expected_goals"):
        assert pd.api.types.is_numeric_dtype(frame[col]), col
    assert (frame["gw"] == 1).all()


def test_live_explain_frame_shape(live_gw1_small: dict[str, Any]) -> None:
    frame = fpl_api.live_explain_frame(live_gw1_small, gw=1)
    assert set(frame.columns) == {
        "gw", "element", "fixture", "identifier", "value", "points", "points_modification",
    }
    # Only players who featured get explain rows.
    assert len(frame) <= len(live_gw1_small["elements"]) * 10


def test_bootstrap_validates_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = fpl_api.FPLClient(cache_enabled=False)
    monkeypatch.setattr(client, "_get", lambda *a, **k: {"unexpected": True})
    with pytest.raises(FetchError, match="unexpected payload"):
        client.bootstrap()


def test_fixtures_validates_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = fpl_api.FPLClient(cache_enabled=False)
    monkeypatch.setattr(client, "_get", lambda *a, **k: {"not": "a list"})
    with pytest.raises(FetchError, match="unexpected payload"):
        client.fixtures()


def test_league_standings_all_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paging must stop on has_next=False and not re-request forever."""
    pages = {
        1: {"standings": {"has_next": True, "results": [{"entry": 1}, {"entry": 2}]}},
        2: {"standings": {"has_next": False, "results": [{"entry": 3}]}},
    }
    calls: list[int] = []
    client = fpl_api.FPLClient(cache_enabled=False)

    def fake(lid: int, page: int = 1) -> dict[str, Any]:
        calls.append(page)
        return pages[page]

    monkeypatch.setattr(client, "league_standings", fake)
    rows = client.league_standings_all(123)
    assert [r["entry"] for r in rows] == [1, 2, 3]
    assert calls == [1, 2]


def test_iter_element_summaries_survives_a_bad_id(monkeypatch: pytest.MonkeyPatch) -> None:
    client = fpl_api.FPLClient(cache_enabled=False)

    def fake(pid: int) -> dict[str, Any]:
        if pid == 2:
            raise FetchError("boom")
        return {"history": [], "history_past": [], "fixtures": []}

    monkeypatch.setattr(client, "element_summary", fake)
    out = list(fpl_api.iter_element_summaries([1, 2, 3], client=client))
    assert [pid for pid, _, _ in out] == [1, 2, 3]
    assert out[1][1] is None and "boom" in out[1][2]
    assert out[0][1] is not None and out[2][1] is not None


# ----------------------------------------------------------------------
# online
# ----------------------------------------------------------------------
@pytest.mark.network
def test_live_fpl_api_end_to_end() -> None:
    """Hit the real FPL API: bootstrap, fixtures, and a live gameweek.

    Also re-runs the points reconciliation against live data, which is the
    check the whole scoring layer depends on.
    """
    client = fpl_api.FPLClient(cache_enabled=False)
    try:
        boot = client.bootstrap()
        assert len(boot["elements"]) > 400
        assert len(boot["teams"]) == 20
        assert len(boot["events"]) == 38
        assert {"GKP", "DEF", "MID", "FWD"} <= {
            t["singular_name_short"] for t in boot["element_types"]
        }

        fixtures = client.fixtures()
        assert len(fixtures) == 380

        done = fpl_api.finished_events(boot)
        if not done:
            pytest.skip("no completed gameweek yet this season")

        gw = done[0]
        payload = client.live(gw)
        stats = fpl_api.live_points_frame(payload, gw)
        explain = fpl_api.live_explain_frame(payload, gw)
        assert len(stats) > 400

        explain["contrib"] = explain["points"].fillna(0) + explain["points_modification"].fillna(0)
        rebuilt = explain.groupby("element", as_index=False)["contrib"].sum()
        merged = stats[["element", "total_points"]].merge(rebuilt, on="element", how="left")
        merged["contrib"] = merged["contrib"].fillna(0)
        assert (merged["contrib"] == merged["total_points"]).all()
    finally:
        client.close()


@pytest.mark.network
def test_live_element_summary_and_league_standings() -> None:
    """Exercise the per-player and league endpoints against the real API."""
    client = fpl_api.FPLClient(cache_enabled=False)
    try:
        summary = client.element_summary(1)
        assert {"fixtures", "history", "history_past"} == set(summary)

        page = client.league_standings(314)
        results = page["standings"]["results"]
        assert len(results) == 50
        assert {"entry", "player_name", "total", "rank"} <= set(results[0])

        eid = results[0]["entry"]
        assert client.entry(eid)["id"] == eid
        assert {"current", "past", "chips"} <= set(client.entry_history(eid))
        assert isinstance(client.entry_transfers(eid), list)
    finally:
        client.close()
