"""Price snapshot parsing and mover detection."""

from __future__ import annotations

import pandas as pd

from fplopt.prices import track


def boot(projections, *, hourly=100, calibrating=False, locked=None):
    return {
        "elements": [
            {
                "id": 1,
                "web_name": "Test",
                "team": 1,
                "element_type": 4,
                "now_cost": 75,
                "cost_change_start": 0,
                "cost_change_event": 0,
                "transfers_in_event": 500,
                "transfers_out_event": 100,
                "selected_by_percent": "12.5",
                "price_change_hourly_rate": hourly,
                "price_change_projections": projections,
                "price_change_locked_until": locked,
                "price_change_calibrating": calibrating,
                "status": "a",
            }
        ]
    }


def test_projection_offsets_are_pulled_apart():
    rows = track._rows(
        boot([{"offset": 0, "projected_percent": "88.0", "likelihood": 1},
              {"offset": 1, "projected_percent": "95.0", "likelihood": 2}]),
        pd.Timestamp("2026-09-01", tz="UTC"),
    )
    assert rows[0]["proj_pct_today"] == 88.0
    assert rows[0]["proj_likelihood_today"] == 1
    assert rows[0]["proj_pct_tomorrow"] == 95.0


def test_net_transfers_is_derived():
    rows = track._rows(boot([]), pd.Timestamp("2026-09-01", tz="UTC"))
    assert rows[0]["net_transfers_event"] == 400


def test_missing_projections_do_not_raise():
    rows = track._rows(boot(None), pd.Timestamp("2026-09-01", tz="UTC"))
    assert rows[0]["proj_pct_today"] is None


def test_movers_filters_below_threshold():
    frame = pd.DataFrame(
        [
            {"player_id": 1, "proj_pct_today": 90.0, "calibrating": False, "locked_until": None},
            {"player_id": 2, "proj_pct_today": 20.0, "calibrating": False, "locked_until": None},
        ]
    )
    out = track.movers(frame, threshold=75.0)
    assert list(out["player_id"]) == [1]
    assert out.iloc[0]["direction"] == "rise"


def test_movers_excludes_locked_and_calibrating():
    """FPL suppressing a change means its projection means nothing right now."""
    frame = pd.DataFrame(
        [
            {"player_id": 1, "proj_pct_today": 99.0, "calibrating": True, "locked_until": None},
            {"player_id": 2, "proj_pct_today": -99.0, "calibrating": False, "locked_until": "2026-09-02"},
            {"player_id": 3, "proj_pct_today": -99.0, "calibrating": False, "locked_until": None},
        ]
    )
    out = track.movers(frame, threshold=75.0)
    assert list(out["player_id"]) == [3]
    assert out.iloc[0]["direction"] == "fall"


def test_movers_sorts_by_absolute_urgency():
    frame = pd.DataFrame(
        [
            {"player_id": 1, "proj_pct_today": 80.0, "calibrating": False, "locked_until": None},
            {"player_id": 2, "proj_pct_today": -120.0, "calibrating": False, "locked_until": None},
        ]
    )
    assert list(track.movers(frame, threshold=75.0)["player_id"]) == [2, 1]
