"""Tests for the FPL 2026/27 scoring engine.

Every expected value here is a real observation from the 2026/27 season, taken
from ``https://fantasy.premierleague.com/api/event/{1,2}/live/``. The cached
payloads live in ``tests/scoring/data/`` so the suite runs offline; refresh
them by re-downloading the same endpoints.

The bulk of the assurance comes from ``test_full_reconciliation_*``, which
recomputes every played player in GW1 and GW2 and demands an exact match on
both the total and the per-identifier breakdown. The hand-written cases exist
to pin specific rules so that a regression names itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fplopt.scoring import (
    allocate_bonus,
    compute_bps,
    defensive_contribution,
    explain_player,
    score_player,
    score_player_multi,
)
from fplopt.scoring import rules as R
from fplopt.scoring.reconcile import (
    load_bootstrap,
    load_fixtures,
    load_live,
    names_from_bootstrap,
    positions_from_bootstrap,
    reconcile_bonus,
    reconcile_bonus_from_live,
    reconcile_points,
)

DATA = Path(__file__).parent / "data"
GKP, DEF, MID, FWD = R.GKP, R.DEF, R.MID, R.FWD


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def bootstrap() -> dict:
    return load_bootstrap(offline=True)


@pytest.fixture(scope="session")
def positions(bootstrap) -> dict[int, int]:
    return positions_from_bootstrap(bootstrap)


@pytest.fixture(scope="session")
def names(bootstrap) -> dict[int, str]:
    return names_from_bootstrap(bootstrap)


@pytest.fixture(scope="session")
def live() -> dict[int, dict]:
    return {gw: load_live(gw, offline=True) for gw in (1, 2)}


@pytest.fixture(scope="session")
def fixtures() -> list[dict]:
    return load_fixtures(offline=True)


def stat_line(**kwargs) -> dict:
    """A zeroed stat line with the given overrides, matching the live schema."""
    base = {
        "minutes": 0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bonus": 0,
        "clearances_blocks_interceptions": 0,
        "recoveries": 0,
        "tackles": 0,
        "defensive_contribution": 0,
    }
    base.update(kwargs)
    return base


# ==========================================================================
# The headline verification: recompute every played player in GW1 and GW2
# ==========================================================================
@pytest.mark.parametrize("gameweek", [1, 2])
def test_full_reconciliation_total_points(gameweek, positions, names, live):
    report = reconcile_points([gameweek], positions, names, live_by_gw=live)
    assert report.checked > 0
    detail = "\n".join(str(m) for m in report.mismatches[:25])
    assert report.mismatches == [], (
        f"GW{gameweek}: {len(report.mismatches)}/{report.checked} players mismatched "
        f"({report.rate:.4%} match rate)\n{detail}"
    )
    assert report.rate == 1.0


@pytest.mark.parametrize("gameweek", [1, 2])
def test_full_reconciliation_line_items(gameweek, positions, names, live):
    """Not just the total -- every per-identifier line must match the API."""
    report = reconcile_points([gameweek], positions, names, live_by_gw=live)
    detail = "\n".join(report.line_mismatches[:25])
    assert report.line_mismatches == [], (
        f"GW{gameweek}: {len(report.line_mismatches)}/{report.line_checked} breakdowns "
        f"differ from the API explain array\n{detail}"
    )
    assert report.line_rate == 1.0


def test_full_reconciliation_expected_volume(positions, names, live):
    """Guards against the reconciliation silently checking nothing."""
    report = reconcile_points([1, 2], positions, names, live_by_gw=live)
    assert report.checked == 591
    assert report.matched == 591
    assert dict(report.by_position) == {"GKP": 38, "DEF": 210, "MID": 278, "FWD": 65}


def test_season_totals_match_bootstrap(bootstrap, positions, live):
    """Summing our per-gameweek scores must reproduce each player's season total.

    An independent cross-check: it uses ``elements[].total_points`` from
    bootstrap-static rather than the live endpoint's per-gameweek figure.
    """
    computed: dict[int, int] = {}
    for gw_live in live.values():
        for element in gw_live["elements"]:
            computed[element["id"]] = computed.get(element["id"], 0) + score_player(
                element["stats"], positions[element["id"]]
            )
    bad = [
        (e["web_name"], e["total_points"], computed.get(e["id"], 0))
        for e in bootstrap["elements"]
        if computed.get(e["id"], 0) != e["total_points"]
    ]
    assert bad == [], f"{len(bad)} season totals differ: {bad[:15]}"


# ==========================================================================
# Bonus allocation
# ==========================================================================
@pytest.mark.parametrize("gameweek", [1, 2])
def test_bonus_allocation_from_live(gameweek, names, live):
    """Re-derive 3/2/1 from published BPS over the complete match roster."""
    report = reconcile_bonus_from_live([gameweek], names, live_by_gw=live)
    assert report.fixtures_checked > 0
    assert report.mismatches == [], "\n".join(report.mismatches[:25])
    assert report.player_rate == 1.0


def test_bonus_allocation_from_fixtures_endpoint(fixtures, names):
    report = reconcile_bonus(fixtures, names, gameweeks=[1, 2])
    assert report.fixtures_checked == 19
    assert report.mismatches == [], "\n".join(report.mismatches[:25])
    assert report.fixture_rate == 1.0


def test_bonus_allocation_total_volume(names, live):
    report = reconcile_bonus_from_live([1, 2], names, live_by_gw=live)
    assert report.fixtures_checked == 19
    assert report.players_checked == 591
    assert report.players_matched == 591


def test_bonus_no_ties():
    assert allocate_bonus({"a": 40, "b": 30, "c": 20, "d": 10}) == {
        "a": 3, "b": 2, "c": 1, "d": 0
    }


def test_bonus_tie_for_first():
    """Real case: fixture 11 (GW2). Cherki 60, Haaland 60, Foden 38, Semenyo 33.

    Published bonus: Cherki 3, Haaland 3, Foden 1, Semenyo 0. The rules say
    "Players 1 & 2 will receive 3 points each and Player 3 will receive 1
    point" -- the 2-point award is consumed by the tie.
    """
    assert allocate_bonus(
        {"Cherki": 60, "Haaland": 60, "Foden": 38, "Semenyo": 33}
    ) == {"Cherki": 3, "Haaland": 3, "Foden": 1, "Semenyo": 0}


def test_bonus_tie_for_second():
    """Real case: fixture 12 (GW2). Scott 39, Truffert 27, Tarkowski 27, DHL 24.

    Published bonus: Scott 3, Truffert 2, Tarkowski 2, Dewsbury-Hall 0.
    """
    assert allocate_bonus(
        {"Scott": 39, "Truffert": 27, "Tarkowski": 27, "Dewsbury-Hall": 24}
    ) == {"Scott": 3, "Truffert": 2, "Tarkowski": 2, "Dewsbury-Hall": 0}


def test_bonus_three_way_tie_for_second():
    """Real case: fixture 6 (GW1). Stach 37; Bijol/Justin/Trafford 28; Muharemovic 25.

    Published bonus: Stach 3, then 2 each to all three tied players, and
    nothing at all for Muharemovic -- the three-way tie exhausts the awards.
    """
    assert allocate_bonus(
        {"Stach": 37, "Bijol": 28, "Justin": 28, "Trafford": 28, "Muharemovic": 25}
    ) == {"Stach": 3, "Bijol": 2, "Justin": 2, "Trafford": 2, "Muharemovic": 0}


def test_bonus_tie_for_third():
    """Rules: "Player 1 gets 3, Player 2 gets 2, Players 3 & 4 get 1 each"."""
    assert allocate_bonus({"a": 40, "b": 30, "c": 20, "d": 20, "e": 10}) == {
        "a": 3, "b": 2, "c": 1, "d": 1, "e": 0
    }


def test_bonus_three_way_tie_for_first_exhausts_awards():
    """Not observed in GW1/GW2; follows from generalising the published rules.

    Three players on the top BPS take 3 each and consume every award slot.
    """
    assert allocate_bonus({"a": 50, "b": 50, "c": 50, "d": 40}) == {
        "a": 3, "b": 3, "c": 3, "d": 0
    }


def test_bonus_empty_fixture():
    assert allocate_bonus({}) == {}


# ==========================================================================
# Appearance points
# ==========================================================================
def test_zero_minutes_scores_nothing():
    # Even with stats attached, a player who did not appear scores nothing.
    assert score_player(stat_line(minutes=0, goals_scored=2, bonus=3), FWD) == 0


def test_short_appearance_is_one_point():
    # Real: Mason-Clark, GW2, 57 minutes, nothing else -> 1 point.
    assert score_player(stat_line(minutes=57), MID) == 1


def test_59_minutes_is_still_one_point():
    assert score_player(stat_line(minutes=59), MID) == 1


def test_60_minutes_is_two_points():
    # Real: Anthony, GW2 -- 60 minutes, midfielder clean sheet, one yellow -> 2.
    assert score_player(
        stat_line(minutes=60, clean_sheets=1, yellow_cards=1, recoveries=1), MID
    ) == 2


def test_58_minute_goalscorer():
    """Real: Angulo, GW1. 58 min, 1 goal, 1 conceded, 1 yellow -> 1 + 5 - 1 = 5.

    Pins the appearance boundary from below and confirms midfielders take no
    goals-conceded deduction.
    """
    assert (
        score_player(
            stat_line(
                minutes=58, goals_scored=1, goals_conceded=1, yellow_cards=1,
                clearances_blocks_interceptions=1, recoveries=5, tackles=1,
                defensive_contribution=7,
            ),
            MID,
        )
        == 5
    )


# ==========================================================================
# Goals, assists, clean sheets
# ==========================================================================
def test_midfielder_hat_trick():
    """Real: Bruno Fernandes, GW2. 90 min, 3 goals, 1 assist, 2 conceded, 3 bonus."""
    stats = stat_line(
        minutes=90, goals_scored=3, assists=1, goals_conceded=2, bonus=3,
        clearances_blocks_interceptions=1, recoveries=5, tackles=2,
        defensive_contribution=8,
    )
    assert score_player(stats, MID) == 23  # 2 + 15 + 3 + 0 + 3
    score = explain_player(stats, MID)
    assert score.get("goals_scored") == 15
    assert score.get("goals_conceded") == 0  # midfielders are never deducted


def test_defender_goal_and_clean_sheet():
    """Real: De Cuyper, GW1. 77 min, 1 goal, 1 assist, clean sheet, 2 bonus -> 17."""
    assert (
        score_player(
            stat_line(
                minutes=77, goals_scored=1, assists=1, clean_sheets=1, bonus=2,
                clearances_blocks_interceptions=1, recoveries=3, tackles=3,
                defensive_contribution=4,
            ),
            DEF,
        )
        == 17  # 2 + 6 + 3 + 4 + 2
    )


def test_forward_brace():
    """Real: Haaland, GW2. 90 min, 2 goals, 1 conceded, 3 bonus -> 13."""
    assert (
        score_player(
            stat_line(
                minutes=90, goals_scored=2, goals_conceded=1, bonus=3,
                clearances_blocks_interceptions=3, recoveries=2,
                defensive_contribution=5,
            ),
            FWD,
        )
        == 13  # 2 + 8 + 3
    )


def test_defender_two_assists():
    """Real: Castagne, GW1. 90 min, 2 assists, 3 conceded, 1 yellow -> 6."""
    assert (
        score_player(
            stat_line(
                minutes=90, assists=2, goals_conceded=3, yellow_cards=1,
                clearances_blocks_interceptions=5, recoveries=4, tackles=1,
                defensive_contribution=6,
            ),
            DEF,
        )
        == 6  # 2 + 6 - 1 - 1
    )


def test_goalkeeper_goal_is_ten_points():
    """2026/27 change: a keeper's goal is 10, not 6.

    Sourced from ``game_config.scoring.goals_scored`` and the official rules
    page constants, which now render separate rows for keeper and defender
    goals. No keeper scored in GW1 or GW2, so this rule is config-verified
    rather than observed.
    """
    assert R.GOAL_POINTS[GKP] == 10
    assert R.GOAL_POINTS[DEF] == 6
    assert score_player(stat_line(minutes=90, goals_scored=1, clean_sheets=1), GKP) == 16


def test_clean_sheet_points_by_position():
    stats = stat_line(minutes=90, clean_sheets=1)
    assert score_player(stats, GKP) == 6
    assert score_player(stats, DEF) == 6
    assert score_player(stats, MID) == 3
    assert score_player(stats, FWD) == 2


def test_clean_sheet_derived_when_absent():
    """Callers without the API's clean_sheets field get it derived correctly."""
    partial = {"minutes": 90, "goals_conceded": 0}
    assert score_player(partial, DEF) == 6
    assert score_player({"minutes": 45, "goals_conceded": 0}, DEF) == 1
    assert score_player({"minutes": 90, "goals_conceded": 1}, DEF) == 2


# ==========================================================================
# Goals conceded and goalkeeping
# ==========================================================================
def test_goals_conceded_is_minus_one_per_two():
    for conceded, expected in [(0, 0), (1, 0), (2, -1), (3, -1), (4, -2), (5, -2), (6, -3)]:
        score = explain_player(
            stat_line(minutes=90, goals_conceded=conceded), DEF
        )
        assert score.get("goals_conceded") == expected, conceded


def test_goalkeeper_conceding_four():
    """Real: Henderson, GW2. 90 min, 4 conceded, 4 saves -> 2 - 2 + 1 = 1."""
    assert (
        score_player(
            stat_line(
                minutes=90, goals_conceded=4, saves=4,
                clearances_blocks_interceptions=2, recoveries=11,
            ),
            GKP,
        )
        == 1
    )


def test_saves_are_one_point_per_three():
    for saves, expected in [(0, 0), (2, 0), (3, 1), (5, 1), (6, 2), (9, 3)]:
        score = explain_player(stat_line(minutes=90, saves=saves), GKP)
        assert score.get("saves") == expected, saves


def test_goalkeeper_clean_sheet_with_saves():
    """Real: Tzolakis, GW1. 90 min, clean sheet, 5 saves, 3 bonus -> 10."""
    assert (
        score_player(
            stat_line(minutes=90, clean_sheets=1, saves=5, bonus=3, recoveries=22),
            GKP,
        )
        == 10  # 2 + 4 + 1 + 3
    )


def test_penalty_save_is_five_points():
    """Config-verified only -- no penalty was saved in GW1 or GW2."""
    assert R.PENALTY_SAVE_POINTS == 5
    assert score_player(stat_line(minutes=90, penalties_saved=1, goals_conceded=1), GKP) == 7


# ==========================================================================
# Negative events
# ==========================================================================
def test_own_goal():
    """Real: Lindelof, GW1. 90 min, 1 own goal, 4 conceded -> 2 - 2 - 2 = -2."""
    assert (
        score_player(
            stat_line(
                minutes=90, own_goals=1, goals_conceded=4,
                clearances_blocks_interceptions=3, recoveries=1,
                defensive_contribution=3,
            ),
            DEF,
        )
        == -2
    )


def test_own_goal_does_not_cancel_attacking_returns():
    """Real: Joao Pedro, GW2. Goal, assist, own goal, 3 conceded, 2 bonus -> 9."""
    assert (
        score_player(
            stat_line(
                minutes=90, goals_scored=1, assists=1, own_goals=1,
                goals_conceded=3, bonus=2, recoveries=2, tackles=1,
                defensive_contribution=3,
            ),
            FWD,
        )
        == 9  # 2 + 4 + 3 - 2
    )


def test_red_card():
    """Real: Gomes, GW1. 39 min, red card, 4 conceded (MID, so no deduction) -> -2."""
    assert (
        score_player(
            stat_line(
                minutes=39, red_cards=1, goals_conceded=4,
                clearances_blocks_interceptions=1, recoveries=2, tackles=4,
                defensive_contribution=7,
            ),
            MID,
        )
        == -2  # 1 - 3
    )


def test_penalty_miss():
    """Real: Thiago, GW1. 82 min, penalty missed, forward clean sheet -> 0."""
    assert (
        score_player(
            stat_line(
                minutes=82, penalties_missed=1, clean_sheets=1,
                clearances_blocks_interceptions=1, recoveries=4,
                defensive_contribution=5,
            ),
            FWD,
        )
        == 0  # 2 - 2
    )


def test_yellow_card():
    assert score_player(stat_line(minutes=19, yellow_cards=1), MID) == 0
    assert score_player(stat_line(minutes=90, yellow_cards=1), MID) == 1


# ==========================================================================
# Defensive contribution
# ==========================================================================
def test_defcon_thresholds_are_position_specific():
    assert R.DEFCON_THRESHOLD[DEF] == 10
    assert R.DEFCON_THRESHOLD[MID] == 12
    assert R.DEFCON_THRESHOLD[FWD] == 12
    assert R.DEFCON_THRESHOLD[GKP] is None


@pytest.mark.parametrize(
    "position,threshold", [(DEF, 10), (MID, 12), (FWD, 12)]
)
def test_defcon_boundary(position, threshold):
    below = explain_player(
        stat_line(minutes=90, defensive_contribution=threshold - 1), position
    )
    at = explain_player(
        stat_line(minutes=90, defensive_contribution=threshold), position
    )
    assert below.get("defensive_contribution") == 0
    assert at.get("defensive_contribution") == 2


def test_defcon_defender_at_ten():
    """Real: Ajer, GW1. 90 min, clean sheet, CBI 8 + tackles 2 = 10 -> 8 points."""
    assert (
        score_player(
            stat_line(
                minutes=90, clean_sheets=1, clearances_blocks_interceptions=8,
                recoveries=3, tackles=2, defensive_contribution=10,
            ),
            DEF,
        )
        == 8  # 2 + 4 + 2
    )


def test_defcon_defender_at_nine_gets_nothing():
    """Real: Van Hecke, GW1. CBI 7 + tackles 2 = 9, one short. 3 conceded -> 1."""
    assert (
        score_player(
            stat_line(
                minutes=90, goals_conceded=3, clearances_blocks_interceptions=7,
                recoveries=3, tackles=2, defensive_contribution=9,
            ),
            DEF,
        )
        == 1  # 2 - 1
    )


def test_defcon_midfielder_at_twelve():
    """Real: Ndiaye, GW1. Assist, clean sheet, recoveries 8 + tackles 4 = 12 -> 9."""
    assert (
        score_player(
            stat_line(
                minutes=90, assists=1, clean_sheets=1, bonus=1,
                recoveries=8, tackles=4, defensive_contribution=12,
            ),
            MID,
        )
        == 9  # 2 + 3 + 1 + 2 + 1
    )


def test_defcon_midfielder_at_eleven_gets_nothing():
    """Real: Scott, GW1. CBI 1 + recoveries 6 + tackles 4 = 11, one short -> 2."""
    assert (
        score_player(
            stat_line(
                minutes=90, goals_conceded=2, clearances_blocks_interceptions=1,
                recoveries=6, tackles=4, defensive_contribution=11,
            ),
            MID,
        )
        == 2
    )


def test_defcon_does_not_stack():
    """A defender on 21 defensive actions still gets exactly 2 points."""
    score = explain_player(stat_line(minutes=90, defensive_contribution=21), DEF)
    assert score.get("defensive_contribution") == 2


def test_goalkeepers_never_get_defcon():
    stats = stat_line(
        minutes=90, clearances_blocks_interceptions=15, recoveries=25, tackles=5
    )
    assert defensive_contribution(stats, GKP) == 0
    assert explain_player(stats, GKP).get("defensive_contribution") == 0


def test_defcon_derivation_excludes_recoveries_for_defenders():
    """Defenders count CBI + tackles only; everyone else adds recoveries."""
    raw = {"clearances_blocks_interceptions": 6, "tackles": 3, "recoveries": 8}
    assert defensive_contribution(raw, DEF) == 9
    assert defensive_contribution(raw, MID) == 17
    assert defensive_contribution(raw, FWD) == 17


def test_defcon_derivation_matches_api_field_for_every_played_player(positions, live):
    """The derivation rule must reproduce the API's own defensive_contribution."""
    bad = []
    for gw, gw_live in live.items():
        for element in gw_live["elements"]:
            stats = element["stats"]
            if stats["minutes"] <= 0:
                continue
            position = positions[element["id"]]
            derived = defensive_contribution(
                {k: v for k, v in stats.items() if k != "defensive_contribution"},
                position,
            )
            if derived != stats["defensive_contribution"]:
                bad.append((gw, element["id"], position, derived,
                            stats["defensive_contribution"]))
    assert bad == [], f"{len(bad)} rows where derived DefCon != API DefCon: {bad[:10]}"


# ==========================================================================
# Multi-fixture gameweeks
# ==========================================================================
def test_double_gameweek_scores_per_fixture():
    """Appearance, clean sheet and DefCon all reset per match, so score per match.

    Two 45-minute halves across two fixtures give 1 + 1 = 2, not the 2 points
    a single 90-minute appearance would earn.
    """
    a = stat_line(minutes=45, clean_sheets=0, defensive_contribution=6)
    b = stat_line(minutes=45, clean_sheets=0, defensive_contribution=6)
    assert score_player_multi([a, b], DEF) == 2
    # And DefCon is judged per match: 6 + 6 never crosses the 10 line.
    assert score_player(stat_line(minutes=90, defensive_contribution=12), DEF) == 4


# ==========================================================================
# API surface
# ==========================================================================
def test_explain_matches_api_shape_for_a_real_player(live, positions):
    """explain_player's line items line up with the API's own explain array."""
    element = next(
        e for e in live[1]["elements"]
        if e["stats"]["minutes"] > 0 and e["stats"]["bonus"] > 0
        and e["stats"]["goals_scored"] > 0
    )
    position = positions[element["id"]]
    score = explain_player(element["stats"], position)
    api = {
        entry["identifier"]: entry["points"]
        for fixture in element["explain"]
        for entry in fixture["stats"]
    }
    mine = {line.identifier: line.points for line in score.lines}
    assert mine == api


def test_explain_excludes_bonus_when_asked():
    stats = stat_line(minutes=90, goals_scored=1, bonus=3)
    assert explain_player(stats, FWD, include_bonus=False).total_points == 6
    assert explain_player(stats, FWD, include_bonus=True).total_points == 9


def test_missing_keys_default_to_zero():
    assert score_player({"minutes": 90, "clean_sheets": 0}, MID) == 2
    assert score_player({"minutes": 30, "clean_sheets": 0}, FWD) == 1
    # ``clean_sheets`` is the one key that is derived rather than zeroed when
    # absent, so that raw non-API stat lines still score correctly.
    assert score_player({"minutes": 90}, MID) == 3


def test_unknown_position_raises():
    # The manager position (5) was removed for 2026/27.
    with pytest.raises(ValueError, match="unknown position"):
        score_player(stat_line(minutes=90), 5)


# ==========================================================================
# BPS
# ==========================================================================
def test_bps_reports_missing_opta_inputs():
    """compute_bps must not silently pretend an API stat line is complete."""
    breakdown = compute_bps(stat_line(minutes=90, saves=4), GKP)
    assert not breakdown.is_exact
    assert "key_passes" in breakdown.assumed_zero
    assert "pass_completion_pct" in breakdown.assumed_zero


def test_bps_is_exact_with_a_full_opta_line():
    full = stat_line(minutes=90, goals_scored=1, tackles=2)
    full.update({key: 0 for key in R.BPS_STATS_REQUIRING_OPTA})
    breakdown = compute_bps(full, MID)
    assert breakdown.is_exact
    # 6 (long play) + 18 (midfielder goal) + 4 (two tackles)
    assert breakdown.total == 28


def test_bps_goals_conceded_is_minus_four_each():
    """2026/27 change, verified by regression on GW1+GW2 (slope -3.90 / -3.98)."""
    assert R.BPS_GOALS_CONCEDED[GKP] == -4
    assert R.BPS_GOALS_CONCEDED[DEF] == -4
    assert R.BPS_GOALS_CONCEDED[MID] == 0
    full = stat_line(minutes=90, goals_conceded=3)
    full.update({key: 0 for key in R.BPS_STATS_REQUIRING_OPTA})
    assert compute_bps(full, DEF).total == 6 - 12


def test_bps_cbi_and_recoveries_are_one_per_three():
    assert R.BPS_CBI_PER == 3
    assert R.BPS_RECOVERIES_PER == 3
    full = stat_line(minutes=90, clearances_blocks_interceptions=8, recoveries=7)
    full.update({key: 0 for key in R.BPS_STATS_REQUIRING_OPTA})
    assert compute_bps(full, DEF).total == 6 + 2 + 2


def test_bps_penalty_goal_replaces_open_play_value():
    full = stat_line(minutes=90, goals_scored=2)
    full.update({key: 0 for key in R.BPS_STATS_REQUIRING_OPTA})
    full["penalties_scored"] = 1
    # 6 + 24 (one open-play forward goal) + 12 (one penalty)
    assert compute_bps(full, FWD).total == 42


def test_bps_save_bonuses_stack():
    full = stat_line(minutes=90, saves=1)
    full.update({key: 0 for key in R.BPS_STATS_REQUIRING_OPTA})
    full["saves_from_inside_box"] = 1
    full["big_chance_saves"] = 1
    assert compute_bps(full, GKP).total == 6 + 2 + 1 + 1


def test_bps_pass_bands_do_not_stack_and_need_the_floor():
    base = stat_line(minutes=90)
    base.update({key: 0 for key in R.BPS_STATS_REQUIRING_OPTA})
    below_floor = dict(base, passes_attempted=29, pass_completion_pct=95.0)
    assert compute_bps(below_floor, MID).total == 6
    top_band = dict(base, passes_attempted=40, pass_completion_pct=95.0)
    assert compute_bps(top_band, MID).total == 6 + 6
    mid_band = dict(base, passes_attempted=40, pass_completion_pct=82.0)
    assert compute_bps(mid_band, MID).total == 6 + 4


def test_bps_being_tackled_is_no_longer_penalised():
    """The -1 "being tackled" BPS deduction was removed for 2026/27."""
    assert not hasattr(R, "BPS_TACKLED")


# ==========================================================================
# Rule-table sanity, guarding against silent edits
# ==========================================================================
def test_scoring_table_matches_api_game_config(bootstrap):
    """rules.py must agree with the server's own game_config.scoring."""
    config = bootstrap["game_config"]["scoring"]
    short = {GKP: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}
    for position, key in short.items():
        assert R.GOAL_POINTS[position] == config["goals_scored"][key]
        assert R.CLEAN_SHEET_POINTS[position] == config["clean_sheets"][key]
        assert R.GOALS_CONCEDED_POINTS[position] == config["goals_conceded"][key]
        assert R.DEFCON_POINTS[position] == config["defensive_contribution"][key]
    assert R.ASSIST_POINTS == config["assists"]
    assert R.SHORT_PLAY_POINTS == config["short_play"]
    assert R.LONG_PLAY_POINTS == config["long_play"]
    assert R.SAVE_POINTS == config["saves"]
    assert R.PENALTY_SAVE_POINTS == config["penalties_saved"]
    assert R.PENALTY_MISS_POINTS == config["penalties_missed"]
    assert R.YELLOW_CARD_POINTS == config["yellow_cards"]
    assert R.RED_CARD_POINTS == config["red_cards"]
    assert R.OWN_GOAL_POINTS == config["own_goals"]


def test_manager_position_is_gone(bootstrap):
    """2026/27 removed the manager element type entirely."""
    assert [t["id"] for t in bootstrap["element_types"]] == [1, 2, 3, 4]
    manager_keys = {
        k: v for k, v in bootstrap["game_config"]["scoring"].items()
        if k.startswith("mng_")
    }
    for value in manager_keys.values():
        if isinstance(value, dict):
            assert set(value.values()) == {0}
        else:
            assert value == 0


def test_cached_fixture_data_is_present():
    for name in ("bootstrap_elements.json", "live_1.json", "live_2.json", "fixtures.json"):
        path = DATA / name
        assert path.exists(), f"missing cached payload {path}"
        assert json.loads(path.read_text())
