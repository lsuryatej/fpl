"""Handcrafted 4-manager league whose ownership/EO answers are checkable by hand.

Squads are trimmed to 5 players each (the maths does not care about squad size)
so every expected number can be worked out on paper.

Players: 1 Haaland, 2 Salah, 3 Saka, 4 Palmer, 5 Watkins, 6 Isak, 7 Raya, 8 Sels

GW1 live points: Haaland 3, Salah 6, Saka 4, Palmer 2, Watkins 8, Isak 0,
                 Raya 5, Sels 1  (Isak played 0 minutes)

Gameweek scores, all hand-computable:

  101 = 2*3 + 2 + 8 + 1            = 17   -> rank 4 (the user, last)
  102 = 3 + 2*6 + 4 + 5            = 24   -> rank 2
  103 = 2*3 + 6 + 4 + 5            = 21   -> rank 3
  104 = 2*6 + 4 + 8 + 5            = 29   -> rank 1

The standings totals in this fixture equal those gameweek scores, so
``verify_totals`` must report a 100% match.

Managers (entry ids 101..104), 101 is "the user":

  101 (user):  Haaland(C,x2), Palmer, Watkins, Sels, Isak[bench]
  102:         Haaland, Salah(C,x2), Saka, Raya, Isak[bench]
  103:         Haaland(C,x2), Salah, Saka, Raya, Palmer[bench]
  104:         Salah(C,x2), Saka, Watkins, Raya, Sels[bench]
"""

from __future__ import annotations

import pytest

from fplopt.league.fetch import LeagueData, Player

POINTS_GW1 = {1: 3, 2: 6, 3: 4, 4: 2, 5: 8, 6: 0, 7: 5, 8: 1}
MINUTES_GW1 = {1: 90, 2: 90, 3: 90, 4: 90, 5: 90, 6: 0, 7: 90, 8: 90}

NAMES = {
    1: ("Haaland", "MCI", "FWD"),
    2: ("Salah", "LIV", "MID"),
    3: ("Saka", "ARS", "MID"),
    4: ("Palmer", "CHE", "MID"),
    5: ("Watkins", "AVL", "FWD"),
    6: ("Isak", "NEW", "FWD"),
    7: ("Raya", "ARS", "GKP"),
    8: ("Sels", "NFO", "GKP"),
}

# element, multiplier, is_captain, is_vice
SQUADS = {
    101: [(1, 2, True, False), (4, 1, False, True), (5, 1, False, False), (8, 1, False, False), (6, 0, False, False)],
    102: [(1, 1, False, False), (2, 2, True, False), (3, 1, False, True), (7, 1, False, False), (6, 0, False, False)],
    103: [(1, 2, True, False), (2, 1, False, True), (3, 1, False, False), (7, 1, False, False), (4, 0, False, False)],
    104: [(2, 2, True, False), (3, 1, False, False), (5, 1, False, True), (7, 1, False, False), (8, 0, False, False)],
}

TOTALS = {101: 17, 102: 24, 103: 21, 104: 29}
RANKS = {104: 1, 102: 2, 103: 3, 101: 4}


def _picks_payload(eid: int) -> dict:
    return {
        "active_chip": None,
        "automatic_subs": [],
        "entry_history": {"event": 1, "points": TOTALS[eid], "value": 1000, "bank": 0,
                          "event_transfers": 0, "event_transfers_cost": 0, "points_on_bench": 0},
        "picks": [
            {
                "element": el,
                "position": i + 1,
                "multiplier": mult,
                "is_captain": cap,
                "is_vice_captain": vice,
                "element_type": 3,
            }
            for i, (el, mult, cap, vice) in enumerate(SQUADS[eid])
        ],
    }


@pytest.fixture
def toy_league() -> LeagueData:
    players = {
        pid: Player(
            id=pid,
            web_name=n,
            full_name=n,
            team=pid,
            team_short=t,
            position=pos,
            now_cost=50,
            total_points=POINTS_GW1[pid],
            global_owned_pct=10.0,
            status="a",
        )
        for pid, (n, t, pos) in NAMES.items()
    }

    standings = [
        {
            "entry": eid,
            "entry_name": f"team{eid}",
            "player_name": f"mgr{eid}",
            "rank": RANKS[eid],
            "last_rank": RANKS[eid],
            "total": TOTALS[eid],
            "event_total": TOTALS[eid],
        }
        for eid in sorted(RANKS, key=lambda e: RANKS[e])
    ]

    data = LeagueData(
        league_id=1,
        league_name="toy",
        user_entry=101,
        gameweeks=[1],
        standings=standings,
        players=players,
        live={1: {pid: {"total_points": POINTS_GW1[pid], "minutes": MINUTES_GW1[pid]} for pid in POINTS_GW1}},
        events=[{"id": 1, "finished": True, "data_checked": True, "average_entry_score": 40}],
        fixtures={1: [{"team_h": 1, "team_a": 2, "finished": True, "finished_provisional": True}]},
        next_gw=2,
    )
    for eid in SQUADS:
        data.picks[eid] = {1: _picks_payload(eid)}
        data.histories[eid] = {
            "current": [
                {
                    "event": 1,
                    "points": TOTALS[eid],
                    "total_points": TOTALS[eid],
                    "value": 1000,
                    "bank": 0,
                    "event_transfers": 0,
                    "event_transfers_cost": 0,
                    "points_on_bench": 0,
                }
            ],
            "past": [],
            "chips": [],
        }
        data.transfers[eid] = []
    return data
