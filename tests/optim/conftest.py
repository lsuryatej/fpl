"""Shared fixtures for the optim test suite.

``synthetic_projections`` produces a DataFrame matching EXACTLY the schema
contract for ``data/projections/*.parquet`` that the model-building agent is
responsible for:

    player_id, gw, position, team, price, p_play, p_start, exp_minutes,
    ep, sd, p10, p25, p50, p75, p90

The optim package is developed and tested entirely against this generator
rather than waiting on that agent's real output; ``fplopt.optim`` code only
needs a DataFrame with these columns; whether it came from a real parquet or
this generator is invisible to it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from fplopt.optim.squad import optimize_squad
from fplopt.optim.state import (
    DEF,
    FWD,
    GKP,
    MID,
    ManagerState,
    PlayerHolding,
)

PROJECTION_COLUMNS = [
    "player_id",
    "gw",
    "position",
    "team",
    "price",
    "p_play",
    "p_start",
    "exp_minutes",
    "ep",
    "sd",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
]

_BASE_PRICE = {GKP: 40, DEF: 40, MID: 45, FWD: 50}
_SQUAD_SHAPE = {GKP: 2, DEF: 5, MID: 5, FWD: 3}  # per-team player count in the synthetic pool


def synthetic_projections(
    n_teams: int = 20,
    gws: Sequence[int] = range(1, 9),
    seed: int = 0,
    quality_range: tuple[float, float] = (0.5, 6.5),
) -> pd.DataFrame:
    """Build a synthetic multi-gameweek projections table matching the real schema.

    ``n_teams`` clubs each get a small fixed-composition pool (2 GK, 5 DEF,
    5 MID, 3 FWD -- plenty of depth for a 15-man squad under the 3-per-club
    limit without needing hundreds of players per position). Each player has
    a random underlying "quality" that drives both price and expected points,
    held constant across gameweeks (matching the constant-price assumption
    documented in :mod:`fplopt.optim.multiweek`); each gameweek then adds
    independent noise so the optimizer sees genuine week-to-week variation to
    plan around.
    """
    rng = np.random.default_rng(seed)
    player_meta: list[tuple[int, int, int, int, float]] = []
    pid = 1
    for team in range(1, n_teams + 1):
        for pos, count in _SQUAD_SHAPE.items():
            for _ in range(count):
                quality = float(rng.uniform(*quality_range))
                price = _BASE_PRICE[pos] + int(round(quality * 9))
                player_meta.append((pid, pos, team, price, quality))
                pid += 1

    rows = []
    for gw in gws:
        for pid, pos, team, price, quality in player_meta:
            noise = rng.normal(0, 0.35)
            ep = max(0.3, quality + noise)
            sd = max(0.4, ep * 0.55 + rng.uniform(0, 0.8))
            p_play = float(np.clip(rng.uniform(0.55, 0.99), 0.0, 1.0))
            p_start = float(np.clip(p_play - rng.uniform(0.0, 0.12), 0.0, 1.0))
            exp_minutes = round(p_start * 85 + max(0.0, p_play - p_start) * 25, 1)
            rows.append(
                {
                    "player_id": pid,
                    "gw": gw,
                    "position": pos,
                    "team": team,
                    "price": price,
                    "p_play": round(p_play, 3),
                    "p_start": round(p_start, 3),
                    "exp_minutes": exp_minutes,
                    "ep": round(ep, 3),
                    "sd": round(sd, 3),
                    "p10": round(max(0.0, ep - 1.28 * sd), 3),
                    "p25": round(max(0.0, ep - 0.67 * sd), 3),
                    "p50": round(ep, 3),
                    "p75": round(ep + 0.67 * sd, 3),
                    "p90": round(ep + 1.28 * sd, 3),
                }
            )
    df = pd.DataFrame(rows, columns=PROJECTION_COLUMNS)
    return df


def make_initial_state(
    projections: pd.DataFrame,
    gw: int,
    entry_id: int = 1,
    free_transfers: int = 1,
    bank_reserve: int = 0,
    chips_remaining: dict[str, int] | None = None,
) -> ManagerState:
    """Build a legal :class:`ManagerState` by solving a real squad for ``gw``.

    Used to bootstrap a plausible "current squad" for multiweek tests without
    hand-writing 15 players every time. Each holding's purchase price is set
    equal to its current price (bought at today's price, no profit or loss
    yet) so tests can layer specific price-history scenarios on top when they
    need to.
    """
    gw_rows = projections[projections["gw"] == gw]
    budget = 1000 - bank_reserve
    result = optimize_squad(gw_rows, budget=budget)
    assert result.is_optimal, f"could not build a synthetic initial squad: {result.status}"
    price_by_id = dict(zip(gw_rows["player_id"], gw_rows["price"]))
    position_by_id = dict(zip(gw_rows["player_id"], gw_rows["position"]))
    team_by_id = dict(zip(gw_rows["player_id"], gw_rows["team"]))
    holdings = []
    for pid in result.squad_ids:
        holdings.append(
            PlayerHolding(
                player_id=pid,
                position=position_by_id[pid],
                team=team_by_id[pid],
                purchase_price=price_by_id[pid],
                current_price=price_by_id[pid],
                is_captain=(pid == result.captain_id),
                is_vice_captain=(pid == result.vice_captain_id),
                multiplier=2 if pid == result.captain_id else (1 if pid in result.starting_ids else 0),
            )
        )
    bank = (budget - result.total_cost) + bank_reserve
    return ManagerState(
        entry_id=entry_id,
        as_of_gw=gw,
        bank=bank,
        free_transfers=free_transfers,
        chips_remaining=chips_remaining or {"wildcard": 2, "free_hit": 2, "bench_boost": 2, "triple_captain": 2},
        squad=tuple(holdings),
    )


@pytest.fixture
def projections() -> pd.DataFrame:
    return synthetic_projections()


@pytest.fixture
def small_horizon_projections() -> pd.DataFrame:
    return synthetic_projections(n_teams=12, gws=range(1, 5), seed=7)
