"""Monte Carlo points-distribution simulator.

Combines :mod:`team_strength` (scorelines), :mod:`minutes` (playing time) and
:mod:`rates` (events per 90) into a full simulated points distribution per
player per gameweek. Every simulated stat line is scored by
``fplopt.scoring.engine.score_player`` / ``fplopt.scoring.bps`` -- this module
never hardcodes a point value, by design.

Per fixture, per Monte Carlo draw:

1. A scoreline is drawn from the team-strength model's Dixon-Coles-corrected
   score matrix.
2. Every registered player on either side gets an independent played/started
   draw (from their :mod:`minutes` estimate) and, conditional on that,
   minutes.
3. Each team's goals are allocated to scorers by a categorical draw weighted
   by ``goals_p90 * minutes_share`` among players who played that draw;
   assists are allocated the same way off a separately-drawn "was this goal
   assisted" Bernoulli calibrated to the league's historical assisted-goal
   rate.
4. Defensive actions (CBI, tackles, recoveries) and saves are drawn as
   Poisson counts scaled by minutes; a goalkeeper's saves are additionally
   scaled by this specific fixture's expected goals against, so a tough
   fixture increases their bonus-threshold chances.
5. Cards and own goals are drawn from position-level historical base rates.
6. The full observable-stat line is scored via
   ``fplopt.scoring.bps.compute_bps`` + ``allocate_bonus`` for bonus, then
   ``fplopt.scoring.engine.score_player`` for the total -- the only place a
   point value is ever assigned.

Known simplifications (see the project verification report for the full,
numbered list):

* Playing-time draws are independent per player, not a constrained
  "exactly ~11 starters" lineup draw -- this mildly overstates variance in a
  team's aggregate minutes.
* Bonus can only be computed from the 15 of 38 BPS actions the public API
  exposes (``fplopt.scoring.rules`` documents exactly which); the 23
  Opta-only actions (shots on target, crosses, dribbles, key passes, fouls,
  etc.) are not simulated at all, so simulated bonus is a systematic,
  known-direction underestimate for high-BPS creative players.
* Penalty-taking, saving and missing are not modelled as separate events;
  penalty goals are already implicit in each player's blended goals-per-90
  rate, but the discrete ``-2`` penalty-miss and ``+5`` penalty-save events
  are not simulated at all.
* Goals conceded (and therefore clean sheets) are attributed to a player for
  the *whole* match once they register any minutes in it, since minutes are
  not tracked by exact substitution time.
* A goal's scorer and assister are drawn independently (no exclusion of
  "can't assist your own goal"); at typical squad sizes this makes a
  negligible difference to the simulated distribution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import team_strength as ts
from ..scoring import bps as bps_mod
from ..scoring import engine as scoring_engine

__all__ = [
    "fit_card_and_og_rates",
    "league_assisted_goal_fraction",
    "simulate_gameweek",
]

log = logging.getLogger(__name__)

_POSITION_CODE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
_MAX_GOALS = 9
_MIN_P_PLAY = 0.02  # players below this never get simulated; distribution is a point mass at 0.


def fit_card_and_og_rates(history_df: pd.DataFrame, seasons: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Position-level yellow card, red card and own-goal rates per 90, from history."""
    seasons = seasons or tuple(sorted(history_df["season"].unique())[-4:])
    frame = history_df[history_df["season"].isin(seasons)].copy()
    frame = frame.dropna(subset=["position", "minutes"])
    frame["position_code"] = frame["position"].map(_POSITION_CODE)
    frame = frame.dropna(subset=["position_code"])
    frame = frame[frame["minutes"] > 0]
    rows = []
    for pos, g in frame.groupby("position_code"):
        exposure = g["minutes"].sum() / 90.0
        rows.append(
            {
                "position": int(pos),
                "yellow_p90": float(g["yellow_cards"].sum() / exposure) if exposure else 0.08,
                "red_p90": float(g["red_cards"].sum() / exposure) if exposure else 0.003,
                "own_goal_p90": float(g["own_goals"].sum() / exposure) if exposure else 0.002,
            }
        )
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


def league_assisted_goal_fraction(history_df: pd.DataFrame, seasons: tuple[str, ...] | None = None) -> float:
    """Fraction of goals that carry an FPL assist, league-wide, from history.

    Used to calibrate how often a simulated goal gets an assist allocated to
    someone at all (own goals, penalties and pure solo efforts are not
    assisted; the rest usually are).
    """
    seasons = seasons or tuple(sorted(history_df["season"].unique())[-4:])
    frame = history_df[history_df["season"].isin(seasons)]
    goals = frame["goals_scored"].sum()
    assists = frame["assists"].sum()
    if not goals:
        return 0.65
    return float(np.clip(assists / goals, 0.3, 0.95))


def _draw_scorelines(model: ts.DixonColesModel, home: str, away: str, n_sims: int, rng: np.random.Generator):
    mat = model.score_matrix(home, away, max_goals=_MAX_GOALS)
    flat = mat.flatten()
    flat = flat / flat.sum()
    n = mat.shape[0]
    idx = rng.choice(len(flat), size=n_sims, p=flat)
    home_goals = idx // n
    away_goals = idx % n
    return home_goals.astype(int), away_goals.astype(int)


def _simulate_side_play(side: pd.DataFrame, n_sims: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Vectorised played/started/minutes draw for one team's roster in one fixture.

    ``side`` must have one row per player with ``p_play``, ``p_start``,
    ``minutes_if_started``, ``minutes_if_sub`` (from :mod:`minutes`).
    Returns arrays shaped ``(n_players, n_sims)``.
    """
    n_players = len(side)
    p_play = side["p_play"].to_numpy()[:, None]
    ratio = np.divide(side["p_start"].to_numpy(), np.maximum(side["p_play"].to_numpy(), 1e-9))[:, None]

    played = rng.random((n_players, n_sims)) < p_play
    started = played & (rng.random((n_players, n_sims)) < ratio)
    sub_on = played & ~started

    minutes_start_mean = side["minutes_if_started"].to_numpy()[:, None]
    minutes_sub_mean = side["minutes_if_sub"].to_numpy()[:, None]
    minutes_start_draw = np.clip(rng.normal(minutes_start_mean, 10.0, size=(n_players, n_sims)), 1, 96)
    minutes_sub_draw = np.clip(rng.normal(minutes_sub_mean, 8.0, size=(n_players, n_sims)), 1, 45)

    minutes = np.where(started, minutes_start_draw, np.where(sub_on, minutes_sub_draw, 0.0))
    return {"played": played, "started": started, "minutes": minutes}


def _allocate_events(
    weights: np.ndarray, counts: np.ndarray, rng: np.random.Generator, max_events: int = 8
) -> np.ndarray:
    """Allocate ``counts[sim]`` weighted-categorical events to players, per sim.

    ``weights`` is ``(n_players, n_sims)`` (zero for anyone who can't receive
    an event, e.g. didn't play). Returns an integer ``(n_players, n_sims)``
    matrix of how many events each player received. Vectorised over sims by
    looping over event *slots* (bounded by ``max_events``), not over sims.
    """
    n_players, n_sims = weights.shape
    out = np.zeros((n_players, n_sims), dtype=int)
    safe_weights = np.maximum(weights, 0.0)
    cumw = np.cumsum(safe_weights, axis=0)
    totw = cumw[-1, :]
    totw_safe = np.where(totw > 0, totw, 1.0)
    for slot in range(max_events):
        active = slot < counts
        if not active.any():
            continue
        u = rng.random(n_sims) * totw_safe
        idx = (cumw < u[None, :]).sum(axis=0)
        idx = np.clip(idx, 0, n_players - 1)
        sim_ids = np.nonzero(active & (totw > 0))[0]
        if len(sim_ids) == 0:
            continue
        np.add.at(out, (idx[sim_ids], sim_ids), 1)
    return out


def _simulate_side_stats(
    side: pd.DataFrame,
    play: dict[str, np.ndarray],
    team_goals: np.ndarray,
    goals_against: np.ndarray,
    save_scale: float,
    assisted_fraction: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Full observable stat line for one side of one fixture, all sims at once."""
    minutes = play["minutes"]
    exposure = minutes / 90.0
    n_players, n_sims = minutes.shape

    goal_weight = np.maximum(side["goals_scored_p90"].to_numpy()[:, None] * exposure, 0.0)
    assist_weight = np.maximum(side["assists_p90"].to_numpy()[:, None] * exposure, 0.0)

    goals = _allocate_events(goal_weight, team_goals, rng)
    n_assisted = rng.binomial(team_goals, assisted_fraction)
    assists = _allocate_events(assist_weight, n_assisted, rng)

    cbi = rng.poisson(np.maximum(side["clearances_blocks_interceptions_p90"].to_numpy()[:, None] * exposure, 0.0))
    tackles = rng.poisson(np.maximum(side["tackles_p90"].to_numpy()[:, None] * exposure, 0.0))
    recoveries = rng.poisson(np.maximum(side["recoveries_p90"].to_numpy()[:, None] * exposure, 0.0))

    is_gk = (side["position"].to_numpy() == 1)[:, None]
    save_rate = np.maximum(side["saves_p90"].to_numpy()[:, None] * exposure * save_scale, 0.0)
    saves = np.where(is_gk, rng.poisson(save_rate), 0)

    yellow = np.minimum(rng.poisson(np.maximum(side["yellow_p90"].to_numpy()[:, None] * exposure, 0.0)), 1)
    red = (rng.random((n_players, n_sims)) < np.clip(side["red_p90"].to_numpy()[:, None] * exposure, 0, 1)).astype(int)
    own_goals = rng.poisson(np.maximum(side["own_goal_p90"].to_numpy()[:, None] * exposure, 0.0))

    conceded = np.broadcast_to(goals_against[None, :], (n_players, n_sims)) * (minutes > 0)

    return {
        "minutes": minutes,
        "goals_scored": goals,
        "assists": assists,
        "clearances_blocks_interceptions": cbi,
        "tackles": tackles,
        "recoveries": recoveries,
        "saves": saves,
        "yellow_cards": yellow,
        "red_cards": red,
        "own_goals": own_goals,
        "goals_conceded": conceded,
    }


def _bonus_for_fixture(
    home_stats: dict[str, np.ndarray],
    away_stats: dict[str, np.ndarray],
    home_positions: np.ndarray,
    away_positions: np.ndarray,
    n_sims: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bonus points for both sides of one fixture, all sims.

    BPS can only be computed from the observable-API stats (see module
    docstring); this is the documented fidelity gap versus real bonus.
    """
    n_home = len(home_positions)
    n_away = len(away_positions)
    home_bonus = np.zeros((n_home, n_sims), dtype=int)
    away_bonus = np.zeros((n_away, n_sims), dtype=int)

    for s in range(n_sims):
        bps_by_slot: dict[tuple[str, int], int] = {}
        for side, stats, positions in (("h", home_stats, home_positions), ("a", away_stats, away_positions)):
            for i, pos in enumerate(positions):
                if stats["minutes"][i, s] <= 0:
                    continue
                row = {k: v[i, s] for k, v in stats.items()}
                bps_by_slot[(side, i)] = bps_mod.compute_bps(row, int(pos)).total
        if not bps_by_slot:
            continue
        awards = bps_mod.allocate_bonus(bps_by_slot)
        for (side, i), bonus in awards.items():
            if side == "h":
                home_bonus[i, s] = bonus
            else:
                away_bonus[i, s] = bonus
    return home_bonus, away_bonus


def _score_side(stats: dict[str, np.ndarray], bonus: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Final total points per player per sim, via the verified scoring engine."""
    n_players, n_sims = bonus.shape
    points = np.zeros((n_players, n_sims), dtype=int)
    for i, pos in enumerate(positions):
        pos = int(pos)
        for s in range(n_sims):
            if stats["minutes"][i, s] <= 0:
                continue
            row = {k: v[i, s] for k, v in stats.items()}
            row["bonus"] = bonus[i, s]
            row["penalties_saved"] = 0
            row["penalties_missed"] = 0
            points[i, s] = scoring_engine.score_player(row, pos)
    return points


def simulate_gameweek(
    fixtures: pd.DataFrame,
    team_model: ts.DixonColesModel,
    player_frame: pd.DataFrame,
    n_sims: int,
    rng: np.random.Generator,
    card_og_rates: pd.DataFrame,
    assisted_fraction: float,
) -> pd.DataFrame:
    """Simulate one gameweek's fixtures and return per-player points-distribution summary.

    Parameters
    ----------
    fixtures:
        One row per fixture: ``home_team`` and ``away_team`` (canonical names
        matching ``team_model``), ``home_team_id``/``away_team_id`` (bootstrap
        team ids, to join against ``player_frame``).
    player_frame:
        One row per player, already joined: ``player_id``, ``position``,
        ``team`` (bootstrap team id), ``price``, everything from
        :func:`fplopt.model.minutes.estimate_minutes` and
        :func:`fplopt.model.rates.estimate_rates`.
    card_og_rates:
        Output of :func:`fit_card_and_og_rates`, joined onto ``player_frame``
        by position before simulation.

    Returns
    -------
    DataFrame with one row per simulated player: ``player_id``, ``position``,
    ``team``, ``price``, ``p_play``, ``p_start``, ``exp_minutes``, ``ep``,
    ``sd``, ``p10``, ``p25``, ``p50``, ``p75``, ``p90``.
    """
    frame = player_frame.merge(card_og_rates, on="position", how="left")
    frame[["yellow_p90", "red_p90", "own_goal_p90"]] = frame[["yellow_p90", "red_p90", "own_goal_p90"]].fillna(
        {"yellow_p90": 0.08, "red_p90": 0.003, "own_goal_p90": 0.002}
    )

    simmed = frame[frame["p_play"] >= _MIN_P_PLAY].copy()
    skipped_ids = frame.loc[frame["p_play"] < _MIN_P_PLAY, "player_id"]

    results: list[pd.DataFrame] = []
    for _, fx in fixtures.iterrows():
        home_side = simmed[simmed["team"] == fx["home_team_id"]].reset_index(drop=True)
        away_side = simmed[simmed["team"] == fx["away_team_id"]].reset_index(drop=True)
        if home_side.empty and away_side.empty:
            continue

        home_goals, away_goals = _draw_scorelines(team_model, fx["home_team"], fx["away_team"], n_sims, rng)
        lh, la = team_model.lambdas(fx["home_team"], fx["away_team"])
        league_avg = float(np.exp(team_model.base))
        home_save_scale = float(np.clip(la / league_avg, 0.5, 2.0))
        away_save_scale = float(np.clip(lh / league_avg, 0.5, 2.0))

        home_play = _simulate_side_play(home_side, n_sims, rng) if len(home_side) else None
        away_play = _simulate_side_play(away_side, n_sims, rng) if len(away_side) else None

        home_stats = (
            _simulate_side_stats(home_side, home_play, home_goals, away_goals, home_save_scale, assisted_fraction, rng)
            if home_play is not None
            else {k: np.zeros((0, n_sims)) for k in
                  ("minutes", "goals_scored", "assists", "clearances_blocks_interceptions",
                   "tackles", "recoveries", "saves", "yellow_cards", "red_cards", "own_goals", "goals_conceded")}
        )
        away_stats = (
            _simulate_side_stats(away_side, away_play, away_goals, home_goals, away_save_scale, assisted_fraction, rng)
            if away_play is not None
            else {k: np.zeros((0, n_sims)) for k in home_stats}
        )

        home_pos = home_side["position"].to_numpy() if len(home_side) else np.array([])
        away_pos = away_side["position"].to_numpy() if len(away_side) else np.array([])
        home_bonus, away_bonus = _bonus_for_fixture(home_stats, away_stats, home_pos, away_pos, n_sims)

        home_points = _score_side(home_stats, home_bonus, home_pos) if len(home_side) else np.zeros((0, n_sims))
        away_points = _score_side(away_stats, away_bonus, away_pos) if len(away_side) else np.zeros((0, n_sims))

        for side_df, points in ((home_side, home_points), (away_side, away_points)):
            if side_df.empty:
                continue
            results.append(_summarise(side_df, points))

    summary = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    zero_rows = _zero_summary(frame[frame["player_id"].isin(skipped_ids)])
    return pd.concat([summary, zero_rows], ignore_index=True)


def _summarise(side_df: pd.DataFrame, points: np.ndarray) -> pd.DataFrame:
    qs = np.percentile(points, [10, 25, 50, 75, 90], axis=1)
    return pd.DataFrame(
        {
            "player_id": side_df["player_id"].to_numpy(),
            "position": side_df["position"].to_numpy(),
            "team": side_df["team"].to_numpy(),
            "price": side_df["price"].to_numpy(),
            "p_play": side_df["p_play"].to_numpy(),
            "p_start": side_df["p_start"].to_numpy(),
            "exp_minutes": side_df["exp_minutes"].to_numpy(),
            "ep": points.mean(axis=1),
            "sd": points.std(axis=1),
            "p10": qs[0], "p25": qs[1], "p50": qs[2], "p75": qs[3], "p90": qs[4],
        }
    )


def _zero_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["player_id", "position", "team", "price", "p_play", "p_start", "exp_minutes",
                     "ep", "sd", "p10", "p25", "p50", "p75", "p90"]
        )
    out = frame[["player_id", "position", "team", "price", "p_play", "p_start", "exp_minutes"]].copy()
    for col in ("ep", "sd", "p10", "p25", "p50", "p75", "p90"):
        out[col] = 0.0
    return out
