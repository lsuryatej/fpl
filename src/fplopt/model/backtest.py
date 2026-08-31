"""Backtest and calibration harness -- the mandatory verification step.

Two backtests are run, deliberately different in what they trade off:

**Primary backtest -- 2026-27 GW1 and GW2.** Fully rule-consistent: both the
prediction and the "actual" points are 2026-27 rules, because that season
*is* 2026-27. GW1 is predicted with zero current-season signal (pure prior);
GW2 is predicted using only GW1. This is exactly the deployment scenario
(predicting GW3 from 2 games of history), just one and two steps earlier.

**Secondary backtest -- five spread-out gameweeks of 2025-26.** Much larger
player-gameweek count, but with one disclosed rule-vintage caveat: defensive
contribution is a brand-new 2026-27 rule. 2025-26 has the raw components
(CBI, tackles, recoveries) needed to compute what defcon *would* have scored,
so "actual" points here are the archive's real stat line rescored through the
current engine (which adds defcon on top of what that season's real bonus/
points actually were) -- consistent with what the model targets, but bonus
itself still reflects that season's own (pre-overhaul) BPS formula, since
recomputing 2026-27 BPS needs Opta inputs the public archive never had. This
is the same fidelity gap :mod:`simulate` has for live projections, just
inherited here too; it is not a new source of error introduced by backtesting.

Every prediction in both backtests is produced by :func:`predict_gameweek_from_history`,
a self-contained "replay" of the exact same team_strength -> minutes -> rates
-> simulate pipeline :mod:`project` uses, fed only with data available before
the target gameweek (no leakage: prior seasons are fit excluding the target
season itself, and only the target season's own earlier gameweeks are used
as "current-season" signal).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd

from ..data import store
from ..scoring import engine as scoring_engine
from . import minutes as mm
from . import rates as rr
from . import simulate as sim
from . import team_strength as ts

__all__ = [
    "rescore_actual",
    "season_mean_baseline",
    "predict_gameweek_from_history",
    "run_backtest",
    "calibration_table",
    "main",
]

log = logging.getLogger(__name__)

_POSITION_CODE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
_RATE_STATS = rr.RATE_STATS


def _season_match_dates(history_df: pd.DataFrame, season: str, gw: int) -> pd.Timestamp | None:
    rows = history_df[(history_df["season"] == season) & (history_df["gw"] == gw)]
    if rows.empty:
        return None
    return pd.to_datetime(rows["kickoff_time"], utc=True, errors="coerce").min()


def _target_fixtures(history_df: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    """Match-level fixtures for one (season, gw), in the shape :func:`simulate.simulate_gameweek` wants."""
    sub = history_df[(history_df["season"] == season) & (history_df["gw"] == gw)]
    sub = sub.dropna(subset=["fixture", "team", "was_home", "team_h_score", "team_a_score"])
    home = sub[sub["was_home"].astype(bool)].groupby("fixture", as_index=False).first()
    away = sub[~sub["was_home"].astype(bool)].groupby("fixture", as_index=False).first()[["fixture", "team"]]
    merged = home.merge(away, on="fixture", suffixes=("_h", "_a"))
    from ._teamnames import normalize_team

    return pd.DataFrame(
        {
            "home_team": merged["team_h"].map(normalize_team),
            "away_team": merged["team_a"].map(normalize_team),
            "home_team_id": merged["team_h"].map(normalize_team),
            "away_team_id": merged["team_a"].map(normalize_team),
        }
    )


def rescore_actual(history_df: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    """Re-score one gameweek's real stat lines through the current (2026-27) engine.

    See the module docstring for exactly what this does and does not fix
    relative to the archive's own ``total_points`` column.
    """
    sub = history_df[(history_df["season"] == season) & (history_df["gw"] == gw)].copy()
    sub["position_code"] = sub["position"].map(_POSITION_CODE)
    sub = sub.dropna(subset=["position_code", "element"])
    rows = []
    for _, r in sub.iterrows():
        stats = {
            "minutes": r.get("minutes", 0) or 0,
            "goals_scored": r.get("goals_scored", 0) or 0,
            "assists": r.get("assists", 0) or 0,
            "goals_conceded": r.get("goals_conceded", 0) or 0,
            "own_goals": r.get("own_goals", 0) or 0,
            "penalties_saved": r.get("penalties_saved", 0) or 0,
            "penalties_missed": r.get("penalties_missed", 0) or 0,
            "yellow_cards": r.get("yellow_cards", 0) or 0,
            "red_cards": r.get("red_cards", 0) or 0,
            "saves": r.get("saves", 0) or 0,
            "bonus": r.get("bonus", 0) or 0,
            "clearances_blocks_interceptions": r.get("clearances_blocks_interceptions", 0) or 0,
            "tackles": r.get("tackles", 0) or 0,
            "recoveries": r.get("recoveries", 0) or 0,
        }
        pos = int(r["position_code"])
        rows.append(
            {
                "element": int(r["element"]),
                "position": pos,
                "actual_points": scoring_engine.score_player(stats, pos),
                "actual_minutes": stats["minutes"],
            }
        )
    return pd.DataFrame(rows)


def season_mean_baseline(history_df: pd.DataFrame, season: str, gw: int) -> pd.Series:
    """Baseline (a): each player's own mean ``total_points`` before this gameweek.

    Prefers the same season's earlier gameweeks; falls back to the player's
    full record in the immediately preceding season for players with no
    same-season history yet (i.e. gw==1). Matched by ``element`` -- FPL ids
    are not guaranteed stable across seasons, so this systematically misses
    transferred-in players for a GW1 target, who then simply have no baseline
    (excluded from that comparison row, not silently given 0).
    """
    same_season_prior = history_df[
        (history_df["season"] == season) & (history_df["gw"] < gw) & (history_df["gw"] >= 1)
    ]
    baseline = same_season_prior.groupby("element")["total_points"].mean()
    if baseline.empty:
        seasons = sorted(history_df["season"].unique())
        if season in seasons and seasons.index(season) > 0:
            prev_season = seasons[seasons.index(season) - 1]
            baseline = history_df[history_df["season"] == prev_season].groupby("element")["total_points"].mean()
    return baseline


def predict_gameweek_from_history(
    history_df: pd.DataFrame,
    season: str,
    gw: int,
    *,
    n_sims: int = 2000,
    half_life_days: float = 250.0,
    recent_window: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    """Replay the full projection pipeline for one historical (season, gw), leak-free.

    Team-strength priors and rate/minutes priors are fit using every season
    strictly before ``season`` plus that season's own gameweeks before ``gw``
    for team-strength (matches are just dated events, so this is a plain date
    cutoff); rate and minutes *priors* are fit on seasons strictly before
    ``season`` only, so the population-level prior itself never sees the
    target season even indirectly.
    """
    from ._teamnames import normalize_team

    cutoff = _season_match_dates(history_df, season, gw)
    if cutoff is None:
        return pd.DataFrame()

    all_seasons = sorted(history_df["season"].unique())
    prior_seasons = tuple(s for s in all_seasons if s < season)

    matches = ts.build_match_table(history_df)
    train_matches = matches[matches["date"] < cutoff]
    if train_matches.empty:
        return pd.DataFrame()

    prior_attack, prior_concede, promoted, baseline = ts._priors_for_fit(train_matches)
    team_model = ts.fit(
        train_matches, as_of=cutoff, half_life_days=half_life_days,
        prior_attack=prior_attack, prior_concede=prior_concede,
        promoted_teams=promoted, promoted_baseline=baseline,
    )

    # "current season" window: that season's own rows strictly before gw.
    season_prior = history_df[(history_df["season"] == season) & (history_df["gw"] < gw)]
    if season_prior.empty and gw > 1:
        return pd.DataFrame()

    latest = season_prior.sort_values("gw").groupby("element").last()
    elements = pd.DataFrame(
        {
            "id": latest.index.astype(int),
            "element_type": latest["position"].map(_POSITION_CODE),
            "team": latest["team"].map(normalize_team),
            "team_name": latest["team"].map(normalize_team),
            "full_name": "",
            "now_cost": latest["value"].fillna(50),
            "status": "a",
            "chance_of_playing_next_round": np.nan,
        }
    ).dropna(subset=["element_type"])
    elements["element_type"] = elements["element_type"].astype(int)
    cum = season_prior.groupby("element")[list(_RATE_STATS) + ["minutes"]].sum()
    elements = elements.merge(cum, left_on="id", right_index=True, how="left").fillna(0.0)

    recent_gws = sorted(season_prior["gw"].dropna().unique())[-recent_window:]
    gw_history = season_prior[season_prior["gw"].isin(recent_gws)][["element", "gw", "minutes", "starts"]]

    minutes_priors = mm.fit_position_priors(history_df, seasons=tuple(s for s in prior_seasons if s >= "2022-23") or prior_seasons)
    minutes_df = mm.estimate_minutes(elements, gw_history, n_gws_so_far=len(recent_gws), priors=minutes_priors)

    rate_priors = rr.fit_rate_priors(history_df, seasons=prior_seasons)
    rates_df = rr.estimate_rates(elements, rate_priors, understat_matched=None)

    card_og_rates = sim.fit_card_and_og_rates(history_df, seasons=prior_seasons)
    assisted_fraction = sim.league_assisted_goal_fraction(history_df, seasons=prior_seasons)

    player_frame = minutes_df.merge(rates_df, on="player_id", how="left")
    fixtures = _target_fixtures(history_df, season, gw)
    if fixtures.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    summary = sim.simulate_gameweek(
        fixtures, team_model, player_frame, n_sims, rng,
        card_og_rates=card_og_rates, assisted_fraction=assisted_fraction,
    )
    summary["season"] = season
    summary["gw"] = gw
    return summary


def calibration_table(predictions: pd.DataFrame, actual_col: str = "actual_points") -> pd.DataFrame:
    """Bucketed calibration check: do p10/p25/p50/p75/p90 contain what they claim?

    For each quantile level, reports the fraction of rows whose actual points
    fell at or below the predicted quantile -- this should equal the nominal
    level (0.10, 0.25, ...) if the distribution is well calibrated.
    """
    rows = []
    for level, col in ((0.10, "p10"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.90, "p90")):
        covered = (predictions[actual_col] <= predictions[col]).mean()
        rows.append({"nominal_quantile": level, "column": col, "empirical_coverage": covered, "n": len(predictions)})
    # Interval coverage: does the [p10, p90] band actually contain ~80% of outcomes?
    band_80 = ((predictions[actual_col] >= predictions["p10"]) & (predictions[actual_col] <= predictions["p90"])).mean()
    band_50 = ((predictions[actual_col] >= predictions["p25"]) & (predictions[actual_col] <= predictions["p75"])).mean()
    rows.append({"nominal_quantile": 0.80, "column": "[p10,p90] band", "empirical_coverage": band_80, "n": len(predictions)})
    rows.append({"nominal_quantile": 0.50, "column": "[p25,p75] band", "empirical_coverage": band_50, "n": len(predictions)})
    return pd.DataFrame(rows)


def _errors(pred: pd.Series, actual: pd.Series) -> dict:
    diff = pred.to_numpy() - actual.to_numpy()
    return {"rmse": float(np.sqrt(np.mean(diff**2))), "mae": float(np.mean(np.abs(diff))), "n": len(diff)}


def run_backtest(
    history_df: pd.DataFrame,
    targets: Sequence[tuple[str, int]],
    *,
    n_sims: int = 2000,
    min_relevance_minutes: float = 1.0,
) -> dict:
    """Run :func:`predict_gameweek_from_history` over every target and score it.

    A player-gameweek is included in the evaluated set only if the player had
    already logged minutes earlier that season OR actually played in the
    target gameweek -- this excludes the large population of players who
    never feature at all (correctly, trivially predicted as ~0 by every
    method, which would otherwise flatter every RMSE equally and hide real
    differences).
    """
    all_preds = []
    per_target = []
    for season, gw in targets:
        t0 = time.monotonic()
        pred = predict_gameweek_from_history(history_df, season, gw, n_sims=n_sims, seed=hash((season, gw)) % (2**32))
        if pred.empty:
            log.warning("no prediction produced for %s GW%d; skipping", season, gw)
            continue
        actual = rescore_actual(history_df, season, gw)
        merged = pred.merge(actual, left_on="player_id", right_on="element", how="left")
        merged["actual_points"] = merged["actual_points"].fillna(0)
        merged["actual_minutes"] = merged["actual_minutes"].fillna(0)

        season_prior_minutes = (
            history_df[(history_df["season"] == season) & (history_df["gw"] < gw)]
            .groupby("element")["minutes"].sum()
        )
        relevant = merged["player_id"].map(season_prior_minutes).fillna(0) > 0
        relevant |= merged["actual_minutes"] >= min_relevance_minutes
        merged = merged[relevant].copy()

        baseline = season_mean_baseline(history_df, season, gw)
        merged["baseline_season_mean"] = merged["player_id"].map(baseline)

        all_preds.append(merged)
        per_target.append(
            {
                "season": season, "gw": gw, "n_players": len(merged),
                "model": _errors(merged["ep"], merged["actual_points"]),
                "baseline_season_mean": _errors(
                    merged["baseline_season_mean"].fillna(merged["ep"].mean()), merged["actual_points"]
                ),
                "seconds": time.monotonic() - t0,
            }
        )
        log.info("%s GW%d: %d relevant players in %.1fs", season, gw, len(merged), time.monotonic() - t0)

    combined = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    overall = {
        "model": _errors(combined["ep"], combined["actual_points"]) if len(combined) else {},
        "baseline_season_mean": (
            _errors(combined["baseline_season_mean"].fillna(combined["ep"].mean()), combined["actual_points"])
            if len(combined)
            else {}
        ),
    }
    calibration = calibration_table(combined) if len(combined) else pd.DataFrame()
    return {"per_target": per_target, "overall": overall, "combined": combined, "calibration": calibration}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--secondary-gws", default="8,14,20,26,32", help="2025-26 gameweeks for the secondary backtest")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    history_df = store.load("history/merged_gw")

    print("=" * 78)
    print("PRIMARY BACKTEST: 2026-27 GW1 (zero current-season signal) and GW2")
    print("=" * 78)
    # The vaastav archive lags the live API: 2026-27 currently holds GW1 only,
    # and GW1 has no within-season signal to predict from. Targeting it produced
    # an empty result that printed as success. Use whatever the archive actually
    # has, and say so when there is nothing to run.
    available = sorted(
        int(g) for g in history_df.loc[history_df["season"] == "2026-27", "gw"].unique()
    )
    primary_targets = [("2026-27", g) for g in available if g >= 2]
    if not primary_targets:
        print(
            f"skipped: the 2026-27 archive holds gameweeks {available}, and a "
            f"backtest needs at least one gameweek with prior-season-week data "
            f"(GW2+). The secondary backtest below carries the evidence."
        )
        primary = None
    else:
        primary = run_backtest(history_df, primary_targets, n_sims=args.n_sims)
    if primary is not None:
        for row in primary["per_target"]:
            print(row)
        print("overall:", primary["overall"])
        print("\ncalibration:\n", primary["calibration"].to_string())

    print("\n" + "=" * 78)
    print("SECONDARY BACKTEST: 2025-26, spread gameweeks (defcon-rescored, bonus-vintage caveat)")
    print("=" * 78)
    secondary_gws = [int(g) for g in args.secondary_gws.split(",")]
    secondary = run_backtest(history_df, [("2025-26", g) for g in secondary_gws], n_sims=args.n_sims)
    for row in secondary["per_target"]:
        print(row)
    print("overall:", secondary["overall"])
    print("\ncalibration:\n", secondary["calibration"].to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
