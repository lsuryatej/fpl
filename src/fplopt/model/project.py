"""CLI + library entry point: full-distribution FPL point projections.

    uv run python -m fplopt.model.project --gws 3,4,5,6,7,8,9,10 --n-sims 5000

Wires together every other module in this package:

  1. :mod:`team_strength` -- one Dixon-Coles fit "as of now", reused for every
     requested gameweek. Team ratings move slowly enough over an 8-gameweek
     horizon (see the chosen ~250-day decay half-life) that refitting per
     gameweek would cost time for essentially no gain, since there are no new
     results to add between now and any of the requested gameweeks anyway.
  2. :mod:`minutes` and :mod:`rates` -- a single current-season snapshot
     (today's ``status``, this season's minutes/goals/defensive actions to
     date), also reused for every requested gameweek. This is a real
     limitation, not just a performance shortcut: the model has no way to
     anticipate a new injury or a rotation change three gameweeks out, so a
     player's per-gameweek row looks identical across the whole horizon
     except for who they are actually facing. See the project verification
     report for the honest accounting of what this costs in practice.
  3. :mod:`simulate` -- the Monte Carlo engine, run independently per
     requested gameweek against that gameweek's actual fixture list (which
     the FPL fixtures endpoint already publishes for the whole season).

Output schema (exactly, per the optimiser this feeds):
    player_id, gw, position, team, price, p_play, p_start, exp_minutes,
    ep, sd, p10, p25, p50, p75, p90
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd

from ..data import fpl_api, history, store
from ..data.http import FetchError
from . import minutes as mm
from . import rates as rr
from . import simulate as sim
from . import team_strength as ts
from ._teamnames import normalize_team

__all__ = ["project", "load_pipeline_inputs", "main"]

log = logging.getLogger(__name__)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "player_id", "gw", "position", "team", "price",
    "p_play", "p_start", "exp_minutes",
    "ep", "sd", "p10", "p25", "p50", "p75", "p90",
)

DEFAULT_HALF_LIFE_DAYS = 250.0  # see team_strength.fit_decay_half_life's report for how this was chosen


def load_pipeline_inputs(*, live_season: str | None = None) -> dict:
    """Fetch/load every dataset the pipeline needs, using the existing data layer's cache.

    Never re-derives anything the data layer already owns: history and FPL
    bootstrap/fixtures/teams come from ``fplopt.data.store`` when cached and
    are refreshed through ``fplopt.data`` clients otherwise; per-gameweek live
    stats and Understat are always fetched fresh (both are cheap and cached
    per-day by the HTTP layer already).
    """
    client = fpl_api.FPLClient()
    try:
        boot = client.bootstrap()
        elements = fpl_api.elements_frame(boot)
        teams = fpl_api.teams_frame(boot)
        events = fpl_api.events_frame(boot)
        raw_fixtures = client.fixtures()
        fixtures = fpl_api.fixtures_frame(raw_fixtures)

        if live_season is None:
            # "2026-27" -> understat "2026"; inferred from the current event's
            # season rather than hardcoded so this keeps working next season.
            live_season = history.SEASONS[-1]
        understat_year = live_season.split("-")[0]

        now = pd.Timestamp.now(tz="UTC")
        deadlines = pd.to_datetime(events["deadline_time"], utc=True)
        n_gws_so_far = int((deadlines <= now).sum())

        gw_history_frames = []
        for gw in range(1, n_gws_so_far + 1):
            try:
                payload = client.live(gw)
            except FetchError as exc:
                log.warning("live GW%s unavailable (%s); minutes model will treat it as unplayed", gw, exc)
                continue
            gw_history_frames.append(fpl_api.live_points_frame(payload, gw))
        gw_history = (
            pd.concat(gw_history_frames, ignore_index=True) if gw_history_frames else pd.DataFrame(columns=["element", "minutes", "starts"])
        )
    finally:
        client.close()

    if store.exists("history/merged_gw"):
        hist = store.load("history/merged_gw")
    else:
        log.info("no cached history; downloading the full archive (one-off, ~20s)")
        hist = history.refresh()

    try:
        understat_players = store.load(f"understat/players_{understat_year}")
    except FileNotFoundError:
        from ..data import understat as understat_mod

        understat_players = understat_mod.league_players(understat_year)
        store.save(understat_players, f"understat/players_{understat_year}")

    return {
        "elements": elements,
        "teams": teams,
        "events": events,
        "fixtures": fixtures,
        "history": hist,
        "understat_players": understat_players,
        "gw_history": gw_history,
        "n_gws_so_far": n_gws_so_far,
        "live_season": live_season,
        "as_of": now,
    }


def project(
    gws: Sequence[int],
    n_sims: int = 5000,
    *,
    inputs: dict | None = None,
    half_life_days: float | None = None,
    seed: int = 20260901,
) -> pd.DataFrame:
    """Simulate every requested gameweek and return one combined DataFrame.

    ``inputs`` lets a caller (tests, the backtest script) reuse an already
    -loaded :func:`load_pipeline_inputs` result instead of hitting the network
    again. ``half_life_days`` defaults to the value
    ``team_strength.fit_decay_half_life`` selected on the full history (see
    the project verification report); pass a value to skip that search when
    it has already been run once and cached upstream.
    """
    data = inputs if inputs is not None else load_pipeline_inputs()
    elements, teams, fixtures_all = data["elements"], data["teams"], data["fixtures"]
    hist, understat_players = data["history"], data["understat_players"]
    gw_history, n_gws_so_far = data["gw_history"], data["n_gws_so_far"]
    live_season, as_of = data["live_season"], data["as_of"]

    id_to_canon = {i: normalize_team(n) for i, n in zip(teams["id"], teams["name"])}

    matches = ts.build_match_table(hist, live_season=live_season, live_fixtures_df=fixtures_all, live_teams_df=teams)
    half_life_days = half_life_days if half_life_days is not None else DEFAULT_HALF_LIFE_DAYS
    prior_attack, prior_concede, promoted, baseline = ts._priors_for_fit(matches)
    team_model = ts.fit(
        matches, as_of=as_of, half_life_days=half_life_days,
        prior_attack=prior_attack, prior_concede=prior_concede,
        promoted_teams=promoted, promoted_baseline=baseline,
    )
    log.info(
        "team-strength fit: %d matches, home_adv=%.3f rho=%.3f converged=%s",
        team_model.n_matches, team_model.home_adv, team_model.rho, team_model.converged,
    )

    minutes_priors = mm.fit_position_priors(hist, early_gws=min(n_gws_so_far, 6) or 2)
    minutes_df = mm.estimate_minutes(elements, gw_history, n_gws_so_far, priors=minutes_priors)

    rate_priors = rr.fit_rate_priors(hist)
    understat_matched = rr.match_understat(elements, understat_players)
    rates_df = rr.estimate_rates(elements, rate_priors, understat_matched)

    card_og_rates = sim.fit_card_and_og_rates(hist)
    assisted_fraction = sim.league_assisted_goal_fraction(hist)

    player_frame = minutes_df.merge(rates_df, on="player_id", how="left")

    rng = np.random.default_rng(seed)
    all_rows: list[pd.DataFrame] = []
    for gw in gws:
        gw_fixtures = fixtures_all[fixtures_all["event"] == gw].copy()
        if gw_fixtures.empty:
            log.warning("gameweek %s has no scheduled fixtures; skipping", gw)
            continue
        gw_fixtures["home_team"] = gw_fixtures["team_h"].map(id_to_canon)
        gw_fixtures["away_team"] = gw_fixtures["team_a"].map(id_to_canon)
        gw_fixtures = gw_fixtures.rename(columns={"team_h": "home_team_id", "team_a": "away_team_id"})

        t0 = time.monotonic()
        summary = sim.simulate_gameweek(
            gw_fixtures, team_model, player_frame, n_sims, rng,
            card_og_rates=card_og_rates, assisted_fraction=assisted_fraction,
        )
        summary["gw"] = gw
        all_rows.append(summary)
        log.info("gw %s: simulated %d players in %.1fs", gw, len(summary), time.monotonic() - t0)

    if not all_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = pd.concat(all_rows, ignore_index=True)
    result = result[list(OUTPUT_COLUMNS)]
    for col in ("player_id", "gw", "position", "team", "price"):
        result[col] = result[col].astype(int)
    return result.sort_values(["gw", "player_id"]).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fplopt.model.project", description=__doc__)
    parser.add_argument("--gws", required=True, help="comma-separated gameweek ids, e.g. 3,4,5,6,7,8,9,10")
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--half-life-days", type=float, default=None)
    parser.add_argument("--out", default=None, help="output dataset name under data/ (default: projections/gw<first>-<last>)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
    )

    gws = [int(g) for g in args.gws.split(",") if g.strip()]
    t0 = time.monotonic()
    result = project(gws, n_sims=args.n_sims, half_life_days=args.half_life_days)
    elapsed = time.monotonic() - t0

    out_name = args.out or f"projections/gw{gws[0]}-{gws[-1]}"
    out_name = out_name[:-len(".parquet")] if out_name.endswith(".parquet") else out_name
    path = store.save(result, out_name)

    print(f"wrote {len(result)} rows across {len(gws)} gameweek(s) to {path} in {elapsed:.1f}s")
    print(f"columns: {list(result.columns)}")
    if not result.empty:
        print(result.head(5).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
