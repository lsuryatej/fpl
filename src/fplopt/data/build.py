"""Refresh every data source and print a summary of what was fetched.

Run with::

    uv run python -m fplopt.data.build                # everything
    uv run python -m fplopt.data.build --skip understat
    uv run python -m fplopt.data.build --understat-seasons 2025 2026
    uv run python -m fplopt.data.build --shot-limit 20

Each source is refreshed independently; a failure in one is reported in the
summary and does not stop the others.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import pandas as pd

from . import fbref, fpl_api, history, store, understat
from .http import FetchError
from .paths import data_dir

__all__ = ["SourceResult", "refresh_fpl", "refresh_history", "refresh_understat", "main"]

log = logging.getLogger(__name__)

#: Understat seasons refreshed by default. The full archive back to 2014 is
#: available but a whole-history shot sweep is ~4700 requests at 1 req/s.
DEFAULT_UNDERSTAT_SEASONS: tuple[str, ...] = ("2024", "2025", "2026")


@dataclass
class SourceResult:
    """Outcome of refreshing one source."""

    source: str
    ok: bool
    datasets: dict[str, pd.DataFrame] = field(default_factory=dict)
    note: str = ""
    seconds: float = 0.0


# ----------------------------------------------------------------------
def refresh_fpl() -> SourceResult:
    """Pull bootstrap, fixtures and every finished gameweek's live feed."""
    started = time.monotonic()
    client = fpl_api.FPLClient()
    datasets: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    try:
        boot = client.bootstrap()
        datasets["fpl/elements"] = fpl_api.elements_frame(boot)
        datasets["fpl/teams"] = fpl_api.teams_frame(boot)
        datasets["fpl/events"] = fpl_api.events_frame(boot)
        datasets["fpl/element_types"] = pd.DataFrame(boot.get("element_types", []))
        datasets["fpl/chips"] = pd.DataFrame(boot.get("chips", []))

        raw_fixtures = client.fixtures()
        datasets["fpl/fixtures"] = fpl_api.fixtures_frame(raw_fixtures)
        datasets["fpl/fixture_stats"] = fpl_api.fixture_stats_frame(raw_fixtures)

        done = fpl_api.finished_events(boot)
        current = fpl_api.current_event(boot)
        notes.append(f"current GW {current}, finished+checked {done or 'none'}")

        live_stats: list[pd.DataFrame] = []
        live_explain: list[pd.DataFrame] = []
        for gw in done:
            try:
                payload = client.live(gw)
            except FetchError as exc:
                log.error("live GW%s failed: %s", gw, exc)
                notes.append(f"live GW{gw} failed")
                continue
            live_stats.append(fpl_api.live_points_frame(payload, gw))
            live_explain.append(fpl_api.live_explain_frame(payload, gw))
        if live_stats:
            datasets["fpl/live_stats"] = pd.concat(live_stats, ignore_index=True)
            datasets["fpl/live_explain"] = pd.concat(live_explain, ignore_index=True)
            notes.append(_verify_live(datasets["fpl/live_stats"], datasets["fpl/live_explain"]))

        for name, frame in datasets.items():
            store.save(frame, name)
        return SourceResult("fpl_api", True, datasets, "; ".join(notes), time.monotonic() - started)
    except FetchError as exc:
        return SourceResult("fpl_api", False, datasets, f"FAILED: {exc}", time.monotonic() - started)
    finally:
        client.close()


def _verify_live(stats: pd.DataFrame, explain: pd.DataFrame) -> str:
    """Check that the explain breakdown reconstructs ``total_points``.

    This is the ground-truth assertion the whole scoring layer is validated
    against, so it runs on every refresh and reports mismatches by count.
    """
    if stats.empty or explain.empty:
        return "live verify: no data"
    explain = explain.copy()
    explain["contrib"] = pd.to_numeric(explain["points"], errors="coerce").fillna(0) + pd.to_numeric(
        explain["points_modification"], errors="coerce"
    ).fillna(0)
    rebuilt = explain.groupby(["gw", "element"], as_index=False)["contrib"].sum()
    merged = stats[["gw", "element", "total_points"]].merge(
        rebuilt, on=["gw", "element"], how="left"
    )
    merged["contrib"] = merged["contrib"].fillna(0)
    mismatch = int((merged["contrib"] != merged["total_points"]).sum())
    return f"live verify: {len(merged) - mismatch}/{len(merged)} players reconcile exactly"


def refresh_history(seasons: Sequence[str] | None = None) -> SourceResult:
    """Pull the whole vaastav archive and normalise it into one frame."""
    started = time.monotonic()
    loader = history.HistoryLoader()
    try:
        frame = loader.load_all(seasons)
        store.save(frame, "history/merged_gw")
        datasets = {"history/merged_gw": frame}

        team_frames = []
        for season in seasons or loader.seasons:
            teams = loader.teams(season)
            if teams is not None:
                team_frames.append(teams)
        if team_frames:
            teams_all = pd.concat(team_frames, ignore_index=True, sort=False)
            store.save(teams_all, "history/teams")
            datasets["history/teams"] = teams_all

        note = f"{frame['season'].nunique()} seasons"
        if loader.errors:
            note += f"; {len(loader.errors)} non-fatal issue(s)"
        return SourceResult("history", True, datasets, note, time.monotonic() - started)
    except FetchError as exc:
        return SourceResult("history", False, {}, f"FAILED: {exc}", time.monotonic() - started)
    finally:
        loader.close()


def refresh_understat(
    seasons: Sequence[str] = DEFAULT_UNDERSTAT_SEASONS, shot_limit: int | None = None
) -> SourceResult:
    """Pull Understat league tables, fixtures and shot-level xG per season."""
    started = time.monotonic()
    client = understat.UnderstatClient()
    datasets: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    try:
        for season in seasons:
            season = str(season)
            try:
                players = client.league_players(season)
                dates = client.league_dates(season)
            except FetchError as exc:
                log.error("understat %s failed: %s", season, exc)
                notes.append(f"{season}: FAILED ({exc})")
                continue
            store.save(players, f"understat/players_{season}")
            store.save(dates, f"understat/fixtures_{season}")
            datasets[f"understat/players_{season}"] = players
            datasets[f"understat/fixtures_{season}"] = dates

            ids = dates.loc[dates["is_result"], "match_id"].astype(str).tolist()
            shots = client.season_shots(season, match_ids=ids, limit=shot_limit)
            store.save(shots, f"understat/shots_{season}")
            datasets[f"understat/shots_{season}"] = shots
            notes.append(
                f"{season}: {len(players)} players, {len(ids)} played matches, {len(shots)} shots"
            )
        ok = bool(datasets)
        if client.errors:
            notes.append(f"{len(client.errors)} match-level error(s)")
        return SourceResult("understat", ok, datasets, "; ".join(notes), time.monotonic() - started)
    finally:
        client.close()


def refresh_fbref() -> SourceResult:
    """Probe FBref and report whether it is reachable.

    Never raises: the Cloudflare block is expected and reported as a note.
    """
    started = time.monotonic()
    result = fbref.probe()
    if result["available"]:
        note = "reachable"
    else:
        details = "; ".join(f"{c['path']}: {c['status']}" for c in result["checks"])
        note = f"BLOCKED (Cloudflare JS challenge) -- {details}"
    return SourceResult("fbref", bool(result["available"]), {}, note, time.monotonic() - started)


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def _date_range(frame: pd.DataFrame) -> str:
    """Return a compact date span for whichever date-ish column the frame has."""
    for col in ("kickoff_time", "date", "datetime", "deadline_time"):
        if col in frame.columns and len(frame):
            series = pd.to_datetime(frame[col], errors="coerce", utc=True)
            series = series.dropna()
            if not series.empty:
                return f"{series.min():%Y-%m-%d} .. {series.max():%Y-%m-%d}"
    for col in ("season", "gw", "event"):
        if col in frame.columns and len(frame):
            values = frame[col].dropna()
            if not values.empty:
                return f"{col} {values.min()} .. {values.max()}"
    return "-"


def _render_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    """Render a fixed-width ASCII table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), line]
    out.extend("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in rows)
    return "\n".join(out)


def print_summary(results: list[SourceResult]) -> None:
    """Print the per-dataset and per-source summary tables."""
    print("\n" + "=" * 78)
    print(f"DATA REFRESH SUMMARY  ({data_dir()})")
    print("=" * 78)

    rows: list[tuple[str, ...]] = []
    for result in results:
        for name, frame in sorted(result.datasets.items()):
            age = store.age_days(name)
            rows.append(
                (
                    name,
                    f"{len(frame):,}",
                    str(frame.shape[1]),
                    _date_range(frame),
                    "just now" if age is not None and age < 0.02 else f"{age:.1f}d" if age else "-",
                )
            )
    if rows:
        print("\nDATASETS")
        print(_render_table(rows, ("dataset", "rows", "cols", "range", "freshness")))
    else:
        print("\nDATASETS: none written")

    print("\nSOURCES")
    source_rows = [
        (r.source, "ok" if r.ok else "FAILED", f"{r.seconds:.1f}s", r.note or "-")
        for r in results
    ]
    print(_render_table(source_rows, ("source", "status", "elapsed", "notes")))

    # Per-season historical breakdown is the number most worth eyeballing.
    hist = next((r for r in results if r.source == "history" and r.ok), None)
    if hist and "history/merged_gw" in hist.datasets:
        summary = history.season_summary(hist.datasets["history/merged_gw"])
        print("\nHISTORICAL SEASONS")
        season_rows = [
            (
                str(r["season"]),
                f"{int(r['rows']):,}",
                str(int(r["players"])),
                f"{r['gw_min']:g}-{r['gw_max']:g}",
            )
            for _, r in summary.iterrows()
        ]
        print(_render_table(season_rows, ("season", "rows", "players", "gws")))
    print()


# ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Refresh every source and print the summary. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="fplopt.data.build", description=__doc__)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["fpl", "history", "understat", "fbref"],
        help="sources to skip",
    )
    parser.add_argument(
        "--understat-seasons",
        nargs="*",
        default=list(DEFAULT_UNDERSTAT_SEASONS),
        help="Understat season start-years to refresh",
    )
    parser.add_argument(
        "--history-seasons",
        nargs="*",
        default=None,
        help="archive seasons to refresh (default: all)",
    )
    parser.add_argument(
        "--shot-limit",
        type=int,
        default=None,
        help="cap matches fetched per Understat season (useful for smoke runs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    jobs: list[tuple[str, Callable[[], SourceResult]]] = [
        ("fpl", refresh_fpl),
        ("history", lambda: refresh_history(args.history_seasons)),
        ("understat", lambda: refresh_understat(args.understat_seasons, args.shot_limit)),
        ("fbref", refresh_fbref),
    ]

    results: list[SourceResult] = []
    for key, job in jobs:
        if key in args.skip:
            log.info("skipping %s", key)
            continue
        log.info("refreshing %s", key)
        try:
            results.append(job())
        except Exception as exc:  # noqa: BLE001 - report and continue to the next source
            log.exception("%s raised an unexpected error", key)
            results.append(SourceResult(key, False, {}, f"UNEXPECTED: {type(exc).__name__}: {exc}"))

    print_summary(results)

    # FBref being blocked is expected and must not fail the run.
    fatal = [r for r in results if not r.ok and r.source != "fbref"]
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
