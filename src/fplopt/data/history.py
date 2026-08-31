"""Loader for the vaastav/Fantasy-Premier-League GitHub archive.

Source layout::

    data/{season}/gws/merged_gw.csv    per-player per-gameweek rows
    data/{season}/players_raw.csv      end-of-season player index
    data/{season}/teams.csv            team index (absent before 2019-20)

The archive's schema drifted a lot over the years, so :func:`load_all` does real
reconciliation rather than a naive concat:

* ``name`` is ``"Aaron_Cresswell"`` in 2016-17, ``"Aaron_Cresswell_376"``
  (trailing element id) in 2017-18 through 2020-21, and plain
  ``"Nathan Redmond"`` from 2021-22 onward. It is normalised to spaces, with
  the trailing id stripped into its own column when present.
* ``position`` and ``team`` (name) columns only exist from 2021-22. For earlier
  seasons they are backfilled by joining ``players_raw.csv`` on ``element``.
* ``xP`` and the ``expected_*`` family start in 2021-22 / 2022-23 respectively.
* ``defensive_contribution``/``tackles``/``recoveries``/``clearances_blocks_interceptions``
  reappear in 2025-26 (they existed with a different set in 2016-17 then vanished).
* Manager-scoring columns (``mng_*``) exist only in 2024-25 and 2025-26.
* 2016-17 through 2018-19 CSVs are not UTF-8; they are latin-1.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from . import store
from .http import FetchError, HTTPStatusError, RateLimitedSession

__all__ = [
    "SEASONS",
    "RAW_BASE",
    "HistoryLoader",
    "load_season",
    "load_all",
    "refresh",
    "season_summary",
    "CANONICAL_COLUMNS",
]

log = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

#: Seasons known to exist in the archive, oldest first.
SEASONS: tuple[str, ...] = (
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
    "2026-27",
)

#: Columns guaranteed present on the tidy output frame, in order.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "season",
    "gw",
    "element",
    "name",
    "position",
    "team",
    "opponent_team",
    "fixture",
    "kickoff_time",
    "was_home",
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
    "xp",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "team_h_score",
    "team_a_score",
)

_NUMERIC_COLUMNS = frozenset(CANONICAL_COLUMNS) - {
    "season",
    "name",
    "position",
    "team",
    "kickoff_time",
    "was_home",
}

# Source column -> canonical column. Anything not listed is passed through
# lowercased so extra per-season columns survive without colliding.
_RENAMES: dict[str, str] = {
    "GW": "gw",
    "round": "round_num",
    "xP": "xp",
}

# Columns in the old (2016-17) schema that duplicate information we already
# carry under a canonical name, or that no longer mean anything.
_DROP = frozenset({"id", "kickoff_time_formatted", "modified"})


def _decode(raw: bytes) -> str:
    """Decode an archive CSV, falling back to latin-1 for the pre-2019 files."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 cannot actually fail, but be explicit rather than silently lossy.
    return raw.decode("latin-1", errors="replace")


def _split_trailing_id(name: str) -> tuple[str, int | None]:
    """Split ``"Aaron_Cresswell_376"`` into ``("Aaron Cresswell", 376)``.

    Names without a trailing integer come back with ``None`` and underscores
    turned into spaces.
    """
    if not isinstance(name, str):
        return "", None
    parts = name.split("_")
    trailing: int | None = None
    if len(parts) > 1 and parts[-1].isdigit():
        trailing = int(parts[-1])
        parts = parts[:-1]
    return " ".join(p for p in parts if p).strip(), trailing


@dataclass
class HistoryLoader:
    """Downloads and normalises the vaastav archive.

    Parameters
    ----------
    seasons:
        Which seasons to consider. Defaults to every season in :data:`SEASONS`.
    min_interval:
        Seconds between GitHub raw requests.
    """

    seasons: tuple[str, ...] = SEASONS
    min_interval: float = 0.4
    cache_enabled: bool = True
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session = RateLimitedSession(
            min_interval=self.min_interval,
            cache_subdir="history",
            cache_enabled=self.cache_enabled,
            extra_headers={"Accept": "text/plain, */*"},
        )
        self._master_teams: dict[tuple[str, int], str] | None = None

    # ------------------------------------------------------------------
    def _fetch_csv(self, season: str, relpath: str, required: bool) -> pd.DataFrame | None:
        """Fetch and parse one archive CSV.

        Returns ``None`` when the file is absent (404) and ``required`` is
        ``False`` -- older seasons genuinely lack ``teams.csv``.
        """
        url = f"{RAW_BASE}/{season}/{relpath}"
        try:
            raw = self.session.get_bytes(url, suffix=".csv")
        except HTTPStatusError as exc:
            if exc.status_code == 404 and not required:
                log.info("%s has no %s (HTTP 404), continuing", season, relpath)
                return None
            self.errors.append(f"{season}/{relpath}: {exc}")
            if required:
                raise
            return None
        except FetchError as exc:
            self.errors.append(f"{season}/{relpath}: {exc}")
            if required:
                raise
            return None

        text = _decode(raw)
        try:
            return pd.read_csv(io.StringIO(text), low_memory=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            message = f"{season}/{relpath}: could not parse CSV: {exc}"
            self.errors.append(message)
            if required:
                raise FetchError(message) from exc
            return None

    # ------------------------------------------------------------------
    def players_raw(self, season: str) -> pd.DataFrame | None:
        """Return ``players_raw.csv`` for ``season``, or ``None`` if unavailable."""
        return self._fetch_csv(season, "players_raw.csv", required=False)

    def master_team_list(self) -> dict[tuple[str, int], str]:
        """Return ``(season, team_id) -> team_name`` from ``data/master_team_list.csv``.

        This file covers 2016-17 through 2023-24 and is the only way to name
        teams for the three seasons that ship no ``teams.csv``. Cached on the
        instance because every season lookup wants it.
        """
        if self._master_teams is not None:
            return self._master_teams
        url = f"{RAW_BASE}/master_team_list.csv"
        mapping: dict[tuple[str, int], str] = {}
        try:
            raw = self.session.get_bytes(url, suffix=".csv")
            frame = pd.read_csv(io.StringIO(_decode(raw)))
        except (FetchError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            log.warning("master_team_list.csv unavailable: %s", exc)
            self.errors.append(f"master_team_list.csv: {exc}")
            self._master_teams = mapping
            return mapping
        for _, row in frame.iterrows():
            try:
                mapping[(str(row["season"]), int(row["team"]))] = str(row["team_name"])
            except (KeyError, TypeError, ValueError):
                continue
        self._master_teams = mapping
        return mapping

    def teams(self, season: str) -> pd.DataFrame | None:
        """Return ``teams.csv`` for ``season``, or ``None`` for seasons that lack it.

        Column *order* changed in 2026-27 and ``strength`` is left blank until
        the season has played out, so downstream code must select by name and
        tolerate nulls.
        """
        frame = self._fetch_csv(season, "teams.csv", required=False)
        if frame is None:
            return None
        frame = frame.copy()
        frame["season"] = season
        return frame

    def merged_gw(self, season: str) -> pd.DataFrame:
        """Return the raw (un-normalised) ``gws/merged_gw.csv`` for ``season``."""
        frame = self._fetch_csv(season, "gws/merged_gw.csv", required=True)
        if frame is None:  # pragma: no cover - required=True raises instead
            raise FetchError(f"{season}: merged_gw.csv unavailable")
        return frame

    # ------------------------------------------------------------------
    def load_season(self, season: str) -> pd.DataFrame:
        """Return one season's gameweek rows normalised to :data:`CANONICAL_COLUMNS`."""
        raw = self.merged_gw(season)
        lookup = self._position_team_lookup(season)
        return normalise_season(raw, season, lookup)

    def _position_team_lookup(self, season: str) -> pd.DataFrame | None:
        """Build an ``element -> (position, team)`` map from players_raw + teams.

        Only needed for 2016-17 through 2020-21, where ``merged_gw.csv`` has
        neither column. Returns ``None`` when the inputs are unavailable.
        """
        players = self.players_raw(season)
        if players is None or "id" not in players.columns:
            return None
        cols = players.columns
        out = pd.DataFrame({"element": pd.to_numeric(players["id"], errors="coerce")})

        if "element_type" in cols:
            position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "MNG"}
            out["position_lookup"] = (
                pd.to_numeric(players["element_type"], errors="coerce").map(position_map)
            )
        if "team" in cols:
            team_ids = pd.to_numeric(players["team"], errors="coerce")
            names: dict[float, str] = {}
            teams = self.teams(season)
            if teams is not None and {"id", "name"} <= set(teams.columns):
                names = dict(
                    zip(pd.to_numeric(teams["id"], errors="coerce"), teams["name"].astype(str))
                )
            if not names:
                # 2016-17 through 2018-19 ship no teams.csv; the repo-level
                # master_team_list.csv covers exactly those seasons.
                master = self.master_team_list()
                names = {
                    float(tid): name
                    for (season_key, tid), name in master.items()
                    if season_key == season
                }
            mapped = team_ids.map(names)
            # Anything still unnamed keeps the numeric id so the column is never
            # null and remains joinable against the raw archive.
            out["team_lookup"] = mapped.fillna(team_ids.astype("Int64").astype(str))
        return out.dropna(subset=["element"])

    # ------------------------------------------------------------------
    def load_all(self, seasons: Iterable[str] | None = None) -> pd.DataFrame:
        """Load and concatenate every requested season into one tidy frame.

        A season that fails to download is logged into ``self.errors`` and
        skipped rather than aborting the whole load.
        """
        frames: list[pd.DataFrame] = []
        for season in seasons or self.seasons:
            try:
                frames.append(self.load_season(season))
                log.info("loaded %s (%d rows)", season, len(frames[-1]))
            except FetchError as exc:
                log.error("skipping %s: %s", season, exc)
                self.errors.append(f"{season}: {exc}")
        if not frames:
            raise FetchError("No historical seasons could be loaded")
        combined = pd.concat(frames, ignore_index=True, sort=False)
        return _order_columns(combined)

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self.session.close()


# ----------------------------------------------------------------------
# pure normalisation (no network, fully unit-testable)
# ----------------------------------------------------------------------
def normalise_season(
    raw: pd.DataFrame, season: str, lookup: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Reconcile one season's ``merged_gw`` frame onto the canonical schema.

    Parameters
    ----------
    raw:
        The frame exactly as parsed from ``merged_gw.csv``.
    season:
        Season label such as ``"2019-20"``, written into the ``season`` column.
    lookup:
        Optional frame with ``element`` plus ``position_lookup``/``team_lookup``
        used to backfill the two columns missing before 2021-22.
    """
    frame = raw.copy()
    frame = frame.drop(columns=[c for c in frame.columns if c in _DROP], errors="ignore")
    frame = frame.rename(columns=_RENAMES)
    # Everything else goes lowercase so e.g. "GW"/"gw" cannot both survive.
    frame.columns = [c if c in _RENAMES.values() else c.lower() for c in frame.columns]

    # --- name -----------------------------------------------------------
    if "name" in frame.columns:
        split = frame["name"].map(_split_trailing_id)
        frame["name"] = [s[0] for s in split]
        trailing = pd.Series([s[1] for s in split], index=frame.index, dtype="Float64")
    else:
        frame["name"] = pd.NA
        trailing = pd.Series(pd.NA, index=frame.index, dtype="Float64")

    # --- element --------------------------------------------------------
    if "element" in frame.columns:
        frame["element"] = pd.to_numeric(frame["element"], errors="coerce").astype("Int64")
    else:
        frame["element"] = trailing.astype("Int64")

    # --- gameweek -------------------------------------------------------
    if "gw" not in frame.columns and "round_num" in frame.columns:
        frame["gw"] = frame["round_num"]
    frame["gw"] = pd.to_numeric(frame.get("gw"), errors="coerce").astype("Int64")

    # --- position / team backfill --------------------------------------
    if lookup is not None and not lookup.empty:
        merged = frame.merge(
            lookup.assign(element=lookup["element"].astype("Int64")),
            on="element",
            how="left",
        )
        if "position" not in merged.columns and "position_lookup" in merged.columns:
            merged["position"] = merged["position_lookup"]
        elif "position_lookup" in merged.columns:
            merged["position"] = merged["position"].fillna(merged["position_lookup"])
        if "team" not in merged.columns and "team_lookup" in merged.columns:
            merged["team"] = merged["team_lookup"]
        elif "team_lookup" in merged.columns:
            merged["team"] = merged["team"].fillna(merged["team_lookup"])
        frame = merged.drop(columns=["position_lookup", "team_lookup"], errors="ignore")

    # Normalise position labels: the archive uses GK in players_raw but GKP in
    # the modern merged_gw files.
    if "position" in frame.columns:
        frame["position"] = (
            frame["position"].astype("string").str.upper().replace({"GKP": "GK", "GOALKEEPER": "GK"})
        )

    # --- typing ---------------------------------------------------------
    frame["season"] = season
    if "kickoff_time" in frame.columns:
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], errors="coerce", utc=True)
    if "was_home" in frame.columns:
        frame["was_home"] = (
            frame["was_home"].astype("string").str.lower().map({"true": True, "false": False})
        ).astype("boolean")

    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    for column in _NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return _order_columns(frame)


def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the canonical columns first, then any season-specific extras."""
    canonical = [c for c in CANONICAL_COLUMNS if c in frame.columns]
    extras = sorted(c for c in frame.columns if c not in CANONICAL_COLUMNS)
    return frame[canonical + extras]


def season_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-season row counts, gameweek span and player/date coverage."""
    if frame.empty:
        return pd.DataFrame(columns=["season", "rows", "players", "gw_min", "gw_max"])
    grouped = frame.groupby("season", dropna=False)
    out = pd.DataFrame(
        {
            "rows": grouped.size(),
            "players": grouped["element"].nunique(),
            "gw_min": grouped["gw"].min(),
            "gw_max": grouped["gw"].max(),
        }
    ).reset_index()
    if "kickoff_time" in frame.columns:
        out = out.merge(
            grouped["kickoff_time"].agg(["min", "max"]).reset_index(),
            on="season",
            how="left",
        ).rename(columns={"min": "first_kickoff", "max": "last_kickoff"})
    return out.sort_values("season").reset_index(drop=True)


# ----------------------------------------------------------------------
# module-level convenience
# ----------------------------------------------------------------------
def load_season(season: str) -> pd.DataFrame:
    """Download and normalise a single season."""
    loader = HistoryLoader()
    try:
        return loader.load_season(season)
    finally:
        loader.close()


def load_all(seasons: Iterable[str] | None = None) -> pd.DataFrame:
    """Download and normalise every season into one frame."""
    loader = HistoryLoader()
    try:
        return loader.load_all(seasons)
    finally:
        loader.close()


def refresh(
    seasons: Iterable[str] | None = None, name: str = "history/merged_gw"
) -> pd.DataFrame:
    """Load every season, persist it as parquet and return the frame.

    Also writes ``history/teams`` with the per-season team index for every
    season that publishes one.
    """
    loader = HistoryLoader()
    try:
        frame = loader.load_all(seasons)
        store.save(frame, name)

        team_frames = []
        for season in seasons or loader.seasons:
            teams = loader.teams(season)
            if teams is not None:
                team_frames.append(teams)
        if team_frames:
            store.save(
                pd.concat(team_frames, ignore_index=True, sort=False), "history/teams"
            )
        if loader.errors:
            log.warning("history refresh completed with %d issue(s)", len(loader.errors))
        return frame
    finally:
        loader.close()
