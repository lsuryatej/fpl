"""Understat scraper: league tables, player/team match logs, shot-level xG.

Two extraction paths are implemented, tried in that order:

1. **AJAX JSON endpoints** (current). As of the 2026/27 season Understat serves
   page data from XHR endpoints rather than inlining it, and the pages
   themselves are now empty shells. The endpoints, read out of the site's own
   ``js/{league,match,player,team}.min.js`` bundles, are::

       GET  /getLeagueData/{league}/{season}
       GET  /getMatchData/{match_id}
       GET  /getPlayerData/{player_id}
       GET  /getTeamData/{team_title}/{season}
       POST /main/getPlayersStats/

   They require ``X-Requested-With: XMLHttpRequest`` plus a matching ``Referer``
   and answer gzipped ``text/javascript``.

2. **Legacy embedded blocks**. Older mirrors and archived pages still carry
   ``var shotsData = JSON.parse('\\x7B\\x22...')``. :func:`parse_embedded_json`
   handles the hex-escaped payload via ``codecs.decode(s, "unicode_escape")``.

Season labels are the calendar year the season *starts* in, as strings:
``"2026"`` is the 2026/27 campaign.
"""

from __future__ import annotations

import codecs
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from . import store
from .http import FetchError, RateLimitedSession

__all__ = [
    "BASE_URL",
    "SEASONS",
    "UnderstatError",
    "UnderstatClient",
    "parse_embedded_json",
    "league_players",
    "league_dates",
    "player_matches",
    "team_matches",
    "match_shots",
    "season_shots",
    "refresh",
]

log = logging.getLogger(__name__)

BASE_URL = "https://understat.com"

#: Season start-years Understat publishes for the EPL, oldest first.
SEASONS: tuple[str, ...] = (
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
)

_AJAX_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

_EMBEDDED_RE = re.compile(
    r"var\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*JSON\.parse\(\s*'(?P<payload>.*?)'\s*\)",
    re.DOTALL,
)

#: Columns emitted by :func:`shots_frame`, in order.
SHOT_COLUMNS: tuple[str, ...] = (
    "season",
    "match_id",
    "shot_id",
    "date",
    "minute",
    "player_id",
    "player",
    "team",
    "h_a",
    "h_team",
    "a_team",
    "h_goals",
    "a_goals",
    "result",
    "is_goal",
    "x",
    "y",
    "xg",
    "situation",
    "shot_type",
    "last_action",
    "player_assisted",
)


class UnderstatError(FetchError):
    """Understat returned something we could not turn into data."""


def parse_embedded_json(html: str, var_name: str) -> Any:
    """Extract ``var <var_name> = JSON.parse('...')`` from an Understat page.

    The payload is hex-escaped JavaScript source, so it is first decoded with
    ``unicode_escape`` and then parsed as JSON.

    Raises
    ------
    UnderstatError
        If the variable is absent or its payload is not valid JSON.
    """
    for match in _EMBEDDED_RE.finditer(html):
        if match.group("var") != var_name:
            continue
        payload = match.group("payload")
        try:
            decoded = codecs.decode(payload, "unicode_escape")
        except (UnicodeDecodeError, ValueError) as exc:
            raise UnderstatError(f"Could not unescape embedded var {var_name!r}: {exc}") from exc
        # unicode_escape produces latin-1 code points; re-encode to recover UTF-8
        # so accented player names survive.
        decoded = decoded.encode("latin-1", errors="backslashreplace").decode(
            "utf-8", errors="replace"
        )
        try:
            return json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise UnderstatError(f"Embedded var {var_name!r} is not valid JSON: {exc}") from exc
    raise UnderstatError(f"No embedded variable named {var_name!r} in page")


@dataclass
class UnderstatClient:
    """Fetches Understat data at roughly one request per second.

    Parameters
    ----------
    min_interval:
        Seconds between requests. Understat is a small site with no published
        rate policy; 1.0s is the conventional courtesy limit.
    """

    min_interval: float = 1.0
    cache_enabled: bool = True
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session = RateLimitedSession(
            min_interval=self.min_interval,
            cache_subdir="understat",
            cache_enabled=self.cache_enabled,
        )

    # ------------------------------------------------------------------
    def _ajax(self, path: str, referer: str) -> Any:
        """GET an XHR endpoint with the headers Understat's own JS sends."""
        headers = dict(_AJAX_HEADERS)
        headers["Referer"] = f"{BASE_URL}/{referer.lstrip('/')}"
        return self.session.get_json(f"{BASE_URL}/{path.lstrip('/')}", headers=headers)

    def _page(self, path: str) -> str:
        """GET a full HTML page (used only for the legacy embedded fallback)."""
        return self.session.get_text(f"{BASE_URL}/{path.lstrip('/')}", suffix=".html")

    def _ajax_or_embedded(self, path: str, referer: str, var_name: str) -> Any:
        """Try the AJAX endpoint, fall back to scraping the page's embedded JSON."""
        try:
            return self._ajax(path, referer)
        except FetchError as ajax_exc:
            log.info("AJAX %s failed (%s); trying embedded JSON on %s", path, ajax_exc, referer)
            try:
                return parse_embedded_json(self._page(referer), var_name)
            except FetchError as page_exc:
                raise UnderstatError(
                    f"Both AJAX ({path}) and embedded ({referer}, var {var_name}) "
                    f"paths failed: {ajax_exc} / {page_exc}"
                ) from page_exc

    # ------------------------------------------------------------------
    # league
    # ------------------------------------------------------------------
    def league_data(self, season: str | int, league: str = "EPL") -> dict[str, Any]:
        """Return the raw league payload with ``teams``, ``players`` and ``dates``."""
        season = str(season)
        data = self._ajax_or_embedded(
            f"getLeagueData/{league}/{season}", f"league/{league}/{season}", "playersData"
        )
        if not isinstance(data, dict) or "players" not in data:
            raise UnderstatError(f"Unexpected league payload for {league} {season}")
        return data

    def league_players(self, season: str | int, league: str = "EPL") -> pd.DataFrame:
        """Return the season-to-date per-player aggregate table.

        Columns include ``goals``, ``xG``, ``npxG``, ``assists``, ``xA``,
        ``shots``, ``key_passes``, ``xGChain``, ``xGBuildup``, ``time``.
        """
        data = self.league_data(season, league)
        frame = pd.DataFrame(data.get("players", []))
        if frame.empty:
            return frame
        frame["season"] = str(season)
        frame["league"] = league
        numeric = [
            "games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes",
            "yellow_cards", "red_cards", "npg", "npxG", "xGChain", "xGBuildup",
        ]
        for col in numeric:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def league_dates(self, season: str | int, league: str = "EPL") -> pd.DataFrame:
        """Return the fixture list with match ids, xG and result flags.

        ``isResult`` marks matches that have been played -- only those have shot
        data. The nested ``h``/``a``/``goals``/``xG`` dicts are flattened.
        """
        data = self.league_data(season, league)
        rows: list[dict[str, Any]] = []
        for match in data.get("dates", []):
            home = match.get("h") or {}
            away = match.get("a") or {}
            goals = match.get("goals") or {}
            xg = match.get("xG") or {}
            rows.append(
                {
                    "season": str(season),
                    "league": league,
                    "match_id": match.get("id"),
                    "is_result": bool(match.get("isResult")),
                    "datetime": match.get("datetime"),
                    "h_team_id": home.get("id"),
                    "h_team": home.get("title"),
                    "a_team_id": away.get("id"),
                    "a_team": away.get("title"),
                    "h_goals": goals.get("h"),
                    "a_goals": goals.get("a"),
                    "h_xg": xg.get("h"),
                    "a_xg": xg.get("a"),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        for col in ("h_goals", "a_goals", "h_xg", "a_xg"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def match_ids(self, season: str | int, league: str = "EPL", played_only: bool = True) -> list[str]:
        """Return the Understat match ids for a season."""
        dates = self.league_dates(season, league)
        if dates.empty:
            return []
        if played_only:
            dates = dates[dates["is_result"]]
        return [str(x) for x in dates["match_id"].tolist()]

    # ------------------------------------------------------------------
    # match / shots
    # ------------------------------------------------------------------
    def match_data(self, match_id: str | int) -> dict[str, Any]:
        """Return the raw match payload with ``shots`` and ``rosters``."""
        match_id = str(match_id)
        data = self._ajax_or_embedded(
            f"getMatchData/{match_id}", f"match/{match_id}", "shotsData"
        )
        # The embedded fallback yields shotsData directly (a {h, a} dict), while
        # the AJAX endpoint wraps it. Normalise both onto the wrapped shape.
        if isinstance(data, dict) and "shots" not in data and {"h", "a"} <= set(data):
            return {"shots": data, "rosters": {}}
        if not isinstance(data, dict) or "shots" not in data:
            raise UnderstatError(f"Unexpected match payload for {match_id}")
        return data

    def match_shots(self, match_id: str | int) -> pd.DataFrame:
        """Return shot-level xG for one match.

        One row per shot with pitch coordinates (``x``/``y`` in 0-1 units where
        x=1 is the attacking goal line), ``xg``, ``situation`` (OpenPlay,
        FromCorner, SetPiece, DirectFreekick, Penalty), ``shot_type``
        (RightFoot / LeftFoot / Head / OtherBodyPart), ``result``, the assisting
        player and the preceding action.
        """
        data = self.match_data(match_id)
        return shots_frame(data.get("shots") or {}, match_id=str(match_id))

    def season_shots(
        self,
        season: str | int,
        league: str = "EPL",
        match_ids: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch every played match in a season and concatenate the shots.

        A single failing match is recorded in ``self.errors`` and skipped so one
        bad id cannot lose the whole season.
        """
        ids = list(match_ids) if match_ids is not None else self.match_ids(season, league)
        if limit is not None:
            ids = ids[:limit]
        frames: list[pd.DataFrame] = []
        for match_id in ids:
            try:
                frame = self.match_shots(match_id)
            except FetchError as exc:
                log.warning("match %s shots failed: %s", match_id, exc)
                self.errors.append(f"match {match_id}: {exc}")
                continue
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=list(SHOT_COLUMNS))
        out = pd.concat(frames, ignore_index=True)
        out["season"] = out["season"].fillna(str(season))
        return out

    # ------------------------------------------------------------------
    # player / team
    # ------------------------------------------------------------------
    def player_data(self, pid: str | int) -> dict[str, Any]:
        """Return the raw player payload with ``matches``, ``shots`` and ``groups``."""
        pid = str(pid)
        data = self._ajax_or_embedded(f"getPlayerData/{pid}", f"player/{pid}", "matchesData")
        if isinstance(data, list):  # legacy embedded matchesData is a bare list
            return {"matches": data, "shots": []}
        if not isinstance(data, dict):
            raise UnderstatError(f"Unexpected player payload for {pid}")
        return data

    def player_matches(self, pid: str | int) -> pd.DataFrame:
        """Return one row per match played by a player, across all seasons held."""
        data = self.player_data(pid)
        frame = pd.DataFrame(data.get("matches", []))
        if frame.empty:
            return frame
        frame["player_id"] = str(pid)
        for col in (
            "goals", "shots", "xG", "time", "h_goals", "a_goals", "xA",
            "assists", "key_passes", "npg", "npxG", "xGChain", "xGBuildup",
        ):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        return frame

    def player_shots(self, pid: str | int) -> pd.DataFrame:
        """Return every shot Understat holds for one player."""
        data = self.player_data(pid)
        return shots_frame({"h": data.get("shots") or []})

    def team_data(self, tid: str | int, season: str | int) -> dict[str, Any]:
        """Return the raw team payload with ``dates``, ``players`` and ``statistics``.

        ``tid`` is the team *title* as Understat spells it (``"Manchester City"``,
        with spaces or underscores), not a numeric id -- the endpoint is keyed on
        the title even though the league payload also exposes numeric ids.
        """
        title = str(tid).replace(" ", "_")
        season = str(season)
        data = self._ajax_or_embedded(
            f"getTeamData/{title}/{season}", f"team/{title}/{season}", "datesData"
        )
        if isinstance(data, list):
            return {"dates": data, "players": [], "statistics": {}}
        if not isinstance(data, dict):
            raise UnderstatError(f"Unexpected team payload for {tid} {season}")
        return data

    def team_matches(self, tid: str | int, season: str | int) -> pd.DataFrame:
        """Return one row per fixture for a team in a season, with per-match xG."""
        data = self.team_data(tid, season)
        rows: list[dict[str, Any]] = []
        for match in data.get("dates", []):
            home = match.get("h") or {}
            away = match.get("a") or {}
            goals = match.get("goals") or {}
            xg = match.get("xG") or {}
            rows.append(
                {
                    "team": str(tid),
                    "season": str(season),
                    "match_id": match.get("id"),
                    "is_result": bool(match.get("isResult")),
                    "side": match.get("side"),
                    "result": match.get("result"),
                    "datetime": match.get("datetime"),
                    "h_team": home.get("title"),
                    "a_team": away.get("title"),
                    "h_goals": goals.get("h"),
                    "a_goals": goals.get("a"),
                    "h_xg": xg.get("h"),
                    "a_xg": xg.get("a"),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        for col in ("h_goals", "a_goals", "h_xg", "a_xg"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self.session.close()


# ----------------------------------------------------------------------
# pure shaping (no network)
# ----------------------------------------------------------------------
def shots_frame(shots: dict[str, list[dict[str, Any]]], match_id: str | None = None) -> pd.DataFrame:
    """Normalise an Understat ``{"h": [...], "a": [...]}`` shot block.

    Accepts either side key being absent. Returns an empty frame with the full
    :data:`SHOT_COLUMNS` schema when there are no shots, so callers can concat
    without special-casing.
    """
    rows: list[dict[str, Any]] = []
    for side in ("h", "a"):
        for shot in shots.get(side) or []:
            team = shot.get("h_team") if shot.get("h_a") == "h" else shot.get("a_team")
            rows.append(
                {
                    "season": shot.get("season"),
                    "match_id": shot.get("match_id", match_id),
                    "shot_id": shot.get("id"),
                    "date": shot.get("date"),
                    "minute": shot.get("minute"),
                    "player_id": shot.get("player_id"),
                    "player": shot.get("player"),
                    "team": team,
                    "h_a": shot.get("h_a", side),
                    "h_team": shot.get("h_team"),
                    "a_team": shot.get("a_team"),
                    "h_goals": shot.get("h_goals"),
                    "a_goals": shot.get("a_goals"),
                    "result": shot.get("result"),
                    "is_goal": shot.get("result") == "Goal",
                    "x": shot.get("X"),
                    "y": shot.get("Y"),
                    "xg": shot.get("xG"),
                    "situation": shot.get("situation"),
                    "shot_type": shot.get("shotType"),
                    "last_action": shot.get("lastAction"),
                    "player_assisted": shot.get("player_assisted"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(SHOT_COLUMNS))
    frame = pd.DataFrame(rows)
    for col in ("minute", "h_goals", "a_goals", "x", "y", "xg"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["match_id"] = frame["match_id"].astype("string")
    frame["shot_id"] = frame["shot_id"].astype("string")
    frame["player_id"] = frame["player_id"].astype("string")
    return frame[list(SHOT_COLUMNS)]


# ----------------------------------------------------------------------
# module-level convenience
# ----------------------------------------------------------------------
def league_players(season: str | int, league: str = "EPL") -> pd.DataFrame:
    """Fetch the per-player season aggregates for a league season."""
    client = UnderstatClient()
    try:
        return client.league_players(season, league)
    finally:
        client.close()


def league_dates(season: str | int, league: str = "EPL") -> pd.DataFrame:
    """Fetch the fixture list (with match ids) for a league season."""
    client = UnderstatClient()
    try:
        return client.league_dates(season, league)
    finally:
        client.close()


def player_matches(pid: str | int) -> pd.DataFrame:
    """Fetch a player's per-match log."""
    client = UnderstatClient()
    try:
        return client.player_matches(pid)
    finally:
        client.close()


def team_matches(tid: str | int, season: str | int) -> pd.DataFrame:
    """Fetch a team's per-match log for one season."""
    client = UnderstatClient()
    try:
        return client.team_matches(tid, season)
    finally:
        client.close()


def match_shots(match_id: str | int) -> pd.DataFrame:
    """Fetch shot-level xG for one match."""
    client = UnderstatClient()
    try:
        return client.match_shots(match_id)
    finally:
        client.close()


def season_shots(season: str | int, league: str = "EPL", limit: int | None = None) -> pd.DataFrame:
    """Fetch every played match's shots for a season."""
    client = UnderstatClient()
    try:
        return client.season_shots(season, league, limit=limit)
    finally:
        client.close()


def refresh(
    seasons: Iterable[str | int], league: str = "EPL", shot_limit: int | None = None
) -> dict[str, pd.DataFrame]:
    """Refresh league players, fixtures and shots for each season and persist them.

    Returns a mapping of dataset name to frame. Seasons that fail are logged and
    omitted rather than aborting the sweep.
    """
    client = UnderstatClient()
    out: dict[str, pd.DataFrame] = {}
    try:
        for season in seasons:
            season = str(season)
            try:
                players = client.league_players(season, league)
                dates = client.league_dates(season, league)
            except FetchError as exc:
                log.error("understat season %s failed: %s", season, exc)
                client.errors.append(f"season {season}: {exc}")
                continue
            store.save(players, f"understat/players_{season}")
            store.save(dates, f"understat/fixtures_{season}")
            out[f"understat/players_{season}"] = players
            out[f"understat/fixtures_{season}"] = dates

            ids = dates.loc[dates["is_result"], "match_id"].astype(str).tolist() if not dates.empty else []
            shots = client.season_shots(season, league, match_ids=ids, limit=shot_limit)
            store.save(shots, f"understat/shots_{season}")
            out[f"understat/shots_{season}"] = shots
            log.info("understat %s: %d players, %d shots", season, len(players), len(shots))
        return out
    finally:
        client.close()
