"""Client for the official Fantasy Premier League JSON API.

No authentication is required for any endpoint exposed here. All calls go
through :class:`~fplopt.data.http.RateLimitedSession`, which enforces a minimum
gap between requests, retries 429/5xx with exponential backoff, and caches
response bodies on disk under ``data/cache/fpl/<date>/``.

Endpoint notes discovered against the live 2026/27 game:

* ``/bootstrap-static/`` also carries ``game_config`` (scoring table, rules)
  alongside the documented ``game_settings``.
* ``/event/{gw}/live/`` is the ground truth for points verification: each
  element carries a ``stats`` block and an ``explain`` breakdown listing the
  points contributed per identifier per fixture.
* ``/entry/{id}/transfers/`` returns a bare JSON list, not an object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd

from .http import FetchError, HTTPStatusError, RateLimitedSession

__all__ = [
    "BASE_URL",
    "FPLClient",
    "FetchError",
    "HTTPStatusError",
    "bootstrap",
    "fixtures",
    "element_summary",
    "entry",
    "entry_history",
    "entry_picks",
    "entry_transfers",
    "league_standings",
    "live",
    "current_event",
    "finished_events",
    "elements_frame",
    "teams_frame",
    "events_frame",
    "fixtures_frame",
    "live_points_frame",
    "live_explain_frame",
]

log = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"

# The FPL edge is fronted by a CDN that answers bot-looking agents with 403,
# so a real browser UA plus a matching Accept header is not optional.
_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://fantasy.premierleague.com/",
    "Origin": "https://fantasy.premierleague.com",
}


@dataclass
class FPLClient:
    """Thin typed wrapper over the FPL REST endpoints.

    Parameters
    ----------
    min_interval:
        Seconds to wait between calls. The FPL API is generous but unpublished;
        0.4s keeps a full-season sweep well inside anything that has ever
        triggered throttling.
    cache_enabled:
        When ``False``, every call hits the network.
    """

    min_interval: float = 0.4
    cache_enabled: bool = True
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.session = RateLimitedSession(
            min_interval=self.min_interval,
            cache_subdir="fpl",
            cache_enabled=self.cache_enabled,
            timeout=self.timeout,
            extra_headers=_HEADERS,
        )

    # ------------------------------------------------------------------
    def _get(self, path: str, use_cache: bool = True) -> Any:
        """GET ``BASE_URL/path`` and return parsed JSON."""
        return self.session.get_json(f"{BASE_URL}/{path.lstrip('/')}", use_cache=use_cache)

    # ------------------------------------------------------------------
    # static game data
    # ------------------------------------------------------------------
    def bootstrap(self) -> dict[str, Any]:
        """Return ``/bootstrap-static/``.

        Keys include ``elements`` (players), ``teams``, ``events`` (gameweeks),
        ``element_types`` (positions), ``chips``, ``element_stats``,
        ``game_settings`` and ``game_config``.
        """
        data = self._get("bootstrap-static/")
        if not isinstance(data, dict) or "elements" not in data:
            raise FetchError("bootstrap-static returned an unexpected payload shape")
        return data

    def fixtures(self, event: int | None = None) -> list[dict[str, Any]]:
        """Return ``/fixtures/``, optionally filtered to a single gameweek.

        Fixtures with ``event is None`` are unscheduled (postponed or awaiting a
        rearranged date) and are retained by the unfiltered call.
        """
        path = "fixtures/" if event is None else f"fixtures/?event={int(event)}"
        data = self._get(path)
        if not isinstance(data, list):
            raise FetchError("fixtures returned an unexpected payload shape")
        return data

    def element_summary(self, pid: int) -> dict[str, Any]:
        """Return ``/element-summary/{pid}/`` for one player.

        Keys: ``fixtures`` (upcoming), ``history`` (this season, per gameweek)
        and ``history_past`` (one row per prior season).
        """
        data = self._get(f"element-summary/{int(pid)}/")
        if not isinstance(data, dict) or "history" not in data:
            raise FetchError(f"element-summary/{pid} returned an unexpected payload shape")
        return data

    def live(self, gw: int) -> dict[str, Any]:
        """Return ``/event/{gw}/live/``.

        This is the authoritative per-player scoring feed. Each entry in
        ``elements`` has ``stats`` (final counting stats plus ``total_points``)
        and ``explain`` (a per-fixture list of ``{identifier, value, points,
        points_modification}`` rows that sum to ``total_points``).

        The result is not read from the disk cache while the gameweek is still
        in flight -- callers should pass a finished gameweek for stable data.
        """
        data = self._get(f"event/{int(gw)}/live/")
        if not isinstance(data, dict) or "elements" not in data:
            raise FetchError(f"event/{gw}/live returned an unexpected payload shape")
        return data

    # ------------------------------------------------------------------
    # manager (entry) data
    # ------------------------------------------------------------------
    def entry(self, eid: int) -> dict[str, Any]:
        """Return ``/entry/{eid}/`` -- one manager's profile and league memberships."""
        return self._get(f"entry/{int(eid)}/")

    def entry_history(self, eid: int) -> dict[str, Any]:
        """Return ``/entry/{eid}/history/`` with ``current``, ``past`` and ``chips``."""
        return self._get(f"entry/{int(eid)}/history/")

    def entry_picks(self, eid: int, gw: int) -> dict[str, Any]:
        """Return ``/entry/{eid}/event/{gw}/picks/``.

        Contains ``picks`` (15 rows with ``element``, ``position``,
        ``multiplier``, captain flags), ``active_chip``, ``automatic_subs`` and
        ``entry_history``. Raises :class:`HTTPStatusError` with status 404 when
        the manager did not enter that gameweek.
        """
        return self._get(f"entry/{int(eid)}/event/{int(gw)}/picks/")

    def entry_transfers(self, eid: int) -> list[dict[str, Any]]:
        """Return ``/entry/{eid}/transfers/`` -- a flat list, newest first.

        A manager who has never transferred returns an empty list, not a 404.
        """
        data = self._get(f"entry/{int(eid)}/transfers/")
        if not isinstance(data, list):
            raise FetchError(f"entry/{eid}/transfers returned an unexpected payload shape")
        return data

    # ------------------------------------------------------------------
    # leagues
    # ------------------------------------------------------------------
    def league_standings(self, lid: int, page: int = 1) -> dict[str, Any]:
        """Return one page of ``/leagues-classic/{lid}/standings/``.

        Pages hold 50 entries. ``result["standings"]["has_next"]`` tells you
        whether another page exists.
        """
        return self._get(f"leagues-classic/{int(lid)}/standings/?page_standings={int(page)}")

    def league_standings_all(self, lid: int, max_pages: int = 20) -> list[dict[str, Any]]:
        """Page through a classic league and return every standings row.

        ``max_pages`` caps the sweep so a mistaken call against the overall
        league (millions of entries) cannot run forever.
        """
        rows: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            payload = self.league_standings(lid, page)
            standings = payload.get("standings", {})
            batch = standings.get("results", [])
            rows.extend(batch)
            if not standings.get("has_next") or not batch:
                break
        return rows

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self.session.close()


# ----------------------------------------------------------------------
# module-level convenience wrappers over a shared default client
# ----------------------------------------------------------------------
_default_client: FPLClient | None = None


def _client() -> FPLClient:
    """Return the lazily created process-wide client."""
    global _default_client
    if _default_client is None:
        _default_client = FPLClient()
    return _default_client


def bootstrap() -> dict[str, Any]:
    """Fetch ``/bootstrap-static/`` using the shared client."""
    return _client().bootstrap()


def fixtures(event: int | None = None) -> list[dict[str, Any]]:
    """Fetch ``/fixtures/`` using the shared client."""
    return _client().fixtures(event)


def element_summary(pid: int) -> dict[str, Any]:
    """Fetch ``/element-summary/{pid}/`` using the shared client."""
    return _client().element_summary(pid)


def entry(eid: int) -> dict[str, Any]:
    """Fetch ``/entry/{eid}/`` using the shared client."""
    return _client().entry(eid)


def entry_history(eid: int) -> dict[str, Any]:
    """Fetch ``/entry/{eid}/history/`` using the shared client."""
    return _client().entry_history(eid)


def entry_picks(eid: int, gw: int) -> dict[str, Any]:
    """Fetch ``/entry/{eid}/event/{gw}/picks/`` using the shared client."""
    return _client().entry_picks(eid, gw)


def entry_transfers(eid: int) -> list[dict[str, Any]]:
    """Fetch ``/entry/{eid}/transfers/`` using the shared client."""
    return _client().entry_transfers(eid)


def league_standings(lid: int, page: int = 1) -> dict[str, Any]:
    """Fetch one standings page using the shared client."""
    return _client().league_standings(lid, page)


def live(gw: int) -> dict[str, Any]:
    """Fetch ``/event/{gw}/live/`` using the shared client."""
    return _client().live(gw)


# ----------------------------------------------------------------------
# pure helpers over already-fetched payloads (offline testable)
# ----------------------------------------------------------------------
def current_event(boot: dict[str, Any]) -> int | None:
    """Return the gameweek currently in progress, or ``None`` before GW1."""
    for event in boot.get("events", []):
        if event.get("is_current"):
            return int(event["id"])
    return None


def next_event(boot: dict[str, Any]) -> int | None:
    """Return the next gameweek whose deadline has not passed, or ``None``."""
    for event in boot.get("events", []):
        if event.get("is_next"):
            return int(event["id"])
    return None


def finished_events(boot: dict[str, Any], require_data_checked: bool = True) -> list[int]:
    """Return the gameweek ids that are complete.

    ``require_data_checked`` additionally insists FPL has signed off on bonus
    and stat corrections, which is what you want before treating live data as
    ground truth.
    """
    out: list[int] = []
    for event in boot.get("events", []):
        if not event.get("finished"):
            continue
        if require_data_checked and not event.get("data_checked"):
            continue
        out.append(int(event["id"]))
    return out


def elements_frame(boot: dict[str, Any]) -> pd.DataFrame:
    """Return ``bootstrap["elements"]`` as a DataFrame with position and team names joined."""
    frame = pd.DataFrame(boot.get("elements", []))
    if frame.empty:
        return frame
    positions = {int(t["id"]): t["singular_name_short"] for t in boot.get("element_types", [])}
    teams = {int(t["id"]): t["name"] for t in boot.get("teams", [])}
    short = {int(t["id"]): t["short_name"] for t in boot.get("teams", [])}
    frame["position"] = frame["element_type"].map(positions)
    frame["team_name"] = frame["team"].map(teams)
    frame["team_short"] = frame["team"].map(short)
    frame["full_name"] = (
        frame["first_name"].fillna("") + " " + frame["second_name"].fillna("")
    ).str.strip()
    # now_cost is in tenths of a million.
    frame["price"] = pd.to_numeric(frame["now_cost"], errors="coerce") / 10.0
    return frame


def teams_frame(boot: dict[str, Any]) -> pd.DataFrame:
    """Return ``bootstrap["teams"]`` as a DataFrame."""
    return pd.DataFrame(boot.get("teams", []))


def events_frame(boot: dict[str, Any]) -> pd.DataFrame:
    """Return ``bootstrap["events"]`` as a DataFrame with the deadline parsed."""
    frame = pd.DataFrame(boot.get("events", []))
    if not frame.empty and "deadline_time" in frame:
        frame["deadline_time"] = pd.to_datetime(frame["deadline_time"], errors="coerce", utc=True)
    return frame


def fixtures_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the ``/fixtures/`` payload as a DataFrame with kickoff parsed.

    The nested ``stats`` column is dropped; use :func:`fixture_stats_frame` for
    the long-form version.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "kickoff_time" in frame:
        frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], errors="coerce", utc=True)
    return frame.drop(columns=[c for c in ("stats",) if c in frame.columns])


def fixture_stats_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the per-fixture ``stats`` blocks into a long DataFrame.

    Columns: ``fixture``, ``identifier``, ``side`` (``h``/``a``), ``element``, ``value``.
    """
    out: list[dict[str, Any]] = []
    for fixture in rows:
        for stat in fixture.get("stats") or []:
            identifier = stat.get("identifier")
            for side in ("h", "a"):
                for item in stat.get(side) or []:
                    out.append(
                        {
                            "fixture": fixture.get("id"),
                            "event": fixture.get("event"),
                            "identifier": identifier,
                            "side": side,
                            "element": item.get("element"),
                            "value": item.get("value"),
                        }
                    )
    return pd.DataFrame(out)


def live_points_frame(payload: dict[str, Any], gw: int) -> pd.DataFrame:
    """Flatten ``/event/{gw}/live/`` stats into one row per player."""
    rows: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        row: dict[str, Any] = {"gw": gw, "element": element.get("id")}
        stats = element.get("stats") or {}
        row.update(stats)
        rows.append(row)
    frame = pd.DataFrame(rows)
    # The API sends the ICT family and xG family as strings.
    for col in (
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def live_explain_frame(payload: dict[str, Any], gw: int) -> pd.DataFrame:
    """Flatten the ``explain`` breakdown into one row per scoring component.

    Columns: ``gw``, ``element``, ``fixture``, ``identifier``, ``value``,
    ``points``, ``points_modification``. Summing ``points`` plus
    ``points_modification`` per element reproduces ``total_points``, which is
    exactly the check to run when verifying a scoring model.
    """
    rows: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        eid = element.get("id")
        for block in element.get("explain") or []:
            fixture = block.get("fixture")
            for stat in block.get("stats") or []:
                rows.append(
                    {
                        "gw": gw,
                        "element": eid,
                        "fixture": fixture,
                        "identifier": stat.get("identifier"),
                        "value": stat.get("value"),
                        "points": stat.get("points"),
                        "points_modification": stat.get("points_modification", 0),
                    }
                )
    return pd.DataFrame(rows)


def iter_element_summaries(
    pids: list[int], client: FPLClient | None = None
) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    """Yield ``(pid, payload, error)`` for each player id.

    Never raises: a failed player yields ``(pid, None, message)`` so a sweep of
    600+ players is not aborted by one bad id.
    """
    cli = client or _client()
    for pid in pids:
        try:
            yield pid, cli.element_summary(pid), None
        except FetchError as exc:
            log.warning("element-summary %s failed: %s", pid, exc)
            yield pid, None, str(exc)
