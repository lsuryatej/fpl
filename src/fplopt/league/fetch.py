"""Fetch + disk-cache every FPL API object needed for mini-league intelligence.

Deliberately self-contained: this module owns its own tiny HTTP client so the
league package never has to wait on ``fplopt.data``.

Cache layout (under ``<repo>/data/cache/league/``)::

    bootstrap.json
    live/gw{N}.json
    league/{league_id}/standings.json
    entry/{eid}/history.json
    entry/{eid}/transfers.json
    entry/{eid}/picks_gw{N}.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests

API_BASE = "https://fantasy.premierleague.com/api"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Roughly 2.5 requests/second.
MIN_REQUEST_INTERVAL = 0.40

DEFAULT_LEAGUE_ID = 490294
DEFAULT_ENTRY_ID = 3539707

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

CHIP_NAMES = ("wildcard", "bboost", "3xc", "freehit", "manager")


def repo_root() -> Path:
    """Walk up from this file until we find the project root (pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def cache_root() -> Path:
    return repo_root() / "data" / "cache" / "league"


class FplClient:
    """Minimal rate-limited, disk-cached FPL API client."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        min_interval: float = MIN_REQUEST_INTERVAL,
        offline: bool = False,
        verbose: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else cache_root()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.offline = offline
        self.verbose = verbose
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Language": "en-GB,en;q=0.9",
            }
        )
        self.n_network_calls = 0
        self.n_cache_hits = 0

    # ---------------------------------------------------------------- internals

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _cache_path(self, rel: str) -> Path:
        return self.cache_dir / rel

    def get(self, path: str, cache_rel: str, max_age: float | None = None) -> Any:
        """GET ``path``, serving from ``cache_rel`` when it exists and is fresh."""
        cpath = self._cache_path(cache_rel)
        if cpath.exists():
            fresh = max_age is None or (time.time() - cpath.stat().st_mtime) < max_age
            if fresh or self.offline:
                try:
                    self.n_cache_hits += 1
                    return json.loads(cpath.read_text())
                except json.JSONDecodeError:
                    pass  # fall through and refetch

        if self.offline:
            raise FileNotFoundError(f"offline mode and no cache for {cache_rel}")

        url = f"{API_BASE}{path}"
        last_err: Exception | None = None
        for attempt in range(4):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=30)
            except requests.RequestException as exc:  # pragma: no cover - network
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            self.n_network_calls += 1
            if resp.status_code == 200:
                data = resp.json()
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps(data))
                return data
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                last_err = RuntimeError(f"HTTP {resp.status_code} for {url}")
                continue
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")
        raise RuntimeError(f"failed to fetch {url}: {last_err}")

    # ------------------------------------------------------------------ getters

    def bootstrap(self, max_age: float = 6 * 3600) -> dict:
        return self.get("/bootstrap-static/", "bootstrap.json", max_age=max_age)

    def live(self, gw: int, finished: bool = True) -> dict:
        # Finished gameweeks never change; unfinished ones get a short TTL.
        max_age = None if finished else 900
        return self.get(f"/event/{gw}/live/", f"live/gw{gw}.json", max_age=max_age)

    def standings(self, league_id: int, max_age: float = 1800) -> dict:
        """Fetch every standings page and merge into one payload."""
        pages: list[dict] = []
        page = 1
        merged: dict | None = None
        while True:
            data = self.get(
                f"/leagues-classic/{league_id}/standings/?page_standings={page}",
                f"league/{league_id}/standings_p{page}.json",
                max_age=max_age,
            )
            pages.append(data)
            if merged is None:
                merged = json.loads(json.dumps(data))
            else:
                merged["standings"]["results"].extend(data["standings"]["results"])
            if not data["standings"].get("has_next"):
                break
            page += 1
            if page > 50:  # safety valve
                break
        assert merged is not None
        merged["standings"]["has_next"] = False
        merged["standings"]["n_pages"] = len(pages)
        return merged

    def fixtures(self, gw: int, finished: bool = True) -> list:
        max_age = None if finished else 900
        return self.get(f"/fixtures/?event={gw}", f"fixtures/gw{gw}.json", max_age=max_age)

    def entry_history(self, eid: int, max_age: float = 1800) -> dict:
        return self.get(f"/entry/{eid}/history/", f"entry/{eid}/history.json", max_age=max_age)

    def entry_transfers(self, eid: int, max_age: float = 1800) -> list:
        return self.get(f"/entry/{eid}/transfers/", f"entry/{eid}/transfers.json", max_age=max_age)

    def entry_picks(self, eid: int, gw: int, finished: bool = True) -> dict:
        max_age = None if finished else 900
        return self.get(
            f"/entry/{eid}/event/{gw}/picks/",
            f"entry/{eid}/picks_gw{gw}.json",
            max_age=max_age,
        )


# --------------------------------------------------------------------- payload


@dataclass
class Player:
    id: int
    web_name: str
    full_name: str
    team: int
    team_short: str
    position: str
    now_cost: int
    total_points: int
    global_owned_pct: float
    status: str

    @property
    def price(self) -> float:
        return self.now_cost / 10.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.web_name} {self.team_short} {self.position} {self.price}>"


@dataclass
class LeagueData:
    """Everything the analytics modules need, already in memory."""

    league_id: int
    league_name: str
    user_entry: int
    gameweeks: list[int]
    standings: list[dict]  # ordered by rank
    picks: dict[int, dict[int, dict]] = field(default_factory=dict)  # eid -> gw -> payload
    histories: dict[int, dict] = field(default_factory=dict)
    transfers: dict[int, list] = field(default_factory=dict)
    players: dict[int, Player] = field(default_factory=dict)
    live: dict[int, dict[int, dict]] = field(default_factory=dict)  # gw -> eid -> stats
    events: list[dict] = field(default_factory=list)
    fixtures: dict[int, list] = field(default_factory=dict)  # gw -> fixture rows
    next_gw: int | None = None
    n_pages: int = 1

    # ------------------------------------------------------------------ helpers

    @property
    def entry_ids(self) -> list[int]:
        return [row["entry"] for row in self.standings]

    @property
    def n_managers(self) -> int:
        return len(self.standings)

    def name_of(self, eid: int) -> str:
        for row in self.standings:
            if row["entry"] == eid:
                return row["entry_name"]
        return str(eid)

    def manager_of(self, eid: int) -> str:
        for row in self.standings:
            if row["entry"] == eid:
                return row["player_name"]
        return str(eid)

    def rank_of(self, eid: int) -> int:
        for row in self.standings:
            if row["entry"] == eid:
                return row["rank"]
        return self.n_managers

    def total_of(self, eid: int) -> int:
        for row in self.standings:
            if row["entry"] == eid:
                return row["total"]
        return 0

    def is_gw_final(self, gw: int) -> bool:
        """True once every fixture in the gameweek has been played and checked."""
        fx = self.fixtures.get(gw)
        if fx is not None:
            return all(f.get("finished") for f in fx)
        for ev in self.events:
            if ev["id"] == gw:
                return bool(ev.get("finished") or ev.get("data_checked"))
        return False

    def pending_fixtures(self, gw: int) -> list[dict]:
        return [
            f
            for f in self.fixtures.get(gw, [])
            if not (f.get("finished") or f.get("finished_provisional"))
        ]

    def pending_teams(self, gw: int) -> set[int]:
        """Teams whose gameweek fixture has not been played yet."""
        out: set[int] = set()
        for f in self.fixtures.get(gw, []):
            if not (f.get("finished") or f.get("finished_provisional")):
                out.add(f["team_h"])
                out.add(f["team_a"])
        return out

    def pname(self, element: int) -> str:
        p = self.players.get(element)
        return p.web_name if p else f"#{element}"

    def player_points(self, element: int, gw: int) -> int:
        stats = self.live.get(gw, {}).get(element)
        return int(stats["total_points"]) if stats else 0

    def player_minutes(self, element: int, gw: int) -> int:
        stats = self.live.get(gw, {}).get(element)
        return int(stats["minutes"]) if stats else 0

    def season_points(self, element: int) -> int:
        return sum(self.player_points(element, gw) for gw in self.gameweeks)


def _finished_gameweeks(events: list[dict]) -> list[int]:
    """Gameweeks that have actually been played (scores settled or in-flight)."""
    out = []
    for ev in events:
        if ev.get("finished") or ev.get("data_checked") or ev.get("is_current"):
            out.append(ev["id"])
    return sorted(set(out))


def load_league(
    league_id: int = DEFAULT_LEAGUE_ID,
    user_entry: int = DEFAULT_ENTRY_ID,
    gameweeks: Iterable[int] | None = None,
    client: FplClient | None = None,
    verbose: bool = True,
) -> LeagueData:
    """Fetch (or load from cache) the full league picture."""
    cl = client or FplClient(verbose=verbose)

    boot = cl.bootstrap()
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    players: dict[int, Player] = {}
    for e in boot["elements"]:
        players[e["id"]] = Player(
            id=e["id"],
            web_name=e["web_name"],
            full_name=f"{e['first_name']} {e['second_name']}",
            team=e["team"],
            team_short=teams.get(e["team"], "???"),
            position=POSITIONS.get(e["element_type"], "?"),
            now_cost=e["now_cost"],
            total_points=e["total_points"],
            global_owned_pct=float(e["selected_by_percent"]),
            status=e.get("status", "a"),
        )

    events = boot["events"]
    finished_map = {ev["id"]: bool(ev.get("finished")) for ev in events}
    gws = list(gameweeks) if gameweeks is not None else _finished_gameweeks(events)
    next_gw = next((ev["id"] for ev in events if ev.get("is_next")), None)

    if verbose:
        print(f"[fetch] gameweeks played: {gws}  next deadline: GW{next_gw}")

    live: dict[int, dict[int, dict]] = {}
    fixtures: dict[int, list] = {}
    for gw in gws:
        payload = cl.live(gw, finished=finished_map.get(gw, False))
        live[gw] = {row["id"]: row["stats"] for row in payload["elements"]}
        fixtures[gw] = cl.fixtures(gw, finished=finished_map.get(gw, False))

    st = cl.standings(league_id)
    rows = sorted(st["standings"]["results"], key=lambda r: r["rank"])
    if verbose:
        print(
            f"[fetch] {st['league']['name']}: {len(rows)} entries "
            f"across {st['standings'].get('n_pages', 1)} standings page(s)"
        )

    data = LeagueData(
        league_id=league_id,
        league_name=st["league"]["name"],
        user_entry=user_entry,
        gameweeks=gws,
        standings=rows,
        players=players,
        live=live,
        events=events,
        fixtures=fixtures,
        next_gw=next_gw,
        n_pages=st["standings"].get("n_pages", 1),
    )

    total = len(rows)
    for i, row in enumerate(rows, 1):
        eid = row["entry"]
        data.histories[eid] = cl.entry_history(eid)
        data.transfers[eid] = cl.entry_transfers(eid)
        data.picks[eid] = {}
        for gw in gws:
            data.picks[eid][gw] = cl.entry_picks(eid, gw, finished=finished_map.get(gw, False))
        if verbose and (i % 10 == 0 or i == total):
            print(f"[fetch]   entries {i}/{total}")

    if verbose:
        print(
            f"[fetch] done: {cl.n_network_calls} network calls, "
            f"{cl.n_cache_hits} cache hits"
        )
    return data


# ------------------------------------------------------------- multiplier logic


def effective_multipliers(data: LeagueData, eid: int, gw: int) -> dict[int, int]:
    """Post-settlement multiplier per element for one manager-gameweek.

    The picks endpoint returns the multipliers as they stood at the deadline.
    Two things change afterwards and neither is reflected there:

    1. Automatic substitutions (reported in ``automatic_subs``).
    2. The captaincy falling through to the vice-captain when the captain
       records zero minutes. FPL does not report this anywhere.
    """
    payload = data.picks.get(eid, {}).get(gw)
    if not payload:
        return {}

    picks = payload["picks"]
    mult = {p["element"]: int(p["multiplier"]) for p in picks}

    # FPL only settles auto-subs and the vice-captain fallback once every
    # fixture in the gameweek has finished. Before then, a zero-minute player
    # simply has not kicked off yet, so the deadline multipliers stand.
    if not data.is_gw_final(gw):
        return mult

    for sub in payload.get("automatic_subs") or []:
        out_el, in_el = sub["element_out"], sub["element_in"]
        if out_el in mult:
            mult[out_el] = 0
        if in_el in mult:
            mult[in_el] = 1

    captain = next((p["element"] for p in picks if p["is_captain"]), None)
    vice = next((p["element"] for p in picks if p["is_vice_captain"]), None)
    cap_mult = next(
        (int(p["multiplier"]) for p in picks if p["is_captain"]), 2
    )
    if captain is not None and data.player_minutes(captain, gw) == 0:
        if vice is not None and data.player_minutes(vice, gw) > 0:
            # Captain scored nothing (0 minutes); armband falls to the vice.
            if mult.get(captain, 0) > 1:
                mult[captain] = 1
            mult[vice] = cap_mult
    return mult


def gw_score(data: LeagueData, eid: int, gw: int) -> int:
    """Recompute a manager's gameweek score from picks + live data."""
    mult = effective_multipliers(data, eid, gw)
    return sum(m * data.player_points(el, gw) for el, m in mult.items())


def recomputed_total(data: LeagueData, eid: int) -> int:
    """Season total recomputed from raw picks, minus transfer hits."""
    hist = data.histories.get(eid, {}).get("current", [])
    hits = sum(h.get("event_transfers_cost", 0) for h in hist if h["event"] in data.gameweeks)
    return sum(gw_score(data, eid, gw) for gw in data.gameweeks) - hits


def verify_totals(data: LeagueData) -> dict:
    """Cross-check recomputed totals against the standings endpoint."""
    mismatches = []
    for row in data.standings:
        eid = row["entry"]
        calc = recomputed_total(data, eid)
        if calc != row["total"]:
            mismatches.append(
                {
                    "entry": eid,
                    "name": row["entry_name"],
                    "api": row["total"],
                    "calc": calc,
                    "delta": calc - row["total"],
                }
            )
    n = len(data.standings)
    return {
        "n_entries": n,
        "n_match": n - len(mismatches),
        "match_rate": (n - len(mismatches)) / n if n else 0.0,
        "mismatches": mismatches,
    }
