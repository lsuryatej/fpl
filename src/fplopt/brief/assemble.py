"""Gather every input the pre-deadline brief needs into one object.

Panels that depend on work not yet finished (projections, the optimizer)
degrade to ``None`` rather than raising, so the brief is useful from the first
run and gains detail as the pipeline fills in.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fplopt.data.paths import data_dir
from fplopt.league import fetch as lfetch
from fplopt.league import gap as lgap
from fplopt.league import ownership as lown
from fplopt.news import risk as nrisk
from fplopt.prices import track as ptrack

ENTRY_ID = 3539707
LEAGUE_ID = 490294
TOTAL_GWS = 38


@dataclass
class Brief:
    """Everything the renderer needs. Optional fields are pending pipeline work."""

    generated_at: dt.datetime
    target_gw: int
    deadline: dt.datetime | None
    remaining_gws: int
    manager: dict[str, Any]
    standing: dict[str, Any]
    squad: list[dict[str, Any]]
    fixtures: list[dict[str, Any]]
    risks: list[nrisk.Risk]
    open_questions: list[str]
    liabilities: list[Any]
    differentials: list[Any]
    ownership: list[Any]
    gap_targets: list[Any]
    overlap: list[Any]
    price_movers: pd.DataFrame
    captaincy: list[Any] = field(default_factory=list)
    projections: pd.DataFrame | None = None
    recommendation: dict[str, Any] | None = None
    chip_plan: list[dict[str, Any]] | None = None
    model_health: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def pending(self) -> list[str]:
        """Which panels are still waiting on upstream work."""
        missing = []
        if self.projections is None:
            missing.append("projections")
        if self.recommendation is None:
            missing.append("recommendation")
        if self.chip_plan is None:
            missing.append("chip plan")
        if self.model_health is None:
            missing.append("model health")
        return missing


def _latest_parquet(folder: str) -> pd.DataFrame | None:
    directory = data_dir() / folder
    if not directory.exists():
        return None
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None
    try:
        return pd.read_parquet(max(files, key=lambda p: p.stat().st_mtime))
    except (OSError, ValueError):
        return None


def _latest_json(folder: str, name: str = "latest.json") -> Any | None:
    path = data_dir() / folder / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _parse_deadline(events: list[dict], gw: int) -> dt.datetime | None:
    for event in events:
        if event["id"] == gw:
            return dt.datetime.fromisoformat(
                event["deadline_time"].replace("Z", "+00:00")
            )
    return None


def _squad_rows(
    data: lfetch.LeagueData,
    boot_elements: dict[int, dict],
    teams: dict[int, str],
    gw: int,
) -> list[dict[str, Any]]:
    picks = data.picks.get(data.user_entry, {}).get(gw, {}).get("picks", [])
    rows = []
    for pick in picks:
        element = boot_elements.get(pick["element"])
        if element is None:
            continue
        rows.append(
            {
                "player_id": element["id"],
                "name": element["web_name"],
                "team": teams.get(element["team"], "???"),
                "position": element["element_type"],
                "price": element["now_cost"] / 10,
                "points": element["total_points"],
                "minutes": element["minutes"],
                "status": element.get("status", "a"),
                "news": (element.get("news") or "").strip(),
                "selected_by": float(element.get("selected_by_percent") or 0),
                "slot": pick["position"],
                "is_captain": pick.get("is_captain", False),
                "is_vice": pick.get("is_vice_captain", False),
                "benched": pick["position"] > 11,
            }
        )
    return sorted(rows, key=lambda r: r["slot"])


def _fixture_rows(
    fixtures: list[dict], teams: dict[int, str]
) -> list[dict[str, Any]]:
    rows = []
    for fixture in fixtures:
        rows.append(
            {
                "kickoff": (fixture.get("kickoff_time") or "")[:16].replace("T", " "),
                "home": teams.get(fixture["team_h"], "???"),
                "away": teams.get(fixture["team_a"], "???"),
                "home_fdr": fixture.get("team_h_difficulty"),
                "away_fdr": fixture.get("team_a_difficulty"),
            }
        )
    return sorted(rows, key=lambda r: r["kickoff"])


def assemble(
    entry_id: int = ENTRY_ID,
    league_id: int = LEAGUE_ID,
    *,
    verbose: bool = False,
) -> Brief:
    """Build the brief from every source currently available."""
    data = lfetch.load_league(league_id=league_id, user_entry=entry_id, verbose=verbose)

    client = lfetch.FplClient(verbose=verbose)
    boot = client.bootstrap()
    elements = {e["id"]: e for e in boot["elements"]}
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}

    played_gw = max(data.gameweeks)
    target_gw = data.next_gw or played_gw + 1
    deadline = _parse_deadline(boot["events"], target_gw)
    remaining = max(0, TOTAL_GWS - target_gw + 1)

    # The standings row already carries the manager and team name, so there is
    # no reason to spend another request on the entry endpoint.
    standings_row = next((r for r in data.standings if r["entry"] == entry_id), {})

    squad = _squad_rows(data, elements, teams, played_gw)
    owned = {row["player_id"] for row in squad}

    try:
        price_history = ptrack.load_history()
        latest_prices = (
            price_history[price_history["ts"] == price_history["ts"].max()]
            if not price_history.empty
            else None
        )
    except (OSError, ValueError):
        latest_prices = None

    user_picks = data.picks.get(entry_id, {}).get(played_gw, {}).get("picks", [])
    risks = nrisk.squad_risks(
        user_picks,
        boot["elements"],
        teams,
        gws_played=played_gw,
        price_frame=latest_prices,
    )

    position = lown.user_position(data)
    ownership = lown.ownership_table(data)[:20]
    targets = lgap.gap_targets(data, remaining)
    overlap = lgap.overlap_table(data)[:15]

    try:
        movers = ptrack.movers(latest_prices) if latest_prices is not None else pd.DataFrame()
    except (KeyError, ValueError):
        movers = pd.DataFrame()
    if not movers.empty:
        movers = movers.assign(owned=movers["player_id"].isin(owned))

    notes = []
    unfinished = [
        e["id"]
        for e in boot["events"]
        if e["id"] == played_gw and not e.get("finished")
    ]
    if unfinished:
        notes.append(
            f"GW{played_gw} is not finalised, so all totals and ranks are provisional."
        )

    return Brief(
        generated_at=dt.datetime.now(dt.timezone.utc).replace(microsecond=0),
        target_gw=target_gw,
        deadline=deadline,
        remaining_gws=remaining,
        manager={
            "entry": entry_id,
            "name": standings_row.get("player_name", ""),
            "team_name": standings_row.get("entry_name", ""),
            "league": data.league_name,
            "n_managers": data.n_managers,
        },
        standing={
            "rank": standings_row.get("rank"),
            "total": standings_row.get("total"),
            "leader": max((r["total"] for r in data.standings), default=0),
            "median": sorted(r["total"] for r in data.standings)[
                len(data.standings) // 2
            ]
            if data.standings
            else 0,
            "net_swing": position.net_swing,
        },
        squad=squad,
        fixtures=_fixture_rows(data.fixtures.get(target_gw, []), teams),
        risks=risks,
        open_questions=nrisk.open_questions(risks),
        liabilities=position.liabilities[:12],
        differentials=position.differentials[:12],
        ownership=ownership,
        gap_targets=targets,
        overlap=overlap,
        price_movers=movers,
        captaincy=lown.captaincy_table(data, played_gw),
        projections=_latest_parquet("projections"),
        recommendation=_latest_json("decisions"),
        chip_plan=_latest_json("decisions", "chips.json"),
        model_health=_latest_json("model", "health.json"),
        notes=notes,
    )
