"""Reconcile the engine against the FPL API's own published points.

This is the verification harness the scoring rules were derived with. It is
deliberately part of the package rather than a throwaway script: any change to
``rules.py`` should be re-run through it.

    uv run python -m fplopt.scoring.reconcile            # live, all finished GWs
    uv run python -m fplopt.scoring.reconcile 1 2        # specific gameweeks
    uv run python -m fplopt.scoring.reconcile --offline  # cached test fixtures

Two independent checks are run:

  1. Points. For every player with minutes > 0, recompute total points from
     raw stats and compare with the API's ``stats.total_points``, and compare
     the per-identifier breakdown against the API's ``explain`` array.
  2. Bonus. For every finished fixture, re-derive the 3/2/1 allocation from the
     published BPS values and compare with the published bonus.
"""

from __future__ import annotations

import datetime as _dt

import argparse
import json
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .bps import allocate_bonus
from .engine import explain_player
from .rules import POSITION_SHORT

BASE_URL = "https://fantasy.premierleague.com/api"
_USER_AGENT = "fplopt-scoring-reconcile/1.0"

#: Cached API payloads used by the offline path and by the test suite.
CACHE_DIR = Path(__file__).resolve().parents[3] / "tests" / "scoring" / "data"


def _fetch(path: str) -> Any:
    request = urllib.request.Request(
        f"{BASE_URL}/{path}", headers={"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)


def load_bootstrap(*, offline: bool = False) -> dict[str, Any]:
    if offline:
        return json.loads((CACHE_DIR / "bootstrap_elements.json").read_text())
    return _fetch("bootstrap-static/")


def load_live(gameweek: int, *, offline: bool = False) -> dict[str, Any]:
    if offline:
        return json.loads((CACHE_DIR / f"live_{gameweek}.json").read_text())
    return _fetch(f"event/{gameweek}/live/")


def load_fixtures(*, offline: bool = False) -> list[dict[str, Any]]:
    if offline:
        return json.loads((CACHE_DIR / "fixtures.json").read_text())
    return _fetch("fixtures/")


def positions_from_bootstrap(bootstrap: Mapping[str, Any]) -> dict[int, int]:
    return {e["id"]: e["element_type"] for e in bootstrap["elements"]}


def names_from_bootstrap(bootstrap: Mapping[str, Any]) -> dict[int, str]:
    return {e["id"]: e["web_name"] for e in bootstrap["elements"]}


@dataclass
class Mismatch:
    gameweek: int
    element: int
    name: str
    position: str
    expected: int
    actual: int
    stats: dict[str, Any]
    detail: str = ""

    def __str__(self) -> str:
        return (
            f"GW{self.gameweek} {self.name} ({self.position}, id={self.element}): "
            f"api={self.expected} engine={self.actual} {self.detail}"
        )


@dataclass
class Report:
    checked: int = 0
    matched: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    line_checked: int = 0
    line_matched: int = 0
    line_mismatches: list[str] = field(default_factory=list)
    by_position: Counter = field(default_factory=Counter)

    @property
    def rate(self) -> float:
        return self.matched / self.checked if self.checked else 0.0

    @property
    def line_rate(self) -> float:
        return self.line_matched / self.line_checked if self.line_checked else 0.0


def reconcile_points(
    gameweeks: Iterable[int],
    positions: Mapping[int, int],
    names: Mapping[int, str],
    *,
    offline: bool = False,
    live_by_gw: Mapping[int, Mapping[str, Any]] | None = None,
) -> Report:
    """Recompute every played player's points and compare with the API."""
    report = Report()
    for gameweek in gameweeks:
        live = (
            live_by_gw[gameweek]
            if live_by_gw is not None
            else load_live(gameweek, offline=offline)
        )
        for element in live["elements"]:
            stats = element["stats"]
            if stats["minutes"] <= 0:
                continue
            element_id = element["id"]
            position = positions[element_id]
            score = explain_player(stats, position)
            report.checked += 1
            report.by_position[POSITION_SHORT[position]] += 1
            if score.total_points == stats["total_points"]:
                report.matched += 1
            else:
                report.mismatches.append(
                    Mismatch(
                        gameweek=gameweek,
                        element=element_id,
                        name=names.get(element_id, str(element_id)),
                        position=POSITION_SHORT[position],
                        expected=stats["total_points"],
                        actual=score.total_points,
                        stats=dict(stats),
                        detail=f"lines={[(l.identifier, l.points) for l in score.lines]}",
                    )
                )

            # Line-item comparison against the API's own explain array.
            api_lines: dict[str, int] = {}
            for fixture in element["explain"]:
                for entry in fixture["stats"]:
                    api_lines[entry["identifier"]] = (
                        api_lines.get(entry["identifier"], 0) + entry["points"]
                    )
            engine_lines: dict[str, int] = {}
            for line in score.lines:
                engine_lines[line.identifier] = (
                    engine_lines.get(line.identifier, 0) + line.points
                )
            report.line_checked += 1
            if api_lines == engine_lines:
                report.line_matched += 1
            else:
                only_api = {k: v for k, v in api_lines.items() if engine_lines.get(k) != v}
                only_eng = {k: v for k, v in engine_lines.items() if api_lines.get(k) != v}
                report.line_mismatches.append(
                    f"GW{gameweek} {names.get(element_id, element_id)} "
                    f"({POSITION_SHORT[position]}): api={only_api} engine={only_eng}"
                )
    return report


@dataclass
class BonusReport:
    fixtures_checked: int = 0
    fixtures_matched: int = 0
    players_checked: int = 0
    players_matched: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def fixture_rate(self) -> float:
        return self.fixtures_matched / self.fixtures_checked if self.fixtures_checked else 0.0

    @property
    def player_rate(self) -> float:
        return self.players_matched / self.players_checked if self.players_checked else 0.0


def reconcile_bonus(
    fixtures: Iterable[Mapping[str, Any]],
    names: Mapping[int, str],
    *,
    gameweeks: Iterable[int] | None = None,
) -> BonusReport:
    """Re-derive 3/2/1 bonus from published BPS and compare with published bonus.

    Includes fixtures that are only ``finished_provisional``. Since 2026/27 a
    gameweek is not marked ``finished`` until 09:00 UK the following day, so
    restricting to ``finished`` would silently skip a whole round of matches
    whose bonus and BPS are already published.
    """
    wanted = set(gameweeks) if gameweeks is not None else None
    report = BonusReport()
    for fixture in fixtures:
        if not (fixture.get("finished") or fixture.get("finished_provisional")):
            continue
        if wanted is not None and fixture.get("event") not in wanted:
            continue
        stats = {s["identifier"]: s for s in fixture.get("stats", [])}
        if "bps" not in stats:
            continue
        bps_by_player = {
            entry["element"]: entry["value"]
            for side in ("h", "a")
            for entry in stats["bps"][side]
        }
        if not bps_by_player:
            continue
        actual_bonus = {
            entry["element"]: entry["value"]
            for side in ("h", "a")
            for entry in stats.get("bonus", {"h": [], "a": []})[side]
        }
        predicted = allocate_bonus(bps_by_player)
        predicted_nonzero = {k: v for k, v in predicted.items() if v}

        report.fixtures_checked += 1
        report.players_checked += len(bps_by_player)
        agree = True
        for player in bps_by_player:
            if predicted.get(player, 0) == actual_bonus.get(player, 0):
                report.players_matched += 1
            else:
                agree = False
        if agree:
            report.fixtures_matched += 1
        else:
            report.mismatches.append(
                f"fixture {fixture['id']} (GW{fixture.get('event')}): "
                f"predicted={{{', '.join(f'{names.get(k, k)}:{v}' for k, v in sorted(predicted_nonzero.items(), key=lambda kv: -kv[1]))}}} "
                f"actual={{{', '.join(f'{names.get(k, k)}:{v}' for k, v in sorted(actual_bonus.items(), key=lambda kv: -kv[1]))}}} "
                f"bps={sorted(((names.get(k, k), v) for k, v in bps_by_player.items()), key=lambda kv: -kv[1])[:6]}"
            )
    return report


def reconcile_bonus_from_live(
    gameweeks: Iterable[int],
    names: Mapping[int, str],
    *,
    offline: bool = False,
    live_by_gw: Mapping[int, Mapping[str, Any]] | None = None,
) -> BonusReport:
    """Bonus reconciliation driven off the live endpoint rather than fixtures.

    Stricter than :func:`reconcile_bonus`. The fixtures endpoint drops players
    whose BPS is exactly zero from its stat lists (6 such rows in GW1+GW2), so
    a fixtures-driven check never presents the allocator with the full match
    roster. The live endpoint publishes every player, and ``explain[].fixture``
    says which match each line belongs to, so the roster can be rebuilt exactly.
    """
    report = BonusReport()
    per_fixture: dict[int, dict[int, int]] = {}
    actual: dict[int, dict[int, int]] = {}
    for gameweek in gameweeks:
        live = (
            live_by_gw[gameweek]
            if live_by_gw is not None
            else load_live(gameweek, offline=offline)
        )
        for element in live["elements"]:
            stats = element["stats"]
            if stats["minutes"] <= 0:
                continue
            for fixture in element["explain"]:
                fixture_id = fixture["fixture"]
                per_fixture.setdefault(fixture_id, {})[element["id"]] = stats["bps"]
                actual.setdefault(fixture_id, {})[element["id"]] = stats["bonus"]

    for fixture_id, bps_by_player in sorted(per_fixture.items()):
        predicted = allocate_bonus(bps_by_player)
        report.fixtures_checked += 1
        report.players_checked += len(bps_by_player)
        agree = True
        for player in bps_by_player:
            if predicted.get(player, 0) == actual[fixture_id].get(player, 0):
                report.players_matched += 1
            else:
                agree = False
                report.mismatches.append(
                    f"fixture {fixture_id} {names.get(player, player)}: "
                    f"bps={bps_by_player[player]} predicted={predicted.get(player, 0)} "
                    f"actual={actual[fixture_id].get(player, 0)}"
                )
        report.fixtures_matched += int(agree)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gameweeks", nargs="*", type=int, help="gameweeks to check")
    parser.add_argument("--offline", action="store_true", help="use cached fixtures")
    args = parser.parse_args(argv)

    bootstrap = load_bootstrap(offline=args.offline)
    positions = positions_from_bootstrap(bootstrap)
    names = names_from_bootstrap(bootstrap)

    if args.gameweeks:
        gameweeks = args.gameweeks
    elif args.offline:
        gameweeks = [1, 2]
    else:
        # NOTE: FPL event objects have no "started" key. Selecting on it silently
        # yielded an empty list and the reconciliation never ran. A gameweek has
        # started once its deadline has passed.
        now = _dt.datetime.now(_dt.timezone.utc)
        gameweeks = [
            e["id"]
            for e in bootstrap["events"]
            if _dt.datetime.fromisoformat(
                e["deadline_time"].replace("Z", "+00:00")
            )
            <= now
        ]

    print(f"Reconciling gameweeks: {gameweeks}\n")

    report = reconcile_points(gameweeks, positions, names, offline=args.offline)
    print("== TOTAL POINTS ==")
    print(f"players with minutes > 0 checked : {report.checked}")
    print(f"exact total-points matches       : {report.matched}")
    print(f"match rate                       : {report.rate:.6%}")
    print(f"by position                      : {dict(report.by_position)}")
    for mismatch in report.mismatches:
        print(f"  MISMATCH {mismatch}")
    print()
    print("== PER-IDENTIFIER BREAKDOWN vs API explain ==")
    print(f"players checked                  : {report.line_checked}")
    print(f"identical breakdowns             : {report.line_matched}")
    print(f"match rate                       : {report.line_rate:.6%}")
    for line in report.line_mismatches[:40]:
        print(f"  MISMATCH {line}")
    print()

    live_bonus = reconcile_bonus_from_live(gameweeks, names, offline=args.offline)
    print("== BONUS ALLOCATION (BPS -> 3/2/1), full roster from live endpoint ==")
    print(f"fixtures checked                 : {live_bonus.fixtures_checked}")
    print(f"fixtures fully correct           : {live_bonus.fixtures_matched}")
    print(f"fixture match rate               : {live_bonus.fixture_rate:.6%}")
    print(f"player-fixture rows checked      : {live_bonus.players_checked}")
    print(f"player rows correct              : {live_bonus.players_matched}")
    print(f"player match rate                : {live_bonus.player_rate:.6%}")
    for line in live_bonus.mismatches:
        print(f"  MISMATCH {line}")
    print()

    fixtures = load_fixtures(offline=args.offline)
    bonus = reconcile_bonus(fixtures, names, gameweeks=gameweeks)
    print("== BONUS ALLOCATION (BPS -> 3/2/1), fixtures endpoint ==")
    print(f"finished fixtures checked        : {bonus.fixtures_checked}")
    print(f"fixtures fully correct           : {bonus.fixtures_matched}")
    print(f"fixture match rate               : {bonus.fixture_rate:.6%}")
    print(f"player-fixture rows checked      : {bonus.players_checked}")
    print(f"player rows correct              : {bonus.players_matched}")
    print(f"player match rate                : {bonus.player_rate:.6%}")
    for line in bonus.mismatches:
        print(f"  MISMATCH {line}")

    ok = (
        report.checked
        and not report.mismatches
        and not report.line_mismatches
        and not bonus.mismatches
        and not live_bonus.mismatches
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
