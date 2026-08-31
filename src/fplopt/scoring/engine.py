"""Exact FPL 2026/27 points engine.

``score_player`` reproduces the integer total that the FPL API reports for a
player in a gameweek. ``explain_player`` returns the same number broken into
line items whose identifiers match the API's own ``explain`` array, so the two
can be diffed directly.

Reconciled against ``event/{1,2}/live/`` for the 2026/27 season -- see
``tests/scoring/test_engine.py`` and the ``reconcile`` module for the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import rules as R

__all__ = [
    "PointsLine",
    "PlayerScore",
    "defensive_contribution",
    "explain_player",
    "score_player",
]


@dataclass(frozen=True)
class PointsLine:
    """One row of a points breakdown.

    ``identifier`` matches the FPL API's ``explain[].stats[].identifier`` so a
    breakdown can be compared row-for-row against the official one.
    """

    identifier: str
    value: int
    points: int
    label: str

    def as_api_dict(self) -> dict[str, int | str]:
        """Shape used by the FPL live endpoint, for direct comparison."""
        return {
            "identifier": self.identifier,
            "points": self.points,
            "value": self.value,
            "points_modification": 0,
        }


@dataclass(frozen=True)
class PlayerScore:
    position: int
    total_points: int
    lines: tuple[PointsLine, ...]

    def get(self, identifier: str) -> int:
        return sum(line.points for line in self.lines if line.identifier == identifier)


def _int(stats: Mapping[str, Any], key: str) -> int:
    value = stats.get(key, 0)
    if value is None:
        return 0
    return int(value)


def _check_position(position: int) -> int:
    position = int(position)
    if position not in R.VALID_POSITIONS:
        raise ValueError(
            f"unknown position {position!r}; expected one of "
            f"{sorted(R.VALID_POSITIONS)} (1=GKP, 2=DEF, 3=MID, 4=FWD). "
            "Note that the manager position was removed in 2026/27."
        )
    return position


def defensive_contribution(stats: Mapping[str, Any], position: int) -> int:
    """Defensive-contribution tally for ``position``.

    Uses the API's own ``defensive_contribution`` field when present, since
    that is authoritative. Otherwise it is derived from the raw components,
    which differ by position: defenders count clearances + blocks +
    interceptions + tackles, while midfielders and forwards additionally count
    recoveries. Goalkeepers always score zero. See ``rules.DEFCON_COMPONENTS``
    for the verification that pinned this down.
    """
    position = _check_position(position)
    if position == R.GKP:
        return 0
    if stats.get("defensive_contribution") is not None:
        return _int(stats, "defensive_contribution")
    return sum(_int(stats, key) for key in R.DEFCON_COMPONENTS[position])


def _appearance_line(minutes: int) -> PointsLine | None:
    if minutes <= 0:
        return None
    if minutes >= R.LONG_PLAY_MINUTES:
        return PointsLine(
            "minutes", minutes, R.LONG_PLAY_POINTS,
            f"Played {minutes} minutes ({R.LONG_PLAY_MINUTES}+)",
        )
    return PointsLine(
        "minutes", minutes, R.SHORT_PLAY_POINTS,
        f"Played {minutes} minutes (under {R.LONG_PLAY_MINUTES})",
    )


def explain_player(
    stats: Mapping[str, Any],
    position: int,
    *,
    include_bonus: bool = True,
) -> PlayerScore:
    """Break a gameweek stat line into scoring components.

    Parameters
    ----------
    stats:
        A single-fixture (or single-gameweek) stat mapping. Recognised keys are
        those of the FPL live endpoint: ``minutes``, ``goals_scored``,
        ``assists``, ``clean_sheets``, ``goals_conceded``, ``own_goals``,
        ``penalties_saved``, ``penalties_missed``, ``yellow_cards``,
        ``red_cards``, ``saves``, ``bonus``, and either
        ``defensive_contribution`` or the raw
        ``clearances_blocks_interceptions`` / ``tackles`` / ``recoveries``.
        Missing keys are treated as zero.
    position:
        1=GKP, 2=DEF, 3=MID, 4=FWD.
    include_bonus:
        Include the ``bonus`` line. Set False to score the non-bonus part only,
        e.g. when bonus is still provisional mid-gameweek.

    Returns
    -------
    PlayerScore
        Lines are emitted only for non-zero point contributions, matching the
        API's behaviour (it omits zero-point stats from ``explain``).
    """
    position = _check_position(position)
    lines: list[PointsLine] = []

    minutes = _int(stats, "minutes")
    appearance = _appearance_line(minutes)
    if appearance is not None:
        lines.append(appearance)

    # A player with no minutes scores nothing at all, whatever else is in the
    # stat line. Verified in GW1/GW2: no zero-minute player received points.
    if minutes <= 0:
        return PlayerScore(position, 0, ())

    goals = _int(stats, "goals_scored")
    if goals:
        lines.append(
            PointsLine("goals_scored", goals, goals * R.GOAL_POINTS[position],
                       f"{goals} goal(s) as {R.POSITION_NAME[position]}")
        )

    assists = _int(stats, "assists")
    if assists:
        lines.append(
            PointsLine("assists", assists, assists * R.ASSIST_POINTS, f"{assists} assist(s)")
        )

    # The API's ``clean_sheets`` field already encodes the 60-minute and
    # zero-conceded conditions. If a caller hands us raw data without it, fall
    # back to deriving it.
    if "clean_sheets" in stats and stats["clean_sheets"] is not None:
        clean_sheets = _int(stats, "clean_sheets")
    else:
        clean_sheets = int(
            minutes >= R.CLEAN_SHEET_MINUTES and _int(stats, "goals_conceded") == 0
        )
    cs_points = clean_sheets * R.CLEAN_SHEET_POINTS[position]
    if cs_points:
        lines.append(PointsLine("clean_sheets", clean_sheets, cs_points, "Clean sheet"))

    conceded = _int(stats, "goals_conceded")
    conceded_points = (conceded // R.GOALS_CONCEDED_PER) * R.GOALS_CONCEDED_POINTS[position]
    if conceded_points:
        lines.append(
            PointsLine("goals_conceded", conceded, conceded_points,
                       f"{conceded} conceded (-1 per {R.GOALS_CONCEDED_PER})")
        )

    saves = _int(stats, "saves")
    save_points = (saves // R.SAVES_PER_POINT) * R.SAVE_POINTS
    if save_points:
        lines.append(
            PointsLine("saves", saves, save_points,
                       f"{saves} saves (1 per {R.SAVES_PER_POINT})")
        )

    pens_saved = _int(stats, "penalties_saved")
    if pens_saved:
        lines.append(
            PointsLine("penalties_saved", pens_saved, pens_saved * R.PENALTY_SAVE_POINTS,
                       f"{pens_saved} penalty save(s)")
        )

    pens_missed = _int(stats, "penalties_missed")
    if pens_missed:
        lines.append(
            PointsLine("penalties_missed", pens_missed, pens_missed * R.PENALTY_MISS_POINTS,
                       f"{pens_missed} penalty miss(es)")
        )

    own_goals = _int(stats, "own_goals")
    if own_goals:
        lines.append(
            PointsLine("own_goals", own_goals, own_goals * R.OWN_GOAL_POINTS,
                       f"{own_goals} own goal(s)")
        )

    yellows = _int(stats, "yellow_cards")
    if yellows:
        lines.append(
            PointsLine("yellow_cards", yellows, yellows * R.YELLOW_CARD_POINTS,
                       f"{yellows} yellow card(s)")
        )

    reds = _int(stats, "red_cards")
    if reds:
        lines.append(
            PointsLine("red_cards", reds, reds * R.RED_CARD_POINTS, f"{reds} red card(s)")
        )

    threshold = R.DEFCON_THRESHOLD[position]
    if threshold is not None:
        defcon = defensive_contribution(stats, position)
        if defcon >= threshold:
            # Does not stack: a single flat award however far past the line.
            lines.append(
                PointsLine("defensive_contribution", defcon, R.DEFCON_POINTS[position],
                           f"Defensive contribution {defcon} (>= {threshold})")
            )

    if include_bonus:
        bonus = _int(stats, "bonus")
        if bonus:
            lines.append(PointsLine("bonus", bonus, bonus, f"{bonus} bonus point(s)"))

    return PlayerScore(position, sum(line.points for line in lines), tuple(lines))


def score_player(stats: Mapping[str, Any], position: int) -> int:
    """Total FPL points for one player's gameweek stat line.

    ``position`` is the FPL ``element_type``: 1=GKP, 2=DEF, 3=MID, 4=FWD.
    """
    return explain_player(stats, position).total_points


def score_player_multi(
    fixtures: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    position: int,
) -> int:
    """Total points across several fixtures in one gameweek (double gameweeks).

    Points are computed per fixture and summed, because the appearance,
    clean-sheet, saves, goals-conceded and DefCon rules all reset per match.
    Summing the gameweek-aggregated stat line instead would be wrong.
    """
    return sum(score_player(f, position) for f in fixtures)
