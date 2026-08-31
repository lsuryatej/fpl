"""Bonus Points System: BPS computation and 3/2/1 bonus allocation.

Two separate jobs live here, and they have very different confidence levels.

``allocate_bonus`` turns a fixture's BPS scores into bonus points. It is fully
verifiable from the public API and is reconciled at 100% against every finished
fixture in GW1 and GW2 of 2026/27.

``compute_bps`` turns a raw stat line into a BPS score. It CANNOT be driven
from the public FPL API alone: 23 of the 38 BPS actions come from the Opta feed
and are not published (passes, key passes, crosses, dribbles, big chances,
shots, fouls, offsides, errors, penalties conceded, goal-line clearances,
winning goals, inside-box and big-chance saves, and the penalty/open-play goal
split). Where you have the API's own ``bps`` field, use it. ``compute_bps``
exists for simulation and for when a full Opta line is available; it returns a
``BpsBreakdown`` that names exactly which unavailable inputs it had to assume
were zero.

All constants come from ``rules`` and are sourced there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Mapping, Sequence

from . import rules as R

__all__ = [
    "BpsBreakdown",
    "BpsLine",
    "allocate_bonus",
    "bonus_from_fixture_bps",
    "compute_bps",
]


@dataclass(frozen=True)
class BpsLine:
    identifier: str
    value: float
    bps: int
    label: str


@dataclass(frozen=True)
class BpsBreakdown:
    total: int
    lines: tuple[BpsLine, ...]
    #: Opta-only inputs that were absent from the stat line and treated as
    #: zero. A non-empty set means ``total`` is a lower bound on the truth for
    #: the positive actions and an upper bound for the negative ones -- in
    #: short, it is an estimate, not the real BPS.
    assumed_zero: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_exact(self) -> bool:
        return not self.assumed_zero


def _num(stats: Mapping[str, Any], key: str) -> float:
    value = stats.get(key, 0)
    return 0.0 if value is None else float(value)


def _int(stats: Mapping[str, Any], key: str) -> int:
    return int(_num(stats, key))


def compute_bps(stats: Mapping[str, Any], position: int) -> BpsBreakdown:
    """Compute a player's BPS for one fixture from a full stat line.

    Recognises every action in the 2026/27 BPS table. Keys not present are
    treated as zero; any Opta-only key that was absent is reported in
    ``BpsBreakdown.assumed_zero`` so callers cannot silently mistake an
    estimate for the real thing.

    ``penalties_scored`` is subtracted from ``goals_scored`` before the
    position-specific goal value is applied, because a goal scored direct from
    a penalty is a flat 12 BPS rather than the open-play value.
    """
    position = int(position)
    if position not in R.VALID_POSITIONS:
        raise ValueError(f"unknown position {position!r}; expected 1..4")

    lines: list[BpsLine] = []

    def add(identifier: str, value: float, bps: int, label: str) -> None:
        if bps:
            lines.append(BpsLine(identifier, value, bps, label))

    minutes = _int(stats, "minutes")
    if minutes <= 0:
        return BpsBreakdown(0, (), frozenset())
    if minutes >= R.BPS_LONG_PLAY_MINUTES:
        add("minutes", minutes, R.BPS_LONG_PLAY, f"Played {minutes} minutes (60+)")
    else:
        add("minutes", minutes, R.BPS_SHORT_PLAY, f"Played {minutes} minutes")

    goals = _int(stats, "goals_scored")
    pen_goals = min(_int(stats, "penalties_scored"), goals)
    open_play_goals = goals - pen_goals
    add("goals_scored", open_play_goals, open_play_goals * R.BPS_GOAL[position],
        "Open-play goals")
    add("penalties_scored", pen_goals, pen_goals * R.BPS_PENALTY_SCORED,
        "Goals direct from a penalty")

    assists = _int(stats, "assists")
    add("assists", assists, assists * R.BPS_ASSIST, "Assists")

    clean_sheets = _int(stats, "clean_sheets")
    add("clean_sheets", clean_sheets, clean_sheets * R.BPS_CLEAN_SHEET[position],
        "Clean sheet")

    saves = _int(stats, "saves")
    add("saves", saves, saves * R.BPS_SAVE, "Saves")
    inside_box = min(_int(stats, "saves_from_inside_box"), saves)
    add("save_from_inside_box", inside_box, inside_box * R.BPS_SAVE_FROM_INSIDE_BOX,
        "Saves from inside the box")
    big_chance_saves = min(_int(stats, "big_chance_saves"), saves)
    add("big_chance_saves", big_chance_saves, big_chance_saves * R.BPS_BIG_CHANCE_SAVE,
        "Saves from a big chance")
    pens_saved = _int(stats, "penalties_saved")
    add("penalties_saved", pens_saved, pens_saved * R.BPS_PENALTY_SAVED, "Penalties saved")

    cbi = _int(stats, "clearances_blocks_interceptions")
    add("clearances_blocks_interceptions", cbi, (cbi // R.BPS_CBI_PER) * R.BPS_CBI,
        f"Clearances/blocks/interceptions (1 per {R.BPS_CBI_PER})")
    recoveries = _int(stats, "recoveries")
    add("recoveries", recoveries, (recoveries // R.BPS_RECOVERIES_PER) * R.BPS_RECOVERIES,
        f"Recoveries (1 per {R.BPS_RECOVERIES_PER})")
    tackles = _int(stats, "tackles")
    add("successful_tackles", tackles, tackles * R.BPS_SUCCESSFUL_TACKLE,
        "Successful tackles")
    glc = _int(stats, "goal_line_clearances")
    add("goal_line_clearances", glc, glc * R.BPS_GOAL_LINE_CLEARANCE,
        "Goal-line clearances")

    key_passes = _int(stats, "key_passes")
    add("key_passes", key_passes, key_passes * R.BPS_KEY_PASS, "Chances created")
    bcc = _int(stats, "big_chances_created")
    add("big_chances_created", bcc, bcc * R.BPS_BIG_CHANCE_CREATED, "Big chances created")
    crosses = _int(stats, "open_play_crosses")
    add("open_play_crosses", crosses, crosses * R.BPS_OPEN_PLAY_CROSS,
        "Successful open-play crosses")
    dribbles = _int(stats, "dribbles")
    add("dribbles", dribbles, dribbles * R.BPS_DRIBBLE, "Successful dribbles")
    sot = _int(stats, "shots_on_target")
    add("shots_on_target", sot, sot * R.BPS_SHOT_ON_TARGET, "Shots on target")
    winning = _int(stats, "winning_goals")
    add("winning_goals", winning, winning * R.BPS_WINNING_GOAL, "Match-winning goal")
    fouls_won = _int(stats, "fouls_won")
    add("fouls_won", fouls_won, fouls_won * R.BPS_FOUL_WON, "Fouls won")

    # Pass-completion bands, highest applicable only, and only above the
    # attempted-passes floor.
    attempted = _int(stats, "passes_attempted")
    if attempted >= R.BPS_ATTEMPTED_PASSES_LIMIT:
        pct = _num(stats, "pass_completion_pct")
        for floor, award in R.BPS_PASS_PERCENTAGE_BANDS:
            if pct >= floor:
                add("pass_percentage", pct, award, f"{pct:.0f}% pass completion")
                break

    conceded = _int(stats, "goals_conceded")
    add("goals_conceded", conceded, conceded * R.BPS_GOALS_CONCEDED[position],
        "Goals conceded (-4 each for GKP/DEF)")
    pens_conceded = _int(stats, "penalties_conceded")
    add("penalties_conceded", pens_conceded, pens_conceded * R.BPS_PENALTY_CONCEDED,
        "Penalties conceded")
    pens_missed = _int(stats, "penalties_missed")
    add("penalties_missed", pens_missed, pens_missed * R.BPS_PENALTY_MISSED,
        "Penalties missed")
    yellows = _int(stats, "yellow_cards")
    add("yellow_cards", yellows, yellows * R.BPS_YELLOW_CARD, "Yellow cards")
    reds = _int(stats, "red_cards")
    add("red_cards", reds, reds * R.BPS_RED_CARD, "Red cards")
    own_goals = _int(stats, "own_goals")
    add("own_goals", own_goals, own_goals * R.BPS_OWN_GOAL, "Own goals")
    bcm = _int(stats, "big_chances_missed")
    add("big_chances_missed", bcm, bcm * R.BPS_BIG_CHANCE_MISSED, "Big chances missed")
    err_goal = _int(stats, "errors_leading_to_goal")
    add("errors_leading_to_goal", err_goal, err_goal * R.BPS_ERROR_LEADING_TO_GOAL,
        "Errors leading to a goal")
    err_shot = _int(stats, "errors_leading_to_goal_attempt")
    add("errors_leading_to_goal_attempt", err_shot,
        err_shot * R.BPS_ERROR_LEADING_TO_GOAL_ATTEMPT, "Errors leading to an attempt")
    fouls = _int(stats, "fouls_conceded")
    add("fouls_conceded", fouls, fouls * R.BPS_FOUL_CONCEDED, "Fouls conceded")
    offsides = _int(stats, "offsides")
    add("offsides", offsides, offsides * R.BPS_OFFSIDE, "Caught offside")
    off_target = _int(stats, "shots_off_target")
    add("shots_off_target", off_target, off_target * R.BPS_SHOT_OFF_TARGET,
        "Shots off target")

    assumed_zero = frozenset(
        key for key in R.BPS_STATS_REQUIRING_OPTA if stats.get(key) is None
    )
    return BpsBreakdown(sum(line.bps for line in lines), tuple(lines), assumed_zero)


def allocate_bonus(bps_by_player: Mapping[Hashable, int]) -> dict[Hashable, int]:
    """Allocate 3/2/1 bonus points across one fixture from BPS scores.

    Ties are resolved exactly as the official rules describe:

      * "If there is a tie for first place, Players 1 & 2 will receive 3 points
        each and Player 3 will receive 1 point."
      * "If there is a tie for second place, Player 1 will receive 3 points and
        Players 2 and 3 will receive 2 points each."
      * "If there is a tie for third place, Player 1 will receive 3 points,
        Player 2 will receive 2 points and Players 3 & 4 will receive 1 point
        each."

    Generalising those three examples: rank the DISTINCT BPS values in
    descending order. Everyone on the top value takes 3. That group consumes
    as many award slots as it has members, and the next distinct group takes
    whatever award slot is left -- so two players tied on top skip the 2 and
    the next group gets 1, and three or more tied on top exhaust the awards
    entirely. Reconciled at 100% against every finished fixture in GW1 and GW2.

    Only players with a BPS entry are considered; callers should exclude
    players who did not appear.
    """
    result: dict[Hashable, int] = {player: 0 for player in bps_by_player}
    if not bps_by_player:
        return result

    groups: dict[int, list[Hashable]] = {}
    for player, score in bps_by_player.items():
        groups.setdefault(int(score), []).append(player)

    slot = 0  # index into BONUS_AWARDS: 0 -> 3 pts, 1 -> 2 pts, 2 -> 1 pt
    for score in sorted(groups, reverse=True):
        if slot >= len(R.BONUS_AWARDS):
            break
        award = R.BONUS_AWARDS[slot]
        members = groups[score]
        for player in members:
            result[player] = award
        slot += len(members)

    return result


def bonus_from_fixture_bps(
    entries: Iterable[Mapping[str, Any]] | Sequence[tuple[Hashable, int]],
    *,
    element_key: str = "element",
    value_key: str = "value",
) -> dict[Hashable, int]:
    """Convenience wrapper for the FPL fixture ``stats`` BPS shape.

    The fixtures endpoint publishes BPS as
    ``{"identifier": "bps", "h": [{"value": 41, "element": 15}, ...], "a": [...]}``.
    Pass the concatenation of ``h`` and ``a``.
    """
    bps_by_player: dict[Hashable, int] = {}
    for entry in entries:
        if isinstance(entry, Mapping):
            bps_by_player[entry[element_key]] = int(entry[value_key])
        else:
            player, score = entry
            bps_by_player[player] = int(score)
    return allocate_bonus(bps_by_player)
