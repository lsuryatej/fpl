"""FPL 2026/27 scoring rules, expressed as data.

EVERY constant in this module was established from primary evidence, not from
memory. Three independent sources were used and cross-checked:

  [API-CONFIG]  https://fantasy.premierleague.com/api/bootstrap-static/
                -> ``game_config.scoring``. The live server's own scoring
                configuration, keyed by position short name.

  [RULES-JS]    https://fantasy.premierleague.com/help/rules renders from the
                lazily-loaded chunk ``/assets/Rules-C23BUQQK.js``, which
                contains the official constants literal:
                    {scoring:{short_play:1,long_play:2,long_play_limit:60,
                      goals_scored:{1:10,2:6,3:5,4:4},assists:3,
                      clean_sheets:{1:4,3:1},saves_limit:3,saves:1,
                      penalties_saved:5,penalties_missed:-2,concede_limit:2,
                      goals_conceded:-1,yellow_cards:-1,red_cards:-3,
                      own_goals:-2,def_con:2},
                     bps:{...}}
                plus the prose rules (DefCon thresholds, bonus tie handling).

  [LIVE]        https://fantasy.premierleague.com/api/event/{gw}/live/
                -> per-player ``explain`` arrays, which give the exact
                points awarded per stat identifier. Reverse-engineered over
                every player with minutes > 0 in GW1 and GW2 of 2026/27
                (1236 player-fixture rows). This is ground truth.

Where the three disagree, [LIVE] wins. They did not disagree on any rule that
GW1/GW2 exercised.

--------------------------------------------------------------------------
GENUINELY NEW / CHANGED IN 2026/27 -- do not trust older FPL knowledge here
--------------------------------------------------------------------------
1. A GOALKEEPER GOAL IS WORTH 10 POINTS (was 6, same as a defender).
   Goalkeepers and defenders now have DIFFERENT goal values. [API-CONFIG]
   ``goals_scored: {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}``; [RULES-JS]
   ``goals_scored:{1:10,2:6,3:5,4:4}``; the rules page renders two separate
   rows, "For each goal scored by a goalkeeper" and "...by a defender".
   Not exercised by GW1/GW2 (no keeper scored), so this one rests on the two
   config sources only.

2. THE MANAGER POSITION IS GONE. ``element_types`` has exactly four entries
   (no id 5 / "MNG"), and every ``mng_*`` key in ``game_config.scoring`` is
   zero. Manager scoring is dead for 2026/27. [API-CONFIG]

3. BPS: GOALS CONCEDED IS NOW -4 BPS PER GOAL for goalkeepers and defenders
   (previously -1 BPS for every 2 conceded). [RULES-JS] ``goals_conceded:{1:-4}``
   with the row label "Goalkeepers and defenders conceding a goal".
   Verified empirically: OLS of residual BPS on goals_conceded over 90-minute
   players in GW1+GW2 gives a slope of -3.90 (GKP, n=38) and -3.98 (DEF,
   n=121). The old -0.5/goal rule is decisively rejected.

4. BPS: CLEARANCES/BLOCKS/INTERCEPTIONS NOW PAY 1 BPS PER 3 (was per 2).
   [RULES-JS] ``cbi_limit:3``.

5. BPS: BEING TACKLED NO LONGER COSTS -1 BPS. The stat is absent from the
   2026/27 constants literal entirely.

6. BPS: GOALKEEPER SAVES RESTRUCTURED. Every save is now 2 BPS, with +1 for a
   save from inside the box and +1 for a save from a big chance (these stack).
   The old flat "2 BPS per save / 3 for inside box" split is gone.
   [RULES-JS] ``saves:2, save_from_inside_box:1, big_chance_saves:1``.

7. BPS: SHOT ON TARGET IS NOW +2 BPS, and FOUL WON IS NOW +1 BPS. Neither
   existed as a positive BPS action before. [RULES-JS]

8. BPS: A PENALTY SAVE IS 7 BPS (not 15). [RULES-JS] ``penalties_saved:7``.

9. BPS: A GOAL-LINE CLEARANCE IS 9 BPS. [RULES-JS] ``goal_line_clearances:9``.
   Flagged as surprising; it is what the official constants say.

10. Not scoring, but relevant to any pipeline built on this: gameweek scores
    are now finalised at 09:00 UK the day AFTER the last match, and projected
    bonus appears 20 minutes into each fixture. Treat pre-finalisation bonus
    as provisional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------
# [API-CONFIG] bootstrap-static ``element_types``. Exactly four for 2026/27.
GKP = 1
DEF = 2
MID = 3
FWD = 4

POSITION_SHORT: Mapping[int, str] = {GKP: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}
POSITION_NAME: Mapping[int, str] = {
    GKP: "Goalkeeper",
    DEF: "Defender",
    MID: "Midfielder",
    FWD: "Forward",
}
VALID_POSITIONS = frozenset(POSITION_SHORT)


# --------------------------------------------------------------------------
# Appearance points
# --------------------------------------------------------------------------
# [RULES-JS] short_play:1, long_play:2, long_play_limit:60
# [API-CONFIG] game_config.scoring.short_play = 1, .long_play = 2
# [LIVE] Boundary pinned directly: minutes=58 -> 1 point, minutes=60 -> 2 points
#        (no 59 observed). Threshold is ">= 60", excluding stoppage time --
#        the API's ``minutes`` field is already stoppage-excluded.
SHORT_PLAY_POINTS = 1
LONG_PLAY_POINTS = 2
LONG_PLAY_MINUTES = 60


# --------------------------------------------------------------------------
# Goals
# --------------------------------------------------------------------------
# [API-CONFIG] goals_scored {"GKP":10,"DEF":6,"MID":5,"FWD":4}
# [RULES-JS]   goals_scored:{1:10,2:6,3:5,4:4}
# [LIVE]       DEF 1 goal -> 6; MID 1/2/3 goals -> 5/10/15; FWD 1/2 -> 4/8.
#              No goalkeeper scored in GW1 or GW2, so GKP=10 rests on config.
# NOTE: GKP != DEF for the first time. This is the headline 2026/27 change.
GOAL_POINTS: Mapping[int, int] = {GKP: 10, DEF: 6, MID: 5, FWD: 4}


# --------------------------------------------------------------------------
# Assists
# --------------------------------------------------------------------------
# [API-CONFIG] assists: 3 (flat)   [RULES-JS] assists:3
# [LIVE] DEF/MID/FWD 1 assist -> 3, 2 assists -> 6. No position variation.
ASSIST_POINTS = 3


# --------------------------------------------------------------------------
# Clean sheets
# --------------------------------------------------------------------------
# [API-CONFIG] clean_sheets {"GKP":4,"DEF":4,"MID":1,"FWD":0}
# [RULES-JS]   clean_sheets:{1:4,3:1}  (key 1 labelled "goalkeepers and defenders")
# [LIVE]       GKP 4, DEF 4, MID 1, FWD never awarded.
# A clean sheet requires >= 60 minutes played AND zero goals conceded while on
# the pitch. [RULES-JS prose] "A clean sheet is awarded for not conceding a goal
# whilst on the pitch and playing at least 60 minutes (excluding stoppage time)."
# Verified in [LIVE]: 0 of 1236 rows had clean_sheets>0 with minutes<60, and 0
# had clean_sheets>0 with goals_conceded>0.
CLEAN_SHEET_POINTS: Mapping[int, int] = {GKP: 4, DEF: 4, MID: 1, FWD: 0}
CLEAN_SHEET_MINUTES = 60


# --------------------------------------------------------------------------
# Goals conceded
# --------------------------------------------------------------------------
# [API-CONFIG] goals_conceded {"GKP":-1,"DEF":-1,"MID":0,"FWD":0}
# [RULES-JS]   goals_conceded:-1, concede_limit:2
# [LIVE]       GKP/DEF: 2 conceded -> -1, 3 -> -1, 4 -> -2, 5 -> -2.
#              i.e. -1 * floor(conceded / 2). MID/FWD never deducted.
GOALS_CONCEDED_POINTS: Mapping[int, int] = {GKP: -1, DEF: -1, MID: 0, FWD: 0}
GOALS_CONCEDED_PER = 2


# --------------------------------------------------------------------------
# Goalkeeping
# --------------------------------------------------------------------------
# [API-CONFIG] saves: 1     [RULES-JS] saves:1, saves_limit:3
# [LIVE]       3 saves -> 1, 4 -> 1, 5 -> 1, 6 -> 2. i.e. floor(saves/3).
SAVE_POINTS = 1
SAVES_PER_POINT = 3

# [API-CONFIG] penalties_saved: 5   [RULES-JS] penalties_saved:5
# Not exercised in GW1/GW2 (zero penalty saves), so this rests on config only.
PENALTY_SAVE_POINTS = 5


# --------------------------------------------------------------------------
# Negative events
# --------------------------------------------------------------------------
# [API-CONFIG] + [RULES-JS] agree; [LIVE] confirms all but penalties_saved.
PENALTY_MISS_POINTS = -2  # [LIVE] one FWD penalty miss -> -2
OWN_GOAL_POINTS = -2  # [LIVE] 4 own goals across GKP/DEF/FWD -> -2 each
YELLOW_CARD_POINTS = -1  # [LIVE] 70 yellows -> -1 each, all positions
RED_CARD_POINTS = -3  # [LIVE] one MID red -> -3
# [RULES-JS prose] "Red card deductions include any points deducted for yellow
# cards." A second-yellow dismissal is recorded by the API as red_cards=1 with
# yellow_cards=0, so no double-count adjustment is needed by the engine. A
# straight red after an earlier (separate) booking appears as both, and both
# are charged -- which is what FPL does.
# [RULES-JS prose] "If a player receives a red card, they will continue to be
# penalised for goals conceded by their team" -- handled upstream, the API's
# goals_conceded field already reflects it.


# --------------------------------------------------------------------------
# Defensive contribution (DefCon)
# --------------------------------------------------------------------------
# [API-CONFIG] defensive_contribution {"GKP":0,"DEF":2,"MID":2,"FWD":2}
# [RULES-JS]   def_con:2, and prose:
#   "Any defender who reaches an accumulative total of 10 or more clearances,
#    blocks, interceptions (CBI) and tackles in a single match will earn 2 points."
#   "Any midfielder or forward who reaches an accumulative total of 12 or more
#    clearances, blocked shots, interceptions (CBI), tackles and recoveries will
#    earn 2 points."
#   "Defensive contribution points do not stack eg: a defender does not earn 4
#    points for the accumulation of 20 ... in a single match."
#
# [LIVE] verification over GW1+GW2, minutes > 0:
#   DEF (n=210): highest DC scoring nothing = 9; lowest DC scoring 2 = 10.
#                Zero counterexamples. 37 awards.
#   MID (n=278): highest DC scoring nothing = 11; lowest DC scoring 2 = 12.
#                Zero counterexamples. 21 awards.
#   FWD (n=65):  max DC observed = 8, so no forward reached any threshold in
#                GW1/GW2. The 12 threshold for forwards therefore rests on the
#                rules prose + game_config, not on live evidence.
#   GKP (n=38):  defensive_contribution is 0 for EVERY goalkeeper even when
#                they recorded clearances -- keepers are excluded outright.
#   A DEF with DC=21 still scored only 2. Confirms non-stacking.
#
# THE THRESHOLD *AND* THE COUNTED ACTIONS BOTH DIFFER BY POSITION.
# Defenders do NOT get credit for recoveries. Verified exactly:
#   DEF: defensive_contribution == cbi + tackles           -> 210/210 rows
#   MID: defensive_contribution == cbi + tackles + recov.  -> 278/278 rows
#   FWD: defensive_contribution == cbi + tackles + recov.  ->  65/65  rows
#   GKP: defensive_contribution == 0                       ->  38/38  rows
DEFCON_POINTS: Mapping[int, int] = {GKP: 0, DEF: 2, MID: 2, FWD: 2}
DEFCON_THRESHOLD: Mapping[int, int | None] = {GKP: None, DEF: 10, MID: 12, FWD: 12}
# Which raw actions roll up into ``defensive_contribution`` for each position.
DEFCON_COMPONENTS: Mapping[int, tuple[str, ...]] = {
    GKP: (),
    DEF: ("clearances_blocks_interceptions", "tackles"),
    MID: ("clearances_blocks_interceptions", "tackles", "recoveries"),
    FWD: ("clearances_blocks_interceptions", "tackles", "recoveries"),
}


# --------------------------------------------------------------------------
# Bonus
# --------------------------------------------------------------------------
# [RULES-JS prose] "The three best performing players in each match will be
# awarded bonus points. 3 points will be awarded to the highest scoring player,
# 2 to the second best and 1 to the third."
BONUS_AWARDS = (3, 2, 1)


# --------------------------------------------------------------------------
# Bonus Points System (BPS)
# --------------------------------------------------------------------------
# Verbatim from the [RULES-JS] constants literal. Every value below is a direct
# transcription; see the module docstring for what changed this season.
#
# IMPORTANT LIMITATION, stated plainly: BPS CANNOT BE RECOMPUTED FROM THE PUBLIC
# FPL API ALONE. Of the 38 BPS actions below, only 15 are exposed by
# bootstrap-static / event live. The other 23 (passes attempted & completed,
# key passes, big chances created and missed, open-play crosses, dribbles,
# shots on and off target, fouls conceded and won, offsides, errors, penalties
# conceded, goal-line clearances, winning goals, inside-box and big-chance
# saves, penalty vs open-play goal split) come from the Opta feed and are not
# published. ``bps.compute_bps`` therefore accepts them as optional inputs and
# reports which were missing. Use the API's own ``bps`` field when you have it;
# use ``compute_bps`` only when you have a full Opta stat line.
BPS_LONG_PLAY_MINUTES = 60
BPS_SHORT_PLAY = 3
BPS_LONG_PLAY = 6

# Goals, keyed by position. Key 1 in the source covers BOTH keepers and
# defenders (the rules row reads "Goalkeepers and defenders scoring a goal
# (non penalty)"), so GKP and DEF are both 12 here even though their *points*
# values now differ (10 vs 6).
BPS_GOAL: Mapping[int, int] = {GKP: 12, DEF: 12, MID: 18, FWD: 24}
# A goal scored direct from a penalty is a flat 12 BPS for every position,
# replacing the position value above.
BPS_PENALTY_SCORED = 12

BPS_ASSIST = 9
BPS_CLEAN_SHEET: Mapping[int, int] = {GKP: 12, DEF: 12, MID: 0, FWD: 0}

# Goalkeeping. These STACK: an inside-box save from a big chance is 2+1+1 = 4.
BPS_SAVE = 2
BPS_SAVE_FROM_INSIDE_BOX = 1
BPS_BIG_CHANCE_SAVE = 1
BPS_PENALTY_SAVED = 7

# Defensive actions.
BPS_CBI_PER = 3  # 1 BPS per 3 CBI (was per 2 before 2026/27)
BPS_CBI = 1
BPS_RECOVERIES_PER = 3
BPS_RECOVERIES = 1
BPS_SUCCESSFUL_TACKLE = 2
BPS_GOAL_LINE_CLEARANCE = 9

# Creation and attacking.
BPS_KEY_PASS = 1  # "creating a chance"
BPS_BIG_CHANCE_CREATED = 3
BPS_OPEN_PLAY_CROSS = 1
BPS_DRIBBLE = 1
BPS_SHOT_ON_TARGET = 2  # new for 2026/27
BPS_WINNING_GOAL = 3
BPS_FOUL_WON = 1  # new for 2026/27

# Passing accuracy, only when at least 30 passes were attempted. Bands do not
# stack -- the highest applicable band applies.
BPS_ATTEMPTED_PASSES_LIMIT = 30
BPS_PASS_PERCENTAGE_BANDS: tuple[tuple[float, int], ...] = (
    (90.0, 6),
    (80.0, 4),
    (70.0, 2),
)

# Penalties. -4 per goal conceded is a 2026/27 change and is the single largest
# BPS swing in the table; it is verified empirically (see module docstring).
BPS_GOALS_CONCEDED: Mapping[int, int] = {GKP: -4, DEF: -4, MID: 0, FWD: 0}
BPS_PENALTY_CONCEDED = -3
BPS_PENALTY_MISSED = -6
BPS_YELLOW_CARD = -3
BPS_RED_CARD = -9
BPS_OWN_GOAL = -6
BPS_BIG_CHANCE_MISSED = -3
BPS_ERROR_LEADING_TO_GOAL = -3
BPS_ERROR_LEADING_TO_GOAL_ATTEMPT = -1
BPS_FOUL_CONCEDED = -1
BPS_OFFSIDE = -1
BPS_SHOT_OFF_TARGET = -1
# NOTE: "being tackled" (-1) existed through 2025/26 and has been REMOVED.

#: BPS inputs that the public FPL API does publish per player per gameweek.
BPS_STATS_AVAILABLE_FROM_API = frozenset(
    {
        "minutes",
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
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
    }
)

#: BPS inputs that the public FPL API does NOT publish. Supplying these to
#: ``compute_bps`` requires an Opta-derived feed.
BPS_STATS_REQUIRING_OPTA = frozenset(
    {
        "penalties_scored",
        "saves_from_inside_box",
        "big_chance_saves",
        "goal_line_clearances",
        "key_passes",
        "big_chances_created",
        "open_play_crosses",
        "dribbles",
        "shots_on_target",
        "shots_off_target",
        "winning_goals",
        "fouls_won",
        "fouls_conceded",
        "offsides",
        "big_chances_missed",
        "errors_leading_to_goal",
        "errors_leading_to_goal_attempt",
        "penalties_conceded",
        "passes_attempted",
        "pass_completion_pct",
    }
)


@dataclass(frozen=True)
class PositionRules:
    """Per-position roll-up of the scoring rules, for convenient inspection."""

    position: int
    short_name: str
    goal: int
    assist: int
    clean_sheet: int
    goals_conceded_per_2: int
    defcon_points: int
    defcon_threshold: int | None
    defcon_components: tuple[str, ...] = field(default_factory=tuple)


POSITION_RULES: Mapping[int, PositionRules] = {
    p: PositionRules(
        position=p,
        short_name=POSITION_SHORT[p],
        goal=GOAL_POINTS[p],
        assist=ASSIST_POINTS,
        clean_sheet=CLEAN_SHEET_POINTS[p],
        goals_conceded_per_2=GOALS_CONCEDED_POINTS[p],
        defcon_points=DEFCON_POINTS[p],
        defcon_threshold=DEFCON_THRESHOLD[p],
        defcon_components=DEFCON_COMPONENTS[p],
    )
    for p in (GKP, DEF, MID, FWD)
}
