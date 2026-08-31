"""Exact FPL 2026/27 scoring engine.

Every rule in :mod:`fplopt.scoring.rules` was established from primary
evidence (the live API's ``explain`` breakdowns, ``game_config.scoring``, and
the official rules page constants) rather than from prior-season knowledge.
Read that module's docstring first -- several rules changed for 2026/27.
"""

from .bps import BpsBreakdown, BpsLine, allocate_bonus, bonus_from_fixture_bps, compute_bps
from .engine import (
    PlayerScore,
    PointsLine,
    defensive_contribution,
    explain_player,
    score_player,
    score_player_multi,
)

__all__ = [
    "BpsBreakdown",
    "BpsLine",
    "PlayerScore",
    "PointsLine",
    "allocate_bonus",
    "bonus_from_fixture_bps",
    "compute_bps",
    "defensive_contribution",
    "explain_player",
    "score_player",
    "score_player_multi",
]
