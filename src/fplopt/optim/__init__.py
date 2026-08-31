"""Multi-week FPL squad optimizer.

Sub-modules:

``state``
    The manager's current situation: squad with individual purchase prices,
    bank, free transfers, chips remaining. Loads from the live FPL API.
``squad``
    Single-gameweek MILP: best 15, best XI, captain, bench order.
``multiweek``
    The core solver: a single MILP over a rolling horizon that jointly
    decides transfers, hits, free-transfer banking, starting XI and captaincy
    for every gameweek in the horizon at once.
``objective``
    Pluggable objective functions (plain expected points, or a risk-adjusted
    mini-league rank utility that rewards differentials).
``chips``
    Evaluates the best gameweek to play each chip by re-solving with that
    chip's relaxed constraints and comparing objective values.
``explain``
    Produces a reasoning trace for any recommended move: EP delta, the
    next-best alternative, binding constraints, and downstream effects.
"""

from __future__ import annotations
