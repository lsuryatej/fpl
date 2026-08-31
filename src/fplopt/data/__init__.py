"""Data ingestion layer for the FPL optimizer.

Sources
-------
:mod:`fplopt.data.fpl_api`
    The official Fantasy Premier League API. ``/event/{gw}/live/`` is the
    ground truth for points verification.
:mod:`fplopt.data.history`
    The vaastav/Fantasy-Premier-League GitHub archive, 2016-17 onward,
    reconciled onto one schema.
:mod:`fplopt.data.understat`
    Understat league tables, per-player/team match logs and shot-level xG.
:mod:`fplopt.data.fbref`
    FBref. Blocked by Cloudflare at time of writing; the module raises
    :class:`~fplopt.data.fbref.FBrefUnavailable` rather than returning
    fabricated data.
:mod:`fplopt.data.store`
    Parquet cache helpers over ``data/``.
:mod:`fplopt.data.build`
    ``main()`` that refreshes everything and prints a summary.
"""

from __future__ import annotations

from . import fbref, fpl_api, history, store, understat
from .http import FetchError, HTTPStatusError

__all__ = [
    "fbref",
    "fpl_api",
    "history",
    "store",
    "understat",
    "FetchError",
    "HTTPStatusError",
]
