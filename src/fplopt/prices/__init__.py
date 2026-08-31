"""Price movement tracking.

FPL exposes point-in-time price signals only. Acting on them requires a
time series, so this package snapshots them on a schedule and builds history.
"""

from fplopt.prices.track import Snapshot, load_history, movers, snapshot

__all__ = ["Snapshot", "snapshot", "load_history", "movers"]
