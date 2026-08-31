"""Snapshot and track FPL player price movements.

The 2026/27 API exposes several price fields that older FPL tooling does not
know about, discovered by inspecting ``bootstrap-static``:

``price_change_hourly_rate``
    Signed integer. Rate at which the player is accumulating net transfer
    pressure. Large positive means a rise is approaching.
``price_change_projections``
    List of ``{offset, projected_percent, likelihood}``. ``offset`` is days
    ahead (0 = tonight). ``projected_percent`` is progress toward a change,
    where +100 is a rise and -100 a fall. ``likelihood`` is a small signed
    integer confidence.
``price_change_locked_until`` / ``price_change_calibrating``
    FPL suppressing changes for a player. Treat projections as unreliable
    while either is set.

None of these are documented publicly, so the semantics above are inferred
from observed values and should be re-checked once history accumulates.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fplopt.data.paths import data_dir

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

_COLUMNS = [
    "ts",
    "player_id",
    "web_name",
    "team",
    "position",
    "now_cost",
    "cost_change_start",
    "cost_change_event",
    "transfers_in_event",
    "transfers_out_event",
    "net_transfers_event",
    "selected_by_percent",
    "hourly_rate",
    "proj_pct_today",
    "proj_likelihood_today",
    "proj_pct_tomorrow",
    "locked_until",
    "calibrating",
    "status",
]


@dataclass(frozen=True)
class Snapshot:
    """One point-in-time reading of every player's price state."""

    taken_at: dt.datetime
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)


def _history_path() -> Path:
    return data_dir() / "prices" / "history.parquet"


def _projection_at(projections: Any, offset: int) -> tuple[float | None, int | None]:
    """Pull (projected_percent, likelihood) for a given day offset."""
    if not isinstance(projections, list):
        return None, None
    for entry in projections:
        if not isinstance(entry, dict) or entry.get("offset") != offset:
            continue
        raw = entry.get("projected_percent")
        try:
            pct = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            pct = None
        likelihood = entry.get("likelihood")
        return pct, likelihood if isinstance(likelihood, int) else None
    return None, None


def _rows(bootstrap: dict[str, Any], taken_at: dt.datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for element in bootstrap["elements"]:
        projections = element.get("price_change_projections")
        pct_today, likelihood_today = _projection_at(projections, 0)
        pct_tomorrow, _ = _projection_at(projections, 1)
        in_event = element.get("transfers_in_event") or 0
        out_event = element.get("transfers_out_event") or 0
        rows.append(
            {
                "ts": taken_at,
                "player_id": element["id"],
                "web_name": element["web_name"],
                "team": element["team"],
                "position": element["element_type"],
                "now_cost": element["now_cost"],
                "cost_change_start": element.get("cost_change_start", 0),
                "cost_change_event": element.get("cost_change_event", 0),
                "transfers_in_event": in_event,
                "transfers_out_event": out_event,
                "net_transfers_event": in_event - out_event,
                "selected_by_percent": float(element.get("selected_by_percent") or 0.0),
                "hourly_rate": element.get("price_change_hourly_rate"),
                "proj_pct_today": pct_today,
                "proj_likelihood_today": likelihood_today,
                "proj_pct_tomorrow": pct_tomorrow,
                "locked_until": element.get("price_change_locked_until"),
                "calibrating": bool(element.get("price_change_calibrating")),
                "status": element.get("status"),
            }
        )
    return rows


def snapshot(bootstrap: dict[str, Any] | None = None) -> Snapshot:
    """Take a price reading and append it to the on-disk history.

    Passing ``bootstrap`` skips the network call, which keeps tests offline.
    """
    if bootstrap is None:
        from fplopt.data.fpl_api import FPLClient

        # The shared cache is keyed by date, which would return this morning's
        # prices on an afternoon poll. Price tracking needs live reads.
        bootstrap = FPLClient(cache_enabled=False).bootstrap()

    taken_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    frame = pd.DataFrame(_rows(bootstrap, taken_at), columns=_COLUMNS)

    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = pd.read_parquet(path)
        combined = pd.concat([previous, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts", "player_id"], keep="last")
    else:
        combined = frame
    combined.to_parquet(path, index=False)

    return Snapshot(taken_at=taken_at, frame=frame)


def load_history() -> pd.DataFrame:
    """Every snapshot taken so far, empty frame if tracking has not started."""
    path = _history_path()
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    return pd.read_parquet(path)


def movers(frame: pd.DataFrame | None = None, *, threshold: float = 75.0) -> pd.DataFrame:
    """Players close to a price change, most imminent first.

    ``threshold`` is absolute ``proj_pct_today``. Players whose changes FPL has
    locked or is recalibrating are excluded, since their projections do not
    currently mean anything.
    """
    if frame is None:
        history = load_history()
        if history.empty:
            return history
        frame = history[history["ts"] == history["ts"].max()]

    candidates = frame[
        frame["proj_pct_today"].notna()
        & (frame["proj_pct_today"].abs() >= threshold)
        & (~frame["calibrating"])
        & (frame["locked_until"].isna())
    ].copy()
    candidates["direction"] = candidates["proj_pct_today"].apply(
        lambda pct: "rise" if pct > 0 else "fall"
    )
    return candidates.sort_values(
        "proj_pct_today", key=lambda s: s.abs(), ascending=False
    )


def main() -> int:
    snap = snapshot()
    history = load_history()
    distinct = history["ts"].nunique() if not history.empty else 0
    print(f"snapshot at {snap.taken_at.isoformat()}: {len(snap)} players")
    print(f"history now holds {distinct} snapshot(s), {len(history)} rows")

    imminent = movers(snap.frame)
    if imminent.empty:
        print("no players within the movement threshold right now")
        return 0
    print(f"\n{len(imminent)} player(s) near a price change:")
    print(f"  {'player':16} {'dir':5} {'pct':>7} {'hourly':>8} {'net xfers':>10}")
    for _, row in imminent.head(25).iterrows():
        print(
            f"  {row['web_name'][:16]:16} {row['direction']:5} "
            f"{row['proj_pct_today']:>7.1f} {str(row['hourly_rate']):>8} "
            f"{row['net_transfers_event']:>10,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
