"""Manager state: squad with purchase prices, bank, free transfers, chips.

Everything downstream (``squad``, ``multiweek``, ``chips``) is built against
:class:`ManagerState`, which is either constructed directly (tests, synthetic
scenarios) or produced by :func:`load_state_from_api` against the live FPL
API for a real manager.

Game-rule constants below were read from the official API's
``bootstrap-static`` -> ``game_settings`` payload, not re-derived or assumed
(see the task brief). They are the single source of truth other ``optim``
modules import from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from fplopt.data.fpl_api import FPLClient

__all__ = [
    "GKP",
    "DEF",
    "MID",
    "FWD",
    "POSITIONS",
    "SQUAD_SIZE",
    "SQUAD_COMPOSITION",
    "SQUAD_TEAM_LIMIT",
    "BUDGET_TENTHS",
    "STARTING_XI_SIZE",
    "MIN_STARTING_GKP",
    "MIN_STARTING_DEF",
    "MIN_STARTING_FWD",
    "BENCH_SIZE",
    "MAX_FREE_TRANSFERS_BANKED",
    "TRANSFER_HIT_COST",
    "MAX_TRANSFERS_PER_GW",
    "CHIP_NAMES",
    "CHIP_COUNT_PER_HALF",
    "PlayerHolding",
    "ManagerState",
    "selling_price",
    "simulate_free_transfers",
    "load_state_from_api",
]

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Verified game rules (official API `game_settings`, 2026/27 season).
# --------------------------------------------------------------------------
GKP, DEF, MID, FWD = 1, 2, 3, 4
POSITIONS = (GKP, DEF, MID, FWD)
POSITION_NAME = {GKP: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}

SQUAD_SIZE = 15
SQUAD_COMPOSITION: Mapping[int, int] = {GKP: 2, DEF: 5, MID: 5, FWD: 3}
SQUAD_TEAM_LIMIT = 3
BUDGET_TENTHS = 1000  # GBP100.0m, in tenths (FPL `now_cost` units)

STARTING_XI_SIZE = 11
MIN_STARTING_GKP = 1
MIN_STARTING_DEF = 3
MIN_STARTING_FWD = 1
BENCH_SIZE = 4  # 1 GK + 3 outfield, ordered

# `max_extra_free_transfers: 4` -> up to 5 total free transfers banked.
MAX_FREE_TRANSFERS_BANKED = 5
TRANSFER_HIT_COST = 4  # points per transfer beyond the free allowance
MAX_TRANSFERS_PER_GW = 20  # hard API cap

CHIP_NAMES = ("wildcard", "free_hit", "bench_boost", "triple_captain")
CHIP_COUNT_PER_HALF = 1  # one of each chip is usable per half-season


def selling_price(purchase_price: int, current_price: int) -> int:
    """Return the tenths-of-a-million price received for selling a player.

    FPL's sell-on rule: on a PROFIT, the manager keeps the purchase price plus
    half the profit, rounded DOWN to the nearest 0.1m (a 50% fee on profit
    only). A loss is never discounted further -- the manager simply receives
    the current price.

    Example: bought at 75 (7.5m), now worth 80 (8.0m) -> profit 5 -> half of 5
    is 2.5, rounded down to 2 -> sells for 77, not 80.

    Both arguments and the return value are integers in tenths of a million,
    matching the FPL API's ``now_cost`` convention. Floor division on an
    integer profit-in-tenths is exactly "round down to the nearest 0.1m"
    because a tenth is the smallest unit prices are expressed in.
    """
    if current_price <= purchase_price:
        return current_price
    profit = current_price - purchase_price
    return purchase_price + profit // 2


@dataclass(frozen=True)
class PlayerHolding:
    """One player in the manager's 15-man squad, with cost-basis tracking."""

    player_id: int
    position: int
    team: int
    purchase_price: int
    current_price: int
    is_captain: bool = False
    is_vice_captain: bool = False
    multiplier: int = 1  # 0 benched, 1 starter, 2 captain, 3 triple-captained

    def __post_init__(self) -> None:
        if self.position not in POSITIONS:
            raise ValueError(f"Invalid position {self.position!r} for player {self.player_id}")

    @property
    def profit(self) -> int:
        """Unrealised profit (current price minus purchase price), tenths."""
        return self.current_price - self.purchase_price

    @property
    def selling_price(self) -> int:
        """What this holding would sell for right now, after the sell-on fee."""
        return selling_price(self.purchase_price, self.current_price)


@dataclass
class ManagerState:
    """A manager's full state as of the gameweek about to be planned for.

    ``as_of_gw`` is the gameweek the squad/bank/free-transfer figures describe
    (i.e. the *next* gameweek a transfer decision applies to -- typically the
    upcoming deadline, not the last one played).
    """

    entry_id: int
    as_of_gw: int
    bank: int
    free_transfers: int
    chips_remaining: dict[str, int]
    squad: tuple[PlayerHolding, ...]

    def __post_init__(self) -> None:
        if len(self.squad) != SQUAD_SIZE:
            raise ValueError(f"Squad must have {SQUAD_SIZE} players, got {len(self.squad)}")
        counts: dict[int, int] = {}
        teams: dict[int, int] = {}
        seen_ids: set[int] = set()
        for holding in self.squad:
            if holding.player_id in seen_ids:
                raise ValueError(f"Duplicate player {holding.player_id} in squad")
            seen_ids.add(holding.player_id)
            counts[holding.position] = counts.get(holding.position, 0) + 1
            teams[holding.team] = teams.get(holding.team, 0) + 1
        if counts != dict(SQUAD_COMPOSITION):
            raise ValueError(f"Squad composition {counts} does not match required {dict(SQUAD_COMPOSITION)}")
        for team, count in teams.items():
            if count > SQUAD_TEAM_LIMIT:
                raise ValueError(f"Team {team} has {count} players, exceeding the limit of {SQUAD_TEAM_LIMIT}")
        if not (1 <= self.free_transfers <= MAX_FREE_TRANSFERS_BANKED):
            raise ValueError(
                f"free_transfers must be between 1 and {MAX_FREE_TRANSFERS_BANKED}, got {self.free_transfers}"
            )
        if self.bank < 0:
            raise ValueError(f"bank cannot be negative, got {self.bank}")

    @property
    def squad_ids(self) -> tuple[int, ...]:
        return tuple(h.player_id for h in self.squad)

    @property
    def squad_value(self) -> int:
        """Total current market value of the squad, tenths."""
        return sum(h.current_price for h in self.squad)

    @property
    def sell_value(self) -> int:
        """What the whole squad would fetch if sold right now, tenths."""
        return sum(h.selling_price for h in self.squad)

    @property
    def team_value(self) -> int:
        """Bank plus what the squad would sell for -- the standard "ITB + squad" figure."""
        return self.bank + self.sell_value

    def by_position(self, position: int) -> tuple[PlayerHolding, ...]:
        return tuple(h for h in self.squad if h.position == position)

    def purchase_price_map(self) -> dict[int, int]:
        return {h.player_id: h.purchase_price for h in self.squad}

    def current_price_map(self) -> dict[int, int]:
        return {h.player_id: h.current_price for h in self.squad}

    def holding(self, player_id: int) -> PlayerHolding | None:
        for h in self.squad:
            if h.player_id == player_id:
                return h
        return None


# --------------------------------------------------------------------------
# Free-transfer bank reconstruction
# --------------------------------------------------------------------------
def simulate_free_transfers(
    current_rows: Sequence[Mapping[str, object]],
    chip_events: Sequence[Mapping[str, object]] = (),
    max_bank: int = MAX_FREE_TRANSFERS_BANKED,
) -> int:
    """Reconstruct the free-transfer bank from ``entry/{id}/history/`` rows.

    The FPL API does not expose the free-transfer counter directly, so this
    replays the accumulation rule gameweek by gameweek from ``current_rows``
    (``history["current"]``, one row per played gameweek with an
    ``event_transfers`` count):

      * The manager's *first* played gameweek (normally gameweek 1, but
        whatever ``min(event)`` is for a manager who joined the game late)
        is a squad *selection*, not a transfer -- it is skipped entirely and
        contributes no roll. Free-transfer accrual starts at 1 for the
        gameweek immediately after it.
      * From there, each gameweek: ``free_used = min(transfers_made, ft_before)``
        -- you can never use more free transfers than you had, and any
        transfers beyond that are hits, which do not draw down the bank
        further.
      * The next gameweek's bank is ``min(max_bank, ft_before - free_used + 1)``:
        one more transfer becomes free next week, capped at ``max_bank``.
      * A gameweek in which Wildcard or Free Hit was active PRESERVES the
        bank unchanged: the chip grants unlimited transfers without drawing
        the bank down, but the week also earns no new free transfer. An
        earlier version treated the chip week as "zero transfers made" and
        still applied the ``+1`` accrual, which reported 3 free transfers
        entering gameweek 4 when the game itself showed 2.

    This is a best-effort reconstruction from public history data, not a
    field the API returns -- treat the result as advisory. Callers who know
    the true figure (e.g. from the FPL web UI) should override
    ``ManagerState.free_transfers`` directly.

    Verified against the live API for entry 3539707: returns 2 entering
    gameweek 3 (2 gameweeks played, no transfers made), and 2 entering
    gameweek 4 after a wildcard was played in gameweek 3 -- both confirmed
    against what the game itself displayed.
    """
    if not current_rows:
        return 1
    rows_sorted = sorted(current_rows, key=lambda r: int(r["event"]))
    chip_weeks = {
        int(c["event"])
        for c in chip_events
        if c.get("name") in ("wildcard", "freehit", "free_hit")
    }
    ft = 1
    for row in rows_sorted[1:]:  # skip the manager's first played gameweek
        event = int(row["event"])
        if event in chip_weeks:
            # Bank held, no accrual.
            continue
        made = int(row.get("event_transfers", 0) or 0)
        free_used = min(made, ft)
        ft = min(max_bank, ft - free_used + 1)
    return ft


def _chips_remaining(chip_events: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Turn ``history["chips"]`` (chips already played) into counts remaining.

    Every chip starts at 2 uses per season (one per half). The API's chip
    identifiers observed in practice are ``wildcard``, ``freehit``, ``bboost``
    and ``3xc``; unrecognised names are logged and ignored rather than
    crashing the loader.
    """
    remaining = {name: 2 for name in CHIP_NAMES}
    api_to_internal = {
        "wildcard": "wildcard",
        "freehit": "free_hit",
        "free_hit": "free_hit",
        "bboost": "bench_boost",
        "bench_boost": "bench_boost",
        "3xc": "triple_captain",
        "triple_captain": "triple_captain",
    }
    for c in chip_events:
        raw_name = c.get("name")
        name = api_to_internal.get(raw_name)
        if name is None:
            log.warning("Unrecognised chip name %r in entry history; ignoring", raw_name)
            continue
        if remaining[name] > 0:
            remaining[name] -= 1
    return remaining


def _reconstruct_purchase_prices(
    squad_ids: Sequence[int],
    elements: Mapping[int, Mapping[str, object]],
    transfers: Sequence[Mapping[str, object]],
) -> dict[int, int]:
    """Reconstruct each held player's purchase price from transfer history.

    ``transfers`` is ``entry/{id}/transfers/``, newest first. For each
    currently-held player, the most recent transfer where they were bought
    (``element_in``) gives the exact price paid. A player with no such record
    has been held since the initial squad and was never transferred in; their
    purchase price is reconstructed as ``now_cost - cost_change_start``, the
    bootstrap-reported price at the start of the season.
    """
    prices: dict[int, int] = {}
    for t in transfers:
        pid_in = t.get("element_in")
        if pid_in in squad_ids and pid_in not in prices:
            prices[pid_in] = int(t["element_in_cost"])
    for pid in squad_ids:
        if pid not in prices:
            el = elements[pid]
            start_price = int(el["now_cost"]) - int(el.get("cost_change_start", 0) or 0)
            prices[pid] = start_price
    return prices


def load_state_from_api(entry_id: int, client: FPLClient | None = None) -> ManagerState:
    """Fetch and reconstruct a :class:`ManagerState` for a real manager.

    Uses (see ``fplopt.data.fpl_api.FPLClient``):

    * ``/bootstrap-static/`` for each player's position, team, current price
      and season-start price (``cost_change_start``).
    * ``/entry/{id}/history/`` for the bank, the per-gameweek transfer counts
      (to reconstruct the free-transfer bank) and the chips already played.
    * ``/entry/{id}/event/{gw}/picks/`` for the most recent squad, captain
      and vice-captain.
    * ``/entry/{id}/transfers/`` to reconstruct each held player's purchase
      price.

    The free-transfer figure is a simulation (see
    :func:`simulate_free_transfers`) since the API does not expose it
    directly -- it is correct as long as the manager's transfer history is
    complete and no undocumented rule change has occurred.
    """
    client = client or FPLClient()
    boot = client.bootstrap()
    elements: dict[int, dict[str, object]] = {int(e["id"]): e for e in boot["elements"]}

    history = client.entry_history(entry_id)
    current_rows = history.get("current", [])
    if not current_rows:
        raise ValueError(f"Entry {entry_id} has no gameweek history yet")
    last_row = current_rows[-1]
    as_of_gw = int(last_row["event"])
    bank = int(last_row["bank"])

    picks_payload = client.entry_picks(entry_id, as_of_gw)
    picks = picks_payload["picks"]
    squad_ids = [int(p["element"]) for p in picks]

    transfers = client.entry_transfers(entry_id)
    purchase_prices = _reconstruct_purchase_prices(squad_ids, elements, transfers)

    holdings = []
    for p in picks:
        pid = int(p["element"])
        el = elements[pid]
        holdings.append(
            PlayerHolding(
                player_id=pid,
                position=int(el["element_type"]),
                team=int(el["team"]),
                purchase_price=purchase_prices[pid],
                current_price=int(el["now_cost"]),
                is_captain=bool(p.get("is_captain", False)),
                is_vice_captain=bool(p.get("is_vice_captain", False)),
                multiplier=int(p.get("multiplier", 1) or 0),
            )
        )

    chip_events = history.get("chips", [])
    chips_remaining = _chips_remaining(chip_events)
    free_transfers = simulate_free_transfers(current_rows, chip_events)

    return ManagerState(
        entry_id=entry_id,
        as_of_gw=as_of_gw,
        bank=bank,
        free_transfers=free_transfers,
        chips_remaining=chips_remaining,
        squad=tuple(holdings),
    )
