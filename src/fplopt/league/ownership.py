"""League-relative ownership, effective ownership and threat-weighted EO.

Global ownership is close to irrelevant when the objective is winning one
45-manager league. What matters is what *these* managers own, weighted by how
dangerous each of them actually is.

Definitions used here
---------------------
owned_pct     fraction of league managers with the player in their 15
start_pct     fraction with the player in the starting XI (multiplier >= 1)
captain_pct   fraction with the armband on the player (includes triple captain)
eo_pct        owned_pct + captain_pct   -- the classic "effective ownership"
mult_eo_pct   mean(multiplier) across the league, i.e. start_pct + captain_pct
              (+2x extra for a triple captain, +bench for bench boost). This is
              the number that actually governs points swing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fetch import LeagueData, effective_multipliers

# --------------------------------------------------------------------- weights


def threat_weights(
    data: LeagueData,
    scheme: str = "contention",
    tau: float | None = None,
    exclude_user: bool = True,
) -> dict[int, float]:
    """Weight each manager by how much of a threat they are to the objective.

    The objective is winning the league, so the "field" that matters is the
    part of the table that can plausibly win it. Schemes:

    ``uniform``     every manager counts equally (classic league ownership).
    ``rank``        linear decay from 1.0 at rank 1 to ~0 at last place.
    ``gap``         proportional to points above the user.
    ``contention``  exp(-(leader - total) / tau); the leader is 1.0 and the
                    weight decays with the deficit to the leader. ``tau``
                    defaults to the league's points standard deviation, which
                    scales the decay to how spread this particular league is.

    Weights are normalised to sum to 1.
    """
    rows = [r for r in data.standings if not (exclude_user and r["entry"] == data.user_entry)]
    if not rows:
        return {}
    totals = [r["total"] for r in rows]
    leader = max(totals)
    n = len(rows)
    user_total = data.total_of(data.user_entry)

    raw: dict[int, float] = {}
    if scheme == "uniform":
        for r in rows:
            raw[r["entry"]] = 1.0
    elif scheme == "rank":
        for r in rows:
            raw[r["entry"]] = max(0.0, (n - r["rank"] + 1) / n)
    elif scheme == "gap":
        for r in rows:
            raw[r["entry"]] = max(0.0, float(r["total"] - user_total))
    elif scheme == "contention":
        if tau is None:
            mean = sum(totals) / n
            var = sum((t - mean) ** 2 for t in totals) / max(1, n - 1)
            tau = max(1.0, math.sqrt(var))
        for r in rows:
            raw[r["entry"]] = math.exp(-(leader - r["total"]) / tau)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    s = sum(raw.values())
    if s <= 0:
        return {k: 1.0 / n for k in raw}
    return {k: v / s for k, v in raw.items()}


# ----------------------------------------------------------------- ownership


@dataclass
class OwnershipRow:
    element: int
    name: str
    team: str
    position: str
    price: float
    owners: int
    owned_pct: float
    start_pct: float
    captain_pct: float
    eo_pct: float
    mult_eo_pct: float
    w_owned_pct: float
    w_eo_pct: float
    global_owned_pct: float
    points: int
    user_owns: bool
    user_mult: int

    @property
    def league_vs_global(self) -> float:
        return self.owned_pct - self.global_owned_pct


def _entry_picks(data: LeagueData, eid: int, gw: int) -> list[dict]:
    payload = data.picks.get(eid, {}).get(gw)
    return payload["picks"] if payload else []


def ownership_table(
    data: LeagueData,
    gw: int | None = None,
    weights: dict[int, float] | None = None,
) -> list[OwnershipRow]:
    """League ownership + EO for every player owned by at least one manager."""
    gw = gw if gw is not None else max(data.gameweeks)
    eids = data.entry_ids
    n = len(eids)
    if weights is None:
        weights = threat_weights(data)

    owners: dict[int, int] = {}
    starters: dict[int, int] = {}
    captains: dict[int, int] = {}
    mult_sum: dict[int, int] = {}
    w_own: dict[int, float] = {}
    w_eo: dict[int, float] = {}
    user_mult: dict[int, int] = {}

    for eid in eids:
        w = weights.get(eid, 0.0)
        for pk in _entry_picks(data, eid, gw):
            el = pk["element"]
            m = int(pk["multiplier"])
            owners[el] = owners.get(el, 0) + 1
            mult_sum[el] = mult_sum.get(el, 0) + m
            if m >= 1:
                starters[el] = starters.get(el, 0) + 1
            if pk["is_captain"]:
                captains[el] = captains.get(el, 0) + 1
            w_own[el] = w_own.get(el, 0.0) + w
            w_eo[el] = w_eo.get(el, 0.0) + w * (1.0 + (1.0 if pk["is_captain"] else 0.0))
            if eid == data.user_entry:
                user_mult[el] = m

    rows: list[OwnershipRow] = []
    for el, cnt in owners.items():
        p = data.players.get(el)
        owned = 100.0 * cnt / n
        cap = 100.0 * captains.get(el, 0) / n
        rows.append(
            OwnershipRow(
                element=el,
                name=p.web_name if p else f"#{el}",
                team=p.team_short if p else "???",
                position=p.position if p else "?",
                price=p.price if p else 0.0,
                owners=cnt,
                owned_pct=owned,
                start_pct=100.0 * starters.get(el, 0) / n,
                captain_pct=cap,
                eo_pct=owned + cap,
                mult_eo_pct=100.0 * mult_sum.get(el, 0) / n,
                w_owned_pct=100.0 * w_own.get(el, 0.0),
                w_eo_pct=100.0 * w_eo.get(el, 0.0),
                global_owned_pct=p.global_owned_pct if p else 0.0,
                points=data.season_points(el),
                user_owns=el in user_mult,
                user_mult=user_mult.get(el, 0),
            )
        )
    rows.sort(key=lambda r: (-r.eo_pct, -r.owned_pct, r.name))
    return rows


# ------------------------------------------------------- points already gained


@dataclass
class PlayerSwing:
    """Points the user has already gained or lost on one player vs the field."""

    element: int
    name: str
    team: str
    position: str
    price: float
    owned_pct: float
    eo_pct: float
    w_eo_pct: float
    user_mult_by_gw: dict[int, int] = field(default_factory=dict)
    field_mult_by_gw: dict[int, float] = field(default_factory=dict)
    points_by_gw: dict[int, int] = field(default_factory=dict)
    swing: float = 0.0  # positive = user ahead of the field on this player
    w_swing: float = 0.0  # same, against the threat-weighted field
    player_points: int = 0

    @property
    def kind(self) -> str:
        return "differential" if self.swing >= 0 else "liability"


def player_swings(
    data: LeagueData,
    weights: dict[int, float] | None = None,
) -> dict[int, PlayerSwing]:
    """Per-player points delta between the user and the league, to date.

    For each gameweek the user's realised contribution from a player is
    ``user_multiplier * player_points``. The field's is the mean of the same
    quantity across the league. The difference, summed over gameweeks, is
    exactly how many points that single player has made or cost the user
    relative to the average manager in this league.
    """
    if weights is None:
        weights = threat_weights(data)
    eids = data.entry_ids
    n = len(eids)
    user = data.user_entry

    mults: dict[int, dict[int, dict[int, int]]] = {}
    for gw in data.gameweeks:
        mults[gw] = {eid: effective_multipliers(data, eid, gw) for eid in eids}

    elements: set[int] = set()
    for gw in data.gameweeks:
        for m in mults[gw].values():
            elements.update(m)

    own_now = {r.element: r for r in ownership_table(data, weights=weights)}

    out: dict[int, PlayerSwing] = {}
    for el in elements:
        p = data.players.get(el)
        row = own_now.get(el)
        sw = PlayerSwing(
            element=el,
            name=p.web_name if p else f"#{el}",
            team=p.team_short if p else "???",
            position=p.position if p else "?",
            price=p.price if p else 0.0,
            owned_pct=row.owned_pct if row else 0.0,
            eo_pct=row.eo_pct if row else 0.0,
            w_eo_pct=row.w_eo_pct if row else 0.0,
            player_points=data.season_points(el),
        )
        total = 0.0
        wtotal = 0.0
        for gw in data.gameweeks:
            pts = data.player_points(el, gw)
            um = mults[gw][user].get(el, 0)
            fm = sum(mults[gw][e].get(el, 0) for e in eids) / n
            wfm = sum(weights.get(e, 0.0) * mults[gw][e].get(el, 0) for e in eids)
            sw.user_mult_by_gw[gw] = um
            sw.field_mult_by_gw[gw] = fm
            sw.points_by_gw[gw] = pts
            total += (um - fm) * pts
            wtotal += (um - wfm) * pts
        sw.swing = total
        sw.w_swing = wtotal
        out[el] = sw
    return out


@dataclass
class UserPosition:
    differentials: list[PlayerSwing]  # user owns, league mostly does not
    liabilities: list[PlayerSwing]  # league owns, user does not
    shared: list[PlayerSwing]
    total_differential_swing: float
    total_liability_swing: float
    net_swing: float


def user_position(
    data: LeagueData,
    diff_threshold: float = 33.0,
    liability_threshold: float = 25.0,
    weights: dict[int, float] | None = None,
) -> UserPosition:
    """Split the world into the user's differentials and the field's weapons."""
    if weights is None:
        weights = threat_weights(data)
    swings = player_swings(data, weights=weights)
    gw = max(data.gameweeks)
    user_squad = {pk["element"] for pk in _entry_picks(data, data.user_entry, gw)}

    diffs, liabs, shared = [], [], []
    for el, sw in swings.items():
        owns_now = el in user_squad
        ever_owned = any(m > 0 for m in sw.user_mult_by_gw.values()) or owns_now
        if owns_now and sw.owned_pct <= diff_threshold:
            diffs.append(sw)
        elif not owns_now and sw.owned_pct >= liability_threshold:
            liabs.append(sw)
        elif ever_owned:
            shared.append(sw)

    diffs.sort(key=lambda s: -s.owned_pct)
    liabs.sort(key=lambda s: s.swing)
    shared.sort(key=lambda s: -s.owned_pct)

    return UserPosition(
        differentials=diffs,
        liabilities=liabs,
        shared=shared,
        total_differential_swing=sum(s.swing for s in diffs),
        total_liability_swing=sum(s.swing for s in liabs),
        net_swing=sum(s.swing for s in swings.values()),
    )


def captaincy_table(data: LeagueData, gw: int) -> list[tuple[str, int, float, int]]:
    """Who the league captained in ``gw``: (name, count, pct, points delivered)."""
    counts: dict[int, int] = {}
    for eid in data.entry_ids:
        for pk in _entry_picks(data, eid, gw):
            if pk["is_captain"]:
                counts[pk["element"]] = counts.get(pk["element"], 0) + 1
    n = data.n_managers
    out = [
        (data.pname(el), c, 100.0 * c / n, data.player_points(el, gw))
        for el, c in counts.items()
    ]
    out.sort(key=lambda t: -t[1])
    return out


# --------------------------------------------------- concentration at the top


@dataclass
class TargetRow:
    element: int
    name: str
    team: str
    position: str
    price: float
    group_owners: int
    group_size: int
    league_owned_pct: float
    w_eo_pct: float
    points: int
    swing: float

    @property
    def group_pct(self) -> float:
        return 100.0 * self.group_owners / self.group_size if self.group_size else 0.0


def group_entries(data: LeagueData, top_n: int) -> list[int]:
    return [r["entry"] for r in data.standings if r["rank"] <= top_n]


def transfer_targets(
    data: LeagueData,
    top_n: int = 10,
    gw: int | None = None,
    weights: dict[int, float] | None = None,
    owned_by_user: bool = False,
) -> list[TargetRow]:
    """Players concentrated among the top ``top_n`` managers, ranked by how
    many of them own the player.

    Filtering to the leaders rather than the whole league is the point: a
    player owned by 40% of the table but by nobody in the top ten is not what
    is beating you.
    """
    gw = gw if gw is not None else max(data.gameweeks)
    group = group_entries(data, top_n)
    if weights is None:
        weights = threat_weights(data)
    swings = player_swings(data, weights=weights)
    rows = {r.element: r for r in ownership_table(data, gw=gw, weights=weights)}
    user_squad = {pk["element"] for pk in _entry_picks(data, data.user_entry, gw)}

    counts: dict[int, int] = {}
    for eid in group:
        for pk in _entry_picks(data, eid, gw):
            counts[pk["element"]] = counts.get(pk["element"], 0) + 1

    out: list[TargetRow] = []
    for el, c in counts.items():
        if (el in user_squad) != owned_by_user:
            continue
        p = data.players.get(el)
        r = rows.get(el)
        sw = swings.get(el)
        out.append(
            TargetRow(
                element=el,
                name=p.web_name if p else f"#{el}",
                team=p.team_short if p else "???",
                position=p.position if p else "?",
                price=p.price if p else 0.0,
                group_owners=c,
                group_size=len(group),
                league_owned_pct=r.owned_pct if r else 0.0,
                w_eo_pct=r.w_eo_pct if r else 0.0,
                points=data.season_points(el),
                swing=sw.swing if sw else 0.0,
            )
        )
    out.sort(key=lambda t: (-t.group_owners, t.swing))
    return out
