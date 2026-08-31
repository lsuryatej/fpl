"""Per-rival profiling: squads, chips, transfer behaviour, playing style."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .fetch import LeagueData, effective_multipliers

# Chip inventory is split across two halves of the season.
CHIP_LABELS = {
    "wildcard": "WC",
    "freehit": "FH",
    "bboost": "BB",
    "3xc": "TC",
    "manager": "AM",
}


DEFAULT_CHIP_WINDOWS = {
    "wildcard": [(2, 19), (20, 38)],
    "freehit": [(2, 19), (20, 38)],
    "bboost": [(1, 19), (20, 38)],
    "3xc": [(1, 19), (20, 38)],
}


@dataclass
class RivalProfile:
    entry: int
    entry_name: str
    manager: str
    rank: int
    total: int
    gw_points: dict[int, int] = field(default_factory=dict)
    squad: list[int] = field(default_factory=list)
    starting_xi: list[int] = field(default_factory=list)
    captain: int | None = None
    vice: int | None = None
    team_value: float = 0.0
    bank: float = 0.0
    chips_used: list[tuple[str, int]] = field(default_factory=list)
    chips_left: dict[str, int] = field(default_factory=dict)
    n_transfers: int = 0
    hits_taken: int = 0
    points_on_bench: int = 0
    # style
    template_score: float = 0.0  # mean league ownership of their 15
    template_z: float = 0.0
    style: str = "balanced"
    overlap_with_user: int = 0
    threat_weight: float = 0.0
    gap_to_user: int = 0
    best_gw: int = 0
    worst_gw: int = 0
    consistency: float = 0.0  # stdev of GW scores

    @property
    def chips_left_h1(self) -> list[str]:
        return sorted(k for k, v in self.chips_left.items() if v > 0)


def _leave_one_out_ownership(
    data: LeagueData, gw: int
) -> tuple[dict[int, int], int]:
    counts: dict[int, int] = {}
    for eid in data.entry_ids:
        payload = data.picks.get(eid, {}).get(gw)
        if not payload:
            continue
        for pk in payload["picks"]:
            counts[pk["element"]] = counts.get(pk["element"], 0) + 1
    return counts, data.n_managers


def build_profiles(
    data: LeagueData,
    weights: dict[int, float] | None = None,
) -> dict[int, RivalProfile]:
    """One profile per league entry (the user included)."""
    from .ownership import threat_weights

    if weights is None:
        weights = threat_weights(data)

    gw = max(data.gameweeks)
    counts, n = _leave_one_out_ownership(data, gw)
    user_total = data.total_of(data.user_entry)

    user_payload = data.picks.get(data.user_entry, {}).get(gw)
    user_squad = {p["element"] for p in user_payload["picks"]} if user_payload else set()

    profiles: dict[int, RivalProfile] = {}
    for row in data.standings:
        eid = row["entry"]
        payload = data.picks.get(eid, {}).get(gw)
        picks = payload["picks"] if payload else []
        squad = [p["element"] for p in picks]
        hist = data.histories.get(eid, {})
        current = hist.get("current", [])
        last = current[-1] if current else {}

        used = [(c["name"], c["event"]) for c in hist.get("chips", [])]
        left: dict[str, int] = {}
        for chip, windows in DEFAULT_CHIP_WINDOWS.items():
            total_available = len(windows)
            spent = sum(1 for name, _ in used if name == chip)
            left[chip] = max(0, total_available - spent)

        # Leave-one-out league ownership of this manager's own squad.
        if squad:
            tmpl = statistics.mean(
                100.0 * (counts.get(el, 0) - 1) / max(1, n - 1) for el in squad
            )
        else:
            tmpl = 0.0

        gw_pts = {r["event"]: r["points"] for r in current}
        scores = list(gw_pts.values())

        profiles[eid] = RivalProfile(
            entry=eid,
            entry_name=row["entry_name"],
            manager=row["player_name"],
            rank=row["rank"],
            total=row["total"],
            gw_points=gw_pts,
            squad=squad,
            starting_xi=[p["element"] for p in picks if p["multiplier"] >= 1],
            captain=next((p["element"] for p in picks if p["is_captain"]), None),
            vice=next((p["element"] for p in picks if p["is_vice_captain"]), None),
            team_value=(last.get("value", 0) + last.get("bank", 0)) / 10.0,
            bank=last.get("bank", 0) / 10.0,
            chips_used=used,
            chips_left=left,
            n_transfers=len(data.transfers.get(eid, [])),
            hits_taken=sum(r.get("event_transfers_cost", 0) for r in current),
            points_on_bench=sum(r.get("points_on_bench", 0) for r in current),
            template_score=tmpl,
            overlap_with_user=len(user_squad & set(squad)),
            threat_weight=weights.get(eid, 0.0),
            gap_to_user=row["total"] - user_total,
            best_gw=max(scores) if scores else 0,
            worst_gw=min(scores) if scores else 0,
            consistency=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        )

    tmpls = [p.template_score for p in profiles.values()]
    mu = statistics.mean(tmpls)
    sd = statistics.pstdev(tmpls) or 1.0
    for p in profiles.values():
        p.template_z = (p.template_score - mu) / sd
        if p.template_z >= 0.75:
            p.style = "template"
        elif p.template_z <= -0.75:
            p.style = "differential"
        else:
            p.style = "balanced"
    return profiles


def squad_lines(data: LeagueData, profile: RivalProfile) -> list[str]:
    """Readable squad listing, starters first, captain marked."""
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    payload = data.picks.get(profile.entry, {}).get(max(data.gameweeks))
    if not payload:
        return []
    rows = []
    for pk in payload["picks"]:
        p = data.players.get(pk["element"])
        if not p:
            continue
        tag = ""
        if pk["is_captain"]:
            tag = "(C)"
        elif pk["is_vice_captain"]:
            tag = "(V)"
        rows.append(
            (
                pk["multiplier"] < 1,
                order.get(p.position, 9),
                -data.season_points(p.id),
                f"{p.web_name}{tag}",
                p.position,
                p.team_short,
                data.season_points(p.id),
                pk["multiplier"] < 1,
            )
        )
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    return [
        f"{'BENCH ' if r[7] else '      '}{r[4]} {r[3]:<20s} {r[5]:<4s} {r[6]:>3d}pts"
        for r in rows
    ]


def transfer_log(data: LeagueData, eid: int) -> list[str]:
    out = []
    for t in sorted(data.transfers.get(eid, []), key=lambda x: x["event"]):
        out.append(
            f"GW{t['event']}: {data.pname(t['element_out'])} -> {data.pname(t['element_in'])}"
        )
    return out


def style_summary(profiles: dict[int, RivalProfile]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in profiles.values():
        out[p.style] = out.get(p.style, 0) + 1
    return out
