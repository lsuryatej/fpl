"""Gap analysis: how far behind, how fast you must run, and against whom.

The central idea in the overlap section: points scored by players you and a
rival both own cannot move you past that rival. Only the non-overlapping part
of your squad does any work. A big squad is therefore mostly dead weight
against any given opponent, and the dead-weight fraction is measurable.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .fetch import LeagueData, effective_multipliers


@dataclass
class GapTarget:
    label: str
    rank: int | None
    total: int
    gap: int
    per_gw: float
    entry: int | None = None
    entry_name: str = ""


@dataclass
class OverlapRow:
    entry: int
    entry_name: str
    manager: str
    rank: int
    total: int
    gap: int
    per_gw: float
    shared_squad: int  # of 15
    shared_xi: int  # of the user's starting XI
    shared_pct: float
    dead_weight_pct: float  # share of the user's live multiplier mass that is shared
    live_edge_players: int  # user's non-shared starters
    shared_names: list[str] = field(default_factory=list)
    unique_user_names: list[str] = field(default_factory=list)
    unique_rival_names: list[str] = field(default_factory=list)


def league_totals(data: LeagueData) -> list[int]:
    return sorted((r["total"] for r in data.standings), reverse=True)


def gap_targets(data: LeagueData, remaining_gws: int) -> list[GapTarget]:
    """Gaps from the user to every interesting rung of the ladder."""
    rows = sorted(data.standings, key=lambda r: r["rank"])
    user_total = data.total_of(data.user_entry)
    user_rank = data.rank_of(data.user_entry)
    n = len(rows)
    totals = [r["total"] for r in rows]

    def rung(rank: int, label: str) -> GapTarget | None:
        if rank < 1 or n == 0:
            return None
        rank = min(rank, n)  # tiny leagues: clamp the rung to last place
        row = rows[rank - 1]
        gap = row["total"] - user_total
        return GapTarget(
            label=label,
            rank=rank,
            total=row["total"],
            gap=gap,
            per_gw=gap / remaining_gws if remaining_gws else float("inf"),
            entry=row["entry"],
            entry_name=row["entry_name"],
        )

    out: list[GapTarget] = []
    for rank, label in [(1, "1st"), (5, "top-5 (5th)"), (10, "top-10 (10th)")]:
        t = rung(rank, label)
        if t:
            out.append(t)

    median_total = int(statistics.median(totals))
    out.append(
        GapTarget(
            label="median",
            rank=None,
            total=median_total,
            gap=median_total - user_total,
            per_gw=(median_total - user_total) / remaining_gws if remaining_gws else 0.0,
        )
    )

    mean_total = statistics.mean(totals)
    out.append(
        GapTarget(
            label="league average",
            rank=None,
            total=int(round(mean_total)),
            gap=int(round(mean_total - user_total)),
            per_gw=(mean_total - user_total) / remaining_gws if remaining_gws else 0.0,
        )
    )

    if user_rank > 1:
        t = rung(user_rank - 1, f"{user_rank - 1}th (one place up)")
        if t:
            out.append(t)
    return out


def squad_overlap(
    data: LeagueData,
    rival_entry: int,
    gw: int | None = None,
    remaining_gws: int = 36,
) -> OverlapRow:
    """How much of the user's squad is unusable for catching this rival."""
    gw = gw if gw is not None else max(data.gameweeks)
    user = data.user_entry
    up = data.picks.get(user, {}).get(gw)
    rp = data.picks.get(rival_entry, {}).get(gw)
    if not up or not rp:
        raise KeyError(f"missing picks for {user}/{rival_entry} gw{gw}")

    user_sq = {p["element"] for p in up["picks"]}
    rival_sq = {p["element"] for p in rp["picks"]}
    user_mult = {p["element"]: int(p["multiplier"]) for p in up["picks"]}
    rival_mult = {p["element"]: int(p["multiplier"]) for p in rp["picks"]}

    shared = user_sq & rival_sq
    user_xi = {e for e, m in user_mult.items() if m >= 1}
    shared_xi = user_xi & {e for e, m in rival_mult.items() if m >= 1}

    # Dead weight: the fraction of the user's multiplier mass tied up in
    # players the rival also fields at the same multiplier. A shared player
    # captained by only one of the two still generates separation, so the
    # neutralised mass is min(user_mult, rival_mult).
    total_mass = sum(user_mult.values())
    dead_mass = sum(min(user_mult[e], rival_mult.get(e, 0)) for e in user_mult)
    dead_pct = 100.0 * dead_mass / total_mass if total_mass else 0.0

    total = data.total_of(rival_entry)
    gap = total - data.total_of(user)
    row = data.standings[[r["entry"] for r in data.standings].index(rival_entry)]

    return OverlapRow(
        entry=rival_entry,
        entry_name=row["entry_name"],
        manager=row["player_name"],
        rank=row["rank"],
        total=total,
        gap=gap,
        per_gw=gap / remaining_gws if remaining_gws else 0.0,
        shared_squad=len(shared),
        shared_xi=len(shared_xi),
        shared_pct=100.0 * len(shared) / max(1, len(user_sq)),
        dead_weight_pct=dead_pct,
        live_edge_players=len(user_xi - shared_xi),
        shared_names=sorted(data.pname(e) for e in shared),
        unique_user_names=sorted(data.pname(e) for e in (user_sq - rival_sq)),
        unique_rival_names=sorted(data.pname(e) for e in (rival_sq - user_sq)),
    )


def overlap_table(
    data: LeagueData, remaining_gws: int = 36, gw: int | None = None
) -> list[OverlapRow]:
    rows = [
        squad_overlap(data, r["entry"], gw=gw, remaining_gws=remaining_gws)
        for r in data.standings
        if r["entry"] != data.user_entry
    ]
    rows.sort(key=lambda r: r.rank)
    return rows


@dataclass
class CatchupPlan:
    remaining_gws: int
    user_total: int
    user_rank: int
    user_ppg: float
    league_ppg: float
    leader_ppg: float
    required_ppg_for_first: float
    required_edge_vs_leader: float
    required_edge_vs_top5: float
    required_edge_vs_top10: float
    required_edge_vs_median: float
    targets: list[GapTarget] = field(default_factory=list)


def catchup_plan(data: LeagueData, remaining_gws: int = 36) -> CatchupPlan:
    gws_played = len(data.gameweeks)
    rows = sorted(data.standings, key=lambda r: r["rank"])
    totals = [r["total"] for r in rows]
    user_total = data.total_of(data.user_entry)
    leader_total = totals[0]

    targets = gap_targets(data, remaining_gws)
    by_label = {t.label: t for t in targets}

    league_ppg = statistics.mean(totals) / gws_played
    leader_ppg = leader_total / gws_played
    user_ppg = user_total / gws_played

    return CatchupPlan(
        remaining_gws=remaining_gws,
        user_total=user_total,
        user_rank=data.rank_of(data.user_entry),
        user_ppg=user_ppg,
        league_ppg=league_ppg,
        leader_ppg=leader_ppg,
        required_ppg_for_first=leader_ppg + by_label["1st"].per_gw,
        required_edge_vs_leader=by_label["1st"].per_gw,
        required_edge_vs_top5=by_label["top-5 (5th)"].per_gw,
        required_edge_vs_top10=by_label["top-10 (10th)"].per_gw,
        required_edge_vs_median=by_label["median"].per_gw,
        targets=targets,
    )


def win_probability_context(data: LeagueData, remaining_gws: int = 36) -> dict:
    """Crude but honest framing of how big the deficit is in noise terms.

    Per-gameweek score differences between two managers in the same league have
    a standard deviation you can estimate directly from the observed spread of
    gameweek scores. Compare the required total edge against that noise to see
    whether the gap is a rounding error or a structural problem.
    """
    # Pool the *within-gameweek* spread. Pooling raw scores across gameweeks
    # would fold in the week-to-week shift in the global average, which is
    # common to everybody and creates no separation.
    by_gw: dict[int, list[int]] = {}
    for eid in data.entry_ids:
        for r in data.histories.get(eid, {}).get("current", []):
            by_gw.setdefault(r["event"], []).append(r["points"])
    variances = [statistics.pvariance(v) for v in by_gw.values() if len(v) > 1]
    sd_single = (sum(variances) / len(variances)) ** 0.5 if variances else 0.0
    # Difference of two independent managers in one GW.
    sd_diff = sd_single * (2 ** 0.5)
    sd_season_diff = sd_diff * (remaining_gws ** 0.5)

    user_total = data.total_of(data.user_entry)
    leader = max(r["total"] for r in data.standings)
    totals = sorted((r["total"] for r in data.standings), reverse=True)
    fifth = totals[4] if len(totals) > 4 else totals[-1]
    tenth = totals[9] if len(totals) > 9 else totals[-1]

    return {
        "sd_gw_score": sd_single,
        "sd_gw_diff": sd_diff,
        "sd_remaining_season_diff": sd_season_diff,
        "gap_to_first_in_sd": (leader - user_total) / sd_season_diff if sd_season_diff else 0,
        "gap_to_fifth_in_sd": (fifth - user_total) / sd_season_diff if sd_season_diff else 0,
        "gap_to_tenth_in_sd": (tenth - user_total) / sd_season_diff if sd_season_diff else 0,
    }
