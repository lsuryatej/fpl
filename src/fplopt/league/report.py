"""Text report: everything the league module knows, in one pass.

    uv run python -m fplopt.league.report
    uv run python -m fplopt.league.report --top 25 --rivals 12
"""

from __future__ import annotations

import argparse
import statistics
import sys

from .fetch import (
    DEFAULT_ENTRY_ID,
    DEFAULT_LEAGUE_ID,
    FplClient,
    LeagueData,
    load_league,
    verify_totals,
)
from .gap import catchup_plan, overlap_table, win_probability_context
from .ownership import (
    captaincy_table,
    ownership_table,
    player_swings,
    threat_weights,
    transfer_targets,
    user_position,
)
from .rivals import build_profiles, squad_lines, style_summary, transfer_log

W = 96


def rule(ch: str = "-") -> str:
    return ch * W


def header(title: str) -> str:
    return f"\n{rule('=')}\n{title}\n{rule('=')}"


def section(title: str) -> str:
    return f"\n{title}\n{rule()}"


# ---------------------------------------------------------------------- blocks


def block_overview(data: LeagueData, remaining: int) -> None:
    rows = sorted(data.standings, key=lambda r: r["rank"])
    totals = [r["total"] for r in rows]
    user = data.rank_of(data.user_entry)
    print(header(f"{data.league_name}  (league {data.league_id})"))
    print(f"Entries          : {data.n_managers}  (standings pages fetched: {data.n_pages})")
    print(f"Gameweeks played : {data.gameweeks}   next deadline: GW{data.next_gw}")
    for gw in data.gameweeks:
        pend = data.pending_fixtures(gw)
        state = "FINAL" if data.is_gw_final(gw) else f"PROVISIONAL ({len(pend)} fixture(s) unplayed)"
        gw_scores = [
            r["points"]
            for e in data.entry_ids
            for r in data.histories[e]["current"]
            if r["event"] == gw
        ]
        ev = next((e for e in data.events if e["id"] == gw), {})
        print(
            f"  GW{gw}: league avg {statistics.mean(gw_scores):5.1f}  "
            f"global avg {ev.get('average_entry_score', 0):>3}  "
            f"best {max(gw_scores):>3}  worst {min(gw_scores):>3}   [{state}]"
        )
        for f in pend:
            th = next((p.team_short for p in data.players.values() if p.team == f["team_h"]), "?")
            ta = next((p.team_short for p in data.players.values() if p.team == f["team_a"]), "?")
            print(f"        unplayed: {th} v {ta}  {f['kickoff_time']}")
    print(
        f"Totals           : leader {max(totals)}  top-5 cut {sorted(totals, reverse=True)[4]}  "
        f"top-10 cut {sorted(totals, reverse=True)[9]}  median {statistics.median(totals):.0f}  "
        f"mean {statistics.mean(totals):.1f}  last {min(totals)}"
    )
    print(
        f"You              : entry {data.user_entry} '{data.name_of(data.user_entry)}' "
        f"({data.manager_of(data.user_entry)})  rank {user}/{data.n_managers}  "
        f"total {data.total_of(data.user_entry)}"
    )


def block_verification(data: LeagueData) -> dict:
    print(section("VERIFICATION"))
    v = verify_totals(data)
    print(
        f"Recomputed season totals from picks x live points, minus hits: "
        f"{v['n_match']}/{v['n_entries']} match the standings endpoint "
        f"({100 * v['match_rate']:.1f}%)"
    )
    for m in v["mismatches"]:
        print(f"  MISMATCH {m['name']:<26s} api={m['api']} calc={m['calc']} delta={m['delta']:+d}")
    print(f"Entries fetched: {data.n_managers} (expected the full league, no truncated page)")
    return v


def block_ownership(data: LeagueData, weights: dict[int, float], top: int) -> None:
    print(section(f"LEAGUE-RELATIVE OWNERSHIP  --  top {top} by effective ownership"))
    print(
        "EO = ownership% + captaincy%.  wEO = the same but each manager weighted by "
        "title threat.\n"
        "Delta = league ownership minus global ownership: positive means this league is "
        "more concentrated\non the player than the wider game."
    )
    rows = ownership_table(data, weights=weights)
    print(
        f"\n{'#':>2} {'player':<16s}{'pos':>4}{'team':>5}{'£':>6}{'own%':>7}{'EO%':>7}"
        f"{'wEO%':>7}{'multEO':>8}{'glob%':>7}{'delta':>7}{'pts':>5}  you"
    )
    for i, r in enumerate(rows[:top], 1):
        mark = "OWN" if r.user_owns else " - "
        print(
            f"{i:>2} {r.name:<16s}{r.position:>4}{r.team:>5}{r.price:>6.1f}{r.owned_pct:>7.1f}"
            f"{r.eo_pct:>7.1f}{r.w_eo_pct:>7.1f}{r.mult_eo_pct:>8.1f}{r.global_owned_pct:>7.1f}"
            f"{r.league_vs_global:>+7.1f}{r.points:>5}  {mark}"
        )

    print(section("CAPTAINCY"))
    for gw in data.gameweeks:
        caps = captaincy_table(data, gw)
        picked = [
            data.pname(p["element"])
            for p in data.picks[data.user_entry][gw]["picks"]
            if p["is_captain"]
        ]
        print(
            f"GW{gw}: "
            + ", ".join(f"{n} {c} ({pct:.0f}%) -> {pts}pts" for n, c, pct, pts in caps[:6])
        )
        print(f"      you captained: {picked[0] if picked else 'n/a'}")


def block_user_position(data: LeagueData, weights: dict[int, float]) -> None:
    up = user_position(data, weights=weights)
    print(section("YOUR LIABILITIES  --  the league owns them, you do not"))
    print(
        "'cost' is the points this single player has already handed the average manager in "
        "this league\nover and above what he gave you: sum over GWs of "
        "(field_multiplier - your_multiplier) x points."
    )
    print(
        f"\n{'player':<16s}{'pos':>4}{'team':>5}{'£':>6}{'own%':>7}{'EO%':>7}{'wEO%':>7}"
        f"{'pts':>5}{'cost':>8}{'wcost':>8}"
    )
    for s in up.liabilities:
        print(
            f"{s.name:<16s}{s.position:>4}{s.team:>5}{s.price:>6.1f}{s.owned_pct:>7.1f}"
            f"{s.eo_pct:>7.1f}{s.w_eo_pct:>7.1f}{s.player_points:>5}{s.swing:>+8.1f}"
            f"{s.w_swing:>+8.1f}"
        )
    print(f"{'TOTAL':<16s}{'':>39}{up.total_liability_swing:>+8.1f}")

    print(section("YOUR DIFFERENTIALS  --  you own them, the league mostly does not"))
    print(
        f"{'player':<16s}{'pos':>4}{'team':>5}{'£':>6}{'own%':>7}{'EO%':>7}{'wEO%':>7}"
        f"{'pts':>5}{'gain':>8}{'wgain':>8}  verdict"
    )
    for s in up.differentials:
        verdict = "PAID OFF" if s.swing > 1.0 else ("neutral" if s.swing > -1.0 else "FAILED")
        print(
            f"{s.name:<16s}{s.position:>4}{s.team:>5}{s.price:>6.1f}{s.owned_pct:>7.1f}"
            f"{s.eo_pct:>7.1f}{s.w_eo_pct:>7.1f}{s.player_points:>5}{s.swing:>+8.1f}"
            f"{s.w_swing:>+8.1f}  {verdict}"
        )
    print(f"{'TOTAL':<16s}{'':>39}{up.total_differential_swing:>+8.1f}")

    swings = player_swings(data, weights=weights)
    ranked = sorted(swings.values(), key=lambda s: s.swing)
    print(section("BIGGEST SINGLE-PLAYER SWINGS AGAINST YOU (all players, any ownership)"))
    for s in ranked[:12]:
        print(
            f"  {s.name:<16s}{s.team:>5}  own {s.owned_pct:5.1f}%  EO {s.eo_pct:6.1f}%  "
            f"pts {s.player_points:>3}  cost {s.swing:+7.1f}"
        )
    print("  " + rule()[:70])
    print(f"  net position vs the field across all players: {sum(s.swing for s in swings.values()):+.1f}")


def block_targets(data: LeagueData, weights: dict[int, float], top_n: int = 10) -> None:
    print(section(f"CONCENTRATION AMONG THE TOP {top_n}  --  what is actually beating you"))
    print(
        f"League-wide ownership is the wrong filter when the goal is catching the leaders. "
        f"These are\nthe players the top {top_n} managers hold that you do not."
    )
    rows = transfer_targets(data, top_n=top_n, weights=weights)
    print(
        f"\n{'player':<16s}{'pos':>4}{'team':>5}{'£':>6}{'top' + str(top_n):>8}"
        f"{'league%':>9}{'wEO%':>8}{'pts':>5}{'cost':>8}"
    )
    for t in rows[:15]:
        print(
            f"{t.name:<16s}{t.position:>4}{t.team:>5}{t.price:>6.1f}"
            f"{str(t.group_owners) + '/' + str(t.group_size):>8}{t.league_owned_pct:>9.1f}"
            f"{t.w_eo_pct:>8.1f}{t.points:>5}{t.swing:>+8.1f}"
        )
    held = transfer_targets(data, top_n=top_n, weights=weights, owned_by_user=True)
    print(f"\nPlayers you hold, and how many of the top {top_n} hold them too:")
    for t in sorted(held, key=lambda x: -x.group_owners):
        print(
            f"  {t.name:<16s}{t.team:>5}{t.price:>6.1f}  "
            f"{t.group_owners}/{t.group_size} of the top {top_n}  "
            f"league {t.league_owned_pct:5.1f}%  {t.points:>3} pts"
        )
    orphans = [t for t in held if t.group_owners == 0]
    if orphans:
        print(
            f"\n{len(orphans)} of your 15 are owned by nobody in the top {top_n}: "
            + ", ".join(t.name for t in orphans)
            + "."
        )


def block_gap(data: LeagueData, remaining: int) -> None:
    print(header(f"GAP ANALYSIS  --  {remaining} gameweeks remain"))
    cp = catchup_plan(data, remaining)
    print(
        f"You: {cp.user_total} pts, rank {cp.user_rank}/{data.n_managers}, "
        f"{cp.user_ppg:.1f} pts/GW.  League {cp.league_ppg:.1f} pts/GW.  "
        f"Leader {cp.leader_ppg:.1f} pts/GW."
    )
    print(
        f"\n{'target':<28s}{'their total':>12}{'gap':>7}{'edge needed /GW':>18}"
        f"{'implied pts/GW':>17}"
    )
    for t in cp.targets:
        implied = cp.league_ppg + t.per_gw if t.label in ("median", "league average") else None
        rung_ppg = t.total / len(data.gameweeks)
        need = rung_ppg + t.per_gw
        print(
            f"{t.label:<28s}{t.total:>12}{t.gap:>7}{t.per_gw:>18.2f}{need:>17.1f}"
        )
    print(
        "\n'edge needed /GW' is how many points per gameweek you must beat that rung by, "
        "for 36 GWs.\n'implied pts/GW' assumes the rung keeps scoring at its current rate."
    )

    ctx = win_probability_context(data, remaining)
    print(
        f"\nNoise scale: within-gameweek SD of manager scores is {ctx['sd_gw_score']:.1f} pts; "
        f"the SD of the\ncumulative difference between two managers over {remaining} GWs is "
        f"{ctx['sd_remaining_season_diff']:.0f} pts."
    )
    print(
        f"  gap to 1st  = {ctx['gap_to_first_in_sd']:.2f} SD of that noise\n"
        f"  gap to 5th  = {ctx['gap_to_fifth_in_sd']:.2f} SD\n"
        f"  gap to 10th = {ctx['gap_to_tenth_in_sd']:.2f} SD"
    )


def block_overlap(data: LeagueData, remaining: int, n_rivals: int) -> None:
    print(section("SQUAD OVERLAP  --  how much of your squad cannot gain on each rival"))
    print(
        "'dead' is the share of your multiplier mass (starters + captain) neutralised because "
        "the rival\nfields the same player at the same multiplier. Only the rest of your team "
        "can close the gap."
    )
    rows = overlap_table(data, remaining_gws=remaining)
    print(
        f"\n{'rk':>3} {'entry':<26s}{'total':>7}{'gap':>6}{'/GW':>7}{'sq':>5}{'xi':>4}"
        f"{'dead%':>8}{'edge':>6}  shared"
    )
    for r in rows[:n_rivals]:
        print(
            f"{r.rank:>3} {r.entry_name[:26]:<26s}{r.total:>7}{r.gap:>6}{r.per_gw:>7.2f}"
            f"{r.shared_squad:>4}/15{r.shared_xi:>4}{r.dead_weight_pct:>8.1f}"
            f"{r.live_edge_players:>6}  {', '.join(r.shared_names)}"
        )
    avg_dead = statistics.mean(r.dead_weight_pct for r in rows)
    avg_shared = statistics.mean(r.shared_squad for r in rows)
    print(
        f"\nAcross all {len(rows)} rivals: mean shared squad {avg_shared:.1f}/15, "
        f"mean dead weight {avg_dead:.1f}%."
    )
    top10 = [r for r in rows if r.rank <= 10]
    if top10:
        print(
            f"Against the top 10 only: mean shared {statistics.mean(r.shared_squad for r in top10):.1f}/15, "
            f"mean dead weight {statistics.mean(r.dead_weight_pct for r in top10):.1f}%."
        )


def block_rivals(data: LeagueData, weights: dict[int, float], n_rivals: int) -> None:
    print(header("RIVAL PROFILES"))
    profiles = build_profiles(data, weights=weights)
    styles = style_summary(profiles)
    print(
        f"Playing styles across {data.n_managers} managers: "
        + ", ".join(f"{k} {v}" for k, v in sorted(styles.items()))
    )
    print(
        "Template score = mean league ownership of a manager's own 15, computed "
        "leave-one-out.\nHigh = follows the crowd, low = hunting differentials."
    )
    ordered = sorted(profiles.values(), key=lambda p: p.rank)
    print(
        f"\n{'rk':>3} {'entry':<26s}{'mgr':<20s}{'tot':>5}{'TV':>7}{'tmpl':>7}{'z':>7}"
        f"{'style':>14}{'ovl':>5}{'trf':>5}{'hit':>5}{'bench':>7}  chips used"
    )
    for p in ordered:
        chips = ", ".join(f"{n}@GW{g}" for n, g in p.chips_used) or "-"
        print(
            f"{p.rank:>3} {p.entry_name[:26]:<26s}{p.manager[:20]:<20s}{p.total:>5}"
            f"{p.team_value:>7.1f}{p.template_score:>7.1f}{p.template_z:>+7.2f}"
            f"{p.style:>14}{p.overlap_with_user:>5}{p.n_transfers:>5}{p.hits_taken:>5}"
            f"{p.points_on_bench:>7}  {chips}"
        )

    chips_gone = [p for p in ordered if p.chips_used]
    print(
        f"\n{len(chips_gone)}/{data.n_managers} managers have already burned a first-half chip. "
        "Breakdown:"
    )
    tally: dict[str, int] = {}
    for p in ordered:
        for n, _ in p.chips_used:
            tally[n] = tally.get(n, 0) + 1
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    print(section(f"THE {n_rivals} MANAGERS THAT MATTER (by threat weight)"))
    threats = sorted(
        (p for p in profiles.values() if p.entry != data.user_entry),
        key=lambda p: -p.threat_weight,
    )[:n_rivals]
    for p in threats:
        print(
            f"\n#{p.rank} {p.entry_name} ({p.manager})  {p.total} pts  "
            f"+{p.gap_to_user} on you  threat weight {100 * p.threat_weight:.1f}%"
        )
        print(
            f"   TV {p.team_value:.1f}  bank {p.bank:.1f}  style {p.style} "
            f"(tmpl {p.template_score:.1f}, z {p.template_z:+.2f})  "
            f"GW scores {list(p.gw_points.values())}  bench pts {p.points_on_bench}"
        )
        print(
            f"   chips used: {', '.join(f'{n}@GW{g}' for n, g in p.chips_used) or 'none'}   "
            f"chips left (H1+H2): {', '.join(f'{k}x{v}' for k, v in sorted(p.chips_left.items()))}"
        )
        tl = transfer_log(data, p.entry)
        print(f"   transfers: {'; '.join(tl) if tl else 'none'}")
        print(f"   overlap with your squad: {p.overlap_with_user}/15")
        for line in squad_lines(data, p):
            print(f"     {line}")


def block_user_squad(data: LeagueData, weights: dict[int, float]) -> None:
    profiles = build_profiles(data, weights=weights)
    p = profiles[data.user_entry]
    print(header("YOUR SQUAD"))
    print(
        f"{p.entry_name} ({p.manager})  rank {p.rank}  {p.total} pts  TV {p.team_value:.1f}  "
        f"bank {p.bank:.1f}"
    )
    print(
        f"template score {p.template_score:.1f} (league mean "
        f"{statistics.mean(x.template_score for x in profiles.values()):.1f}), "
        f"z {p.template_z:+.2f} -> {p.style}"
    )
    print(f"GW scores {list(p.gw_points.values())}  points left on bench {p.points_on_bench}")
    print(f"chips used: {', '.join(f'{n}@GW{g}' for n, g in p.chips_used) or 'none'}")
    tl = transfer_log(data, data.user_entry)
    print(f"transfers: {'; '.join(tl) if tl else 'none'}")
    own = {r.element: r for r in ownership_table(data, weights=weights)}
    print(f"\n{'':6}{'pos':<4}{'player':<20s}{'team':<5}{'£':>6}{'pts':>5}{'own%':>8}{'EO%':>8}")
    payload = data.picks[data.user_entry][max(data.gameweeks)]
    order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    rows = []
    for pk in payload["picks"]:
        pl = data.players[pk["element"]]
        r = own.get(pk["element"])
        rows.append(
            (
                (pk["multiplier"] < 1, order[pl.position], -data.season_points(pl.id), pl.id),
                pl,
                pk,
                r,
            )
        )
    rows.sort(key=lambda t: t[0])
    for (benched, _, _, _), pl, pk, r in rows:
        tag = "(C)" if pk["is_captain"] else ("(V)" if pk["is_vice_captain"] else "")
        print(
            f"{'BENCH ' if benched else '      '}{pl.position:<4}{pl.web_name + tag:<20s}"
            f"{pl.team_short:<5}{pl.price:>6.1f}{data.season_points(pl.id):>5}"
            f"{r.owned_pct if r else 0:>8.1f}{r.eo_pct if r else 0:>8.1f}"
        )


def block_takeaways(data: LeagueData, weights: dict[int, float], remaining: int) -> None:
    print(header("WHAT THE NUMBERS SAY"))
    cp = catchup_plan(data, remaining)
    up = user_position(data, weights=weights)
    profiles = build_profiles(data, weights=weights)
    me = profiles[data.user_entry]
    rows = overlap_table(data, remaining_gws=remaining)
    top10 = [r for r in rows if r.rank <= 10]
    own_rows = ownership_table(data, weights=weights)
    core = [r for r in own_rows if r.owned_pct >= 40.0]
    core_missing = [r for r in core if not r.user_owns]

    lines = [
        f"1. The deficit is {cp.targets[0].gap} points with {remaining} GWs left: "
        f"{cp.required_edge_vs_leader:.2f} pts/GW of edge over the current leader to win, "
        f"{cp.required_edge_vs_top5:.2f} for top-5, {cp.required_edge_vs_top10:.2f} for top-10.",
        f"2. You are scoring {cp.user_ppg:.1f} pts/GW against a league mean of "
        f"{cp.league_ppg:.1f} and a leader rate of {cp.leader_ppg:.1f}. Nobody holds "
        f"{cp.leader_ppg:.0f} pts/GW for a season, so assume the field regresses toward the "
        f"league mean: you then need roughly {cp.league_ppg + cp.required_edge_vs_leader:.0f} "
        f"pts/GW to win, {cp.league_ppg + cp.required_edge_vs_top10:.0f} for top-10. That is "
        f"+{cp.league_ppg + cp.required_edge_vs_leader - cp.user_ppg:.0f} and "
        f"+{cp.league_ppg + cp.required_edge_vs_top10 - cp.user_ppg:.0f} on your current rate. "
        f"Two gameweeks is a tiny sample and your {cp.user_ppg:.0f} pts/GW is not a fixed "
        f"property, but the arithmetic is the arithmetic.",
        f"3. Your squad's template score is {me.template_score:.1f} vs a league mean of "
        f"{statistics.mean(p.template_score for p in profiles.values()):.1f} "
        f"(z {me.template_z:+.2f}). You are already the most differential-heavy end of this "
        f"league, and it has produced last place. Differentials are a tool for closing a gap "
        f"you can see, not a default posture.",
        f"4. There are {len(core)} players owned by 40%+ of this league; you own "
        f"{len(core) - len(core_missing)}. The {len(core_missing)} you are missing "
        f"({', '.join(r.name for r in core_missing)}) have cost you "
        f"{sum(s.swing for s in up.liabilities if s.name in {r.name for r in core_missing}):+.0f} "
        f"points already.",
        f"5. Your differentials have returned {up.total_differential_swing:+.0f} points against "
        f"the field; your missing template players have cost {up.total_liability_swing:+.0f}. "
        f"Net {up.net_swing:+.0f}.",
    ]
    if top10:
        lines.append(
            f"6. Against the top 10 your squads share only "
            f"{statistics.mean(r.shared_squad for r in top10):.1f} of 15 players and "
            f"{statistics.mean(r.dead_weight_pct for r in top10):.0f}% of your multiplier mass is "
            f"dead weight. Low overlap cuts both ways: it is exactly why you can still catch "
            f"them and exactly why you fell behind."
        )
    bench_me = me.points_on_bench
    bench_league = statistics.mean(p.points_on_bench for p in profiles.values())
    lines.append(
        f"7. You have left {bench_me} points on your bench across {len(data.gameweeks)} GWs "
        f"against a league mean of {bench_league:.0f}. Bench management is not where this "
        f"deficit came from; the starting XI is."
    )
    caps = captaincy_table(data, max(data.gameweeks))
    if caps:
        top_cap = caps[0]
        best_cap = max(caps, key=lambda c: c[3])
        lines.append(
            f"8. Captaincy in GW{max(data.gameweeks)}: the modal pick was {top_cap[0]} "
            f"({top_cap[2]:.0f}% of the league, {top_cap[3]} pts) but {best_cap[0]} returned "
            f"{best_cap[3]}. Armband choice alone moved managers by "
            f"{best_cap[3] - top_cap[3]:+d} pts that week."
        )
    pend = [f for gw in data.gameweeks for f in data.pending_fixtures(gw)]
    if pend:
        lines.append(
            f"9. {len(pend)} fixture(s) from the current gameweek are still unplayed, so every "
            f"total above is provisional."
        )
    for line in lines:
        print("\n" + line)


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Mini-league intelligence report")
    ap.add_argument("--league", type=int, default=DEFAULT_LEAGUE_ID)
    ap.add_argument("--entry", type=int, default=DEFAULT_ENTRY_ID)
    ap.add_argument("--top", type=int, default=20, help="rows in the ownership table")
    ap.add_argument("--rivals", type=int, default=6, help="deep-dive rival profiles")
    ap.add_argument("--remaining", type=int, default=36, help="gameweeks remaining")
    ap.add_argument("--overlap-rows", type=int, default=15)
    ap.add_argument("--offline", action="store_true", help="cache only, no network")
    ap.add_argument("--scheme", default="contention", help="threat weighting scheme")
    args = ap.parse_args(argv)

    client = FplClient(offline=args.offline)
    data = load_league(
        league_id=args.league, user_entry=args.entry, client=client, verbose=True
    )
    if not any(r["entry"] == args.entry for r in data.standings):
        print(f"entry {args.entry} is not in league {args.league}", file=sys.stderr)
        return 1

    weights = threat_weights(data, scheme=args.scheme)

    block_overview(data, args.remaining)
    block_verification(data)
    block_ownership(data, weights, args.top)
    block_user_squad(data, weights)
    block_user_position(data, weights)
    block_targets(data, weights, top_n=10)
    block_gap(data, args.remaining)
    block_overlap(data, args.remaining, args.overlap_rows)
    block_rivals(data, weights, args.rivals)
    block_takeaways(data, weights, args.remaining)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
