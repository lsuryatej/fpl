"""Solve the transfer plan and write the decision the dashboard reads.

Compares three courses of action for the upcoming gameweek -- keep the squad
and use the available free transfers, or play a wildcard -- and reports the
expected-points difference between them rather than asserting one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fplopt.data.fpl_api import FPLClient
from fplopt.data.paths import data_dir
from fplopt.league import fetch as lfetch
from fplopt.league import ownership as lown
from fplopt.optim.multiweek import greedy_baseline, solve_multiweek
from fplopt.optim.objective import MiniLeagueContext, rank_utility, recommend_lambda
from fplopt.optim.state import load_state_from_api

LEAGUE_ID = 490294

ENTRY_ID = 3539707


def describe(plan, names: dict[int, str]) -> list[dict]:
    rows = []
    for gw_plan in plan.gws:
        rows.append(
            {
                "gw": gw_plan.gw,
                "chip": gw_plan.chip,
                "in": [names.get(i, str(i)) for i in gw_plan.transfers_in],
                "out": [names.get(i, str(i)) for i in gw_plan.transfers_out],
                "captain": names.get(gw_plan.captain_id, ""),
                "vice": names.get(gw_plan.vice_captain_id, ""),
                "hits": gw_plan.hits,
                "hit_cost": gw_plan.hit_cost,
                "ft_before": gw_plan.free_transfers_before,
                "ft_used": gw_plan.free_transfers_used,
                "bank_after": gw_plan.bank_after / 10,
                "points": round(gw_plan.raw_points, 2),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--projections", default=None)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument(
        "--league-aware", action="store_true",
        help="optimise mini-league rank utility instead of raw expected points",
    )
    parser.add_argument(
        "--lam", type=float, default=0.20,
        help="risk lambda. 0.20 sits at the efficient point of the EP-vs-ownership "
             "frontier; the auto-recommended 0.348 costs 16.5 EP for little extra edge",
    )
    args = parser.parse_args()

    gws = [int(g) for g in args.horizon.split(",")]
    path = Path(args.projections) if args.projections else (
        data_dir() / "projections" / f"gw{gws[0]}-{gws[-1]}.parquet"
    )
    projections = pd.read_parquet(path)

    client = FPLClient()
    boot = client.bootstrap()
    names = {e["id"]: e["web_name"] for e in boot["elements"]}

    value_col = "ep"
    if args.league_aware:
        data = lfetch.load_league(league_id=LEAGUE_ID, user_entry=ENTRY_ID, verbose=False)
        rows = lown.ownership_table(data)
        # EO is a percentage in the league module; the objective wants a fraction.
        eo = {r.element: r.eo_pct / 100.0 for r in rows}
        standings = {r["entry"]: r for r in data.standings}
        me = standings[ENTRY_ID]
        leader = max(r["total"] for r in data.standings)
        context = MiniLeagueContext(
            rank=me["rank"],
            n_managers=len(data.standings),
            points_behind_leader=leader - me["total"],
            gws_remaining=len(gws) and (38 - gws[0] + 1),
        )
        lam = args.lam if args.lam is not None else recommend_lambda(context)
        _auto = recommend_lambda(context)
        if abs(lam - _auto) > 1e-9:
            print(f'(auto-recommended lambda was {_auto:.3f}; using {lam:.3f})')
        projections = rank_utility(projections, eo, lam)
        value_col = "objective_score"
        print(f"league-aware objective: rank {context.rank}/{context.n_managers}, "
              f"{context.points_behind_leader} behind, lambda={lam:.3f}")
        print(f"players with league EO on record: {len(eo)}\n")

    state = load_state_from_api(ENTRY_ID, client)
    print(f"state: bank {state.bank/10:.1f}m, free transfers {state.free_transfers}")
    print(f"chips remaining: {state.chips_remaining}")
    print(f"squad value: {sum(h.current_price for h in state.squad)/10:.1f}m\n")

    def ep_total(plan, proj: pd.DataFrame) -> float:
        """A plan's true expected points, whatever objective produced it.

        raw_points is denominated in value_col, so under the league-aware
        objective it is a rank-utility score, NOT points. Reporting it as
        points would overstate the plan by a wide margin.
        """
        ep_by = proj.set_index(["gw", "player_id"])["ep"]
        total = 0.0
        for g in plan.gws:
            for pid in g.starting_ids:
                total += float(ep_by.get((g.gw, pid), 0.0))
            total += float(ep_by.get((g.gw, g.captain_id), 0.0))
            total -= g.hit_cost
        return total

    def raw_total(plan) -> float:
        """Undiscounted expected points.

        solve_multiweek reports a DISCOUNTED objective while greedy_baseline
        reports an undiscounted one, so objective_value is not comparable
        across the two. Summing raw_points puts every option on one scale.
        """
        return sum(g.raw_points for g in plan.gws)

    results = {}
    for label, chip_plan in (("no chip", None), ("wildcard now", {gws[0]: "wildcard"})):
        plan = solve_multiweek(
            state, projections, gws, value_col=value_col,
            discount=args.discount, chip_plan=chip_plan, time_limit=args.time_limit,
        )
        results[label] = plan
        print(f"{label:14} status={plan.status} "
              f"true_EP={ep_total(plan, projections):.2f} "
              f"objective={raw_total(plan):.2f} "
              f"({plan.solver_seconds:.1f}s, pool {plan.pool_size})")

    greedy = greedy_baseline(state, projections, gws, value_col=value_col)
    print(f"{'greedy 1-week':14} true_EP={ep_total(greedy, projections):.2f} "
          f"objective={raw_total(greedy):.2f}")

    best_label = max(results, key=lambda k: raw_total(results[k]))
    best = results[best_label]
    best_ep = ep_total(best, projections)
    # Report the gap in POINTS, not objective units. Under the league-aware
    # objective those differ by a wide margin and calling a rank-utility delta
    # "expected points" overstates the case for whichever option won.
    delta = best_ep - min(ep_total(r, projections) for r in results.values())

    out = {
        "headline": (
            f"{best_label}: {best_ep:.1f} expected points over GW{gws[0]}-{gws[-1]}"
            + (" (chosen on league rank utility)" if args.league_aware else "")
        ),
        "summary": [
            f"Best course of action is '{best_label}', worth {delta:.1f} more expected "
            f"points over the {len(gws)}-gameweek horizon than the alternative.",
            f"Multi-week planning beats one-week-greedy by "
            f"{best_ep - ep_total(greedy, projections):.1f} expected points.",
        ],
        "options": {
            k: {"true_ep": round(ep_total(v, projections), 2),
                "objective": round(raw_total(v), 2)}
            for k, v in results.items()
        },
        "greedy_baseline": round(ep_total(greedy, projections), 2),
        "true_ep": round(best_ep, 2),
        "plan": describe(best, names),
        "state": {
            "bank": state.bank / 10,
            "free_transfers": state.free_transfers,
            "chips": state.chips_remaining,
        },
    }
    out["objective"] = "league rank utility" if args.league_aware else "expected points"
    target = data_dir() / "decisions"
    target.mkdir(parents=True, exist_ok=True)
    (target / "latest.json").write_text(json.dumps(out, indent=2))

    print(f"\nBEST: {out['headline']}")
    for row in out["plan"][:4]:
        print(f"  GW{row['gw']}: OUT {row['out']} IN {row['in']} "
              f"| C {row['captain']} | hits {row['hits']} | pts {row['points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
