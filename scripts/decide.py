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
from fplopt.optim.multiweek import greedy_baseline, solve_multiweek
from fplopt.optim.state import load_state_from_api

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
    args = parser.parse_args()

    gws = [int(g) for g in args.horizon.split(",")]
    path = Path(args.projections) if args.projections else (
        data_dir() / "projections" / f"gw{gws[0]}-{gws[-1]}.parquet"
    )
    projections = pd.read_parquet(path)

    client = FPLClient()
    boot = client.bootstrap()
    names = {e["id"]: e["web_name"] for e in boot["elements"]}

    state = load_state_from_api(ENTRY_ID, client)
    print(f"state: bank {state.bank/10:.1f}m, free transfers {state.free_transfers}")
    print(f"chips remaining: {state.chips_remaining}")
    print(f"squad value: {sum(h.current_price for h in state.squad)/10:.1f}m\n")

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
            state, projections, gws,
            discount=args.discount, chip_plan=chip_plan, time_limit=args.time_limit,
        )
        results[label] = plan
        print(f"{label:14} status={plan.status} raw={raw_total(plan):.2f} "
              f"discounted={plan.objective_value:.2f} "
              f"({plan.solver_seconds:.1f}s, pool {plan.pool_size})")

    greedy = greedy_baseline(state, projections, gws)
    print(f"{'greedy 1-week':14} raw={raw_total(greedy):.2f} "
          f"discounted={greedy.objective_value:.2f}")

    best_label = max(results, key=lambda k: raw_total(results[k]))
    best = results[best_label]
    delta = raw_total(best) - min(raw_total(r) for r in results.values())

    out = {
        "headline": f"{best_label}: {raw_total(best):.1f} expected points over GW{gws[0]}-{gws[-1]}",
        "summary": [
            f"Best course of action is '{best_label}', worth {delta:.1f} more expected "
            f"points over the {len(gws)}-gameweek horizon than the alternative.",
            f"Multi-week planning beats one-week-greedy by "
            f"{raw_total(best) - raw_total(greedy):.1f} points (undiscounted).",
        ],
        "options": {k: round(raw_total(v), 2) for k, v in results.items()},
        "greedy_baseline": round(raw_total(greedy), 2),
        "plan": describe(best, names),
        "state": {
            "bank": state.bank / 10,
            "free_transfers": state.free_transfers,
            "chips": state.chips_remaining,
        },
    }
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
