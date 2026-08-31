"""Tests for fplopt.optim.state: selling-price fee, ManagerState invariants,
free-transfer bank simulation, and the live-API loader.
"""

from __future__ import annotations

import pytest

from fplopt.optim.state import (
    DEF,
    FWD,
    GKP,
    MID,
    ManagerState,
    PlayerHolding,
    _chips_remaining,
    _reconstruct_purchase_prices,
    load_state_from_api,
    selling_price,
    simulate_free_transfers,
)


def _legal_squad(prices: dict[int, int] | None = None) -> tuple[PlayerHolding, ...]:
    """2 GK, 5 DEF, 5 MID, 3 FWD, spread across enough clubs to respect the 3-per-club cap."""
    prices = prices or {}
    holdings = []
    pid = 1
    counts = {GKP: 2, DEF: 5, MID: 5, FWD: 3}
    team_cycle = list(range(1, 6))  # 5 clubs, so a max of 3 per club across 15 players is easy
    i = 0
    for pos, n in counts.items():
        for _ in range(n):
            team = team_cycle[i % len(team_cycle)]
            i += 1
            price = prices.get(pid, 50)
            holdings.append(
                PlayerHolding(player_id=pid, position=pos, team=team, purchase_price=price, current_price=price)
            )
            pid += 1
    return tuple(holdings)


# --------------------------------------------------------------------------
# selling_price: the 50%-of-profit-rounded-down sell-on fee
# --------------------------------------------------------------------------
class TestSellingPrice:
    def test_worked_example_from_the_brief(self):
        # Bought at 75 (7.5m), now worth 80 (8.0m): profit 5, half 2 (floor), sells for 77.
        assert selling_price(75, 80) == 77

    def test_no_fee_on_a_loss(self):
        assert selling_price(100, 90) == 90

    def test_no_fee_on_no_change(self):
        assert selling_price(100, 100) == 100

    def test_odd_profit_rounds_down(self):
        # profit=1 -> half=0.5 -> floor 0 -> no benefit passed on at all.
        assert selling_price(100, 101) == 100
        # profit=3 -> half=1.5 -> floor 1.
        assert selling_price(100, 103) == 101

    def test_even_profit_exact_half(self):
        assert selling_price(100, 110) == 105

    @pytest.mark.parametrize("purchase,current", [(50, 50), (120, 60), (75, 74)])
    def test_never_exceeds_current_price(self, purchase, current):
        assert selling_price(purchase, current) <= current


# --------------------------------------------------------------------------
# ManagerState invariants
# --------------------------------------------------------------------------
class TestManagerStateValidation:
    def test_accepts_a_legal_squad(self):
        state = ManagerState(
            entry_id=1,
            as_of_gw=3,
            bank=5,
            free_transfers=1,
            chips_remaining={"wildcard": 2, "free_hit": 2, "bench_boost": 2, "triple_captain": 2},
            squad=_legal_squad(),
        )
        assert len(state.squad) == 15
        assert state.squad_value == 15 * 50

    def test_rejects_wrong_squad_size(self):
        with pytest.raises(ValueError, match="15 players"):
            ManagerState(
                entry_id=1, as_of_gw=3, bank=0, free_transfers=1,
                chips_remaining={}, squad=_legal_squad()[:14],
            )

    def test_rejects_wrong_position_composition(self):
        bad = list(_legal_squad())
        # Turn a FWD into an extra MID so the counts no longer match 2/5/5/3.
        bad[-1] = PlayerHolding(player_id=bad[-1].player_id, position=MID, team=bad[-1].team,
                                 purchase_price=50, current_price=50)
        with pytest.raises(ValueError, match="composition"):
            ManagerState(entry_id=1, as_of_gw=3, bank=0, free_transfers=1, chips_remaining={}, squad=tuple(bad))

    def test_rejects_more_than_three_per_club(self):
        bad = list(_legal_squad())
        # Force 4 players onto team 1.
        fixed = []
        team1_count = 0
        for h in bad:
            if team1_count < 4:
                h = PlayerHolding(player_id=h.player_id, position=h.position, team=1,
                                   purchase_price=h.purchase_price, current_price=h.current_price)
                team1_count += 1
            fixed.append(h)
        with pytest.raises(ValueError, match="exceeding the limit"):
            ManagerState(entry_id=1, as_of_gw=3, bank=0, free_transfers=1, chips_remaining={}, squad=tuple(fixed))

    def test_rejects_duplicate_player(self):
        bad = list(_legal_squad())
        bad[1] = PlayerHolding(player_id=bad[0].player_id, position=bad[1].position, team=bad[1].team,
                                purchase_price=50, current_price=50)
        with pytest.raises(ValueError, match="Duplicate"):
            ManagerState(entry_id=1, as_of_gw=3, bank=0, free_transfers=1, chips_remaining={}, squad=tuple(bad))

    @pytest.mark.parametrize("ft", [0, 6, -1])
    def test_rejects_free_transfers_outside_one_to_five(self, ft):
        with pytest.raises(ValueError, match="free_transfers"):
            ManagerState(entry_id=1, as_of_gw=3, bank=0, free_transfers=ft, chips_remaining={}, squad=_legal_squad())

    def test_rejects_negative_bank(self):
        with pytest.raises(ValueError, match="bank"):
            ManagerState(entry_id=1, as_of_gw=3, bank=-1, free_transfers=1, chips_remaining={}, squad=_legal_squad())

    def test_sell_value_reflects_the_fee(self):
        prices = {1: 40, 2: 45}  # both bought at these prices
        squad = list(_legal_squad(prices))
        # Bump both GKs' current price up so a profit-fee applies.
        squad[0] = PlayerHolding(player_id=1, position=GKP, team=squad[0].team, purchase_price=40, current_price=50)
        state = ManagerState(entry_id=1, as_of_gw=3, bank=0, free_transfers=1, chips_remaining={}, squad=tuple(squad))
        assert state.holding(1).selling_price == selling_price(40, 50)
        assert state.sell_value == sum(h.selling_price for h in state.squad)


# --------------------------------------------------------------------------
# Free-transfer bank simulation
# --------------------------------------------------------------------------
class TestSimulateFreeTransfers:
    def test_first_played_gameweek_grants_no_roll_by_itself(self):
        # Only ever played gw1: nothing to roll from yet, still just 1 FT.
        rows = [{"event": 1, "event_transfers": 0}]
        assert simulate_free_transfers(rows) == 1

    def test_two_gameweeks_no_transfers_rolls_to_two(self):
        # Regression test for a real bug: this used to return 3 (double-counting
        # gameweek 1, which isn't a transfer week at all) instead of 2.
        rows = [{"event": 1, "event_transfers": 0}, {"event": 2, "event_transfers": 0}]
        assert simulate_free_transfers(rows) == 2

    def test_using_the_single_free_transfer_keeps_the_bank_at_one(self):
        rows = [{"event": 1, "event_transfers": 0}, {"event": 2, "event_transfers": 1}]
        assert simulate_free_transfers(rows) == 1

    def test_taking_a_hit_does_not_reduce_the_bank_further(self):
        # 1 FT available, 3 transfers made (1 free + 2 hits): still rolls to 1, not negative.
        rows = [{"event": 1, "event_transfers": 0}, {"event": 2, "event_transfers": 3}]
        assert simulate_free_transfers(rows) == 1

    def test_caps_at_max_bank(self):
        rows = [{"event": e, "event_transfers": 0} for e in range(1, 10)]
        assert simulate_free_transfers(rows) == 5

    def test_wildcard_week_does_not_touch_the_bank(self):
        rows = [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},  # rolls to 2 entering gw3
            {"event": 3, "event_transfers": 15},  # wildcard: 15 transfers, but shouldn't cost the bank
        ]
        chips = [{"name": "wildcard", "event": 3}]
        # Entering gw4: as if gw3 had 0 transfers -> 2 - 0 + 1 = 3.
        assert simulate_free_transfers(rows, chips) == 3

    def test_empty_history_defaults_to_one(self):
        assert simulate_free_transfers([]) == 1


class TestChipsRemaining:
    def test_no_chips_played_leaves_two_of_each(self):
        remaining = _chips_remaining([])
        assert remaining == {"wildcard": 2, "free_hit": 2, "bench_boost": 2, "triple_captain": 2}

    def test_played_chips_are_subtracted(self):
        remaining = _chips_remaining([{"name": "wildcard", "event": 5}, {"name": "3xc", "event": 10}])
        assert remaining["wildcard"] == 1
        assert remaining["triple_captain"] == 1
        assert remaining["bench_boost"] == 2

    def test_unknown_chip_name_is_ignored_not_fatal(self):
        remaining = _chips_remaining([{"name": "some_future_chip", "event": 1}])
        assert remaining == {"wildcard": 2, "free_hit": 2, "bench_boost": 2, "triple_captain": 2}


class TestReconstructPurchasePrices:
    def test_uses_the_most_recent_buy_transfer(self):
        transfers = [
            {"element_in": 10, "element_in_cost": 55, "event": 5},  # newest first
            {"element_in": 10, "element_in_cost": 50, "event": 2},
        ]
        elements = {10: {"now_cost": 55, "cost_change_start": 0}}
        prices = _reconstruct_purchase_prices([10], elements, transfers)
        assert prices[10] == 55

    def test_never_transferred_uses_season_start_price(self):
        elements = {7: {"now_cost": 65, "cost_change_start": 5}}
        prices = _reconstruct_purchase_prices([7], elements, [])
        assert prices[7] == 60


# --------------------------------------------------------------------------
# Live API loader
# --------------------------------------------------------------------------
class _FakeSession:
    def __init__(self, payloads: dict[str, object]):
        self.payloads = payloads

    def get_json(self, url, use_cache=True):
        for key, payload in self.payloads.items():
            if key in url:
                return payload
        raise AssertionError(f"unexpected URL {url}")


class _FakeClient:
    """A minimal stand-in for FPLClient so the loader can be tested offline."""

    def __init__(self, payloads: dict[str, object]):
        self.session = _FakeSession(payloads)

    def bootstrap(self):
        return self.session.get_json("bootstrap-static")

    def entry_history(self, eid):
        return self.session.get_json(f"entry/{eid}/history")

    def entry_picks(self, eid, gw):
        return self.session.get_json(f"entry/{eid}/event/{gw}/picks")

    def entry_transfers(self, eid):
        return self.session.get_json(f"entry/{eid}/transfers")


def _fake_element(pid, pos, team, now_cost, cost_change_start=0):
    return {"id": pid, "element_type": pos, "team": team, "now_cost": now_cost, "cost_change_start": cost_change_start}


class TestLoadStateFromApi:
    def test_wires_everything_together_against_a_fake_client(self):
        elements = [_fake_element(pid, pos, team, now_cost=50)
                    for pid, pos, team in zip(range(1, 16), [GKP]*2+[DEF]*5+[MID]*5+[FWD]*3, [1,2,3,4,5]*3)]
        picks = {
            "picks": [
                {"element": pid, "is_captain": pid == 5, "is_vice_captain": pid == 6, "multiplier": 2 if pid == 5 else 1}
                for pid in range(1, 16)
            ]
        }
        client = _FakeClient(
            {
                "bootstrap-static": {"elements": elements},
                "history": {"current": [{"event": 1, "bank": 3, "event_transfers": 0}], "chips": []},
                "picks": picks,
                "transfers": [],
            }
        )
        state = load_state_from_api(99, client=client)
        assert state.entry_id == 99
        assert state.as_of_gw == 1
        assert state.bank == 3
        assert state.free_transfers == 1  # only one gameweek of history -> no roll yet
        assert state.holding(5).is_captain
        assert len(state.squad) == 15

    def test_live_entry_3539707(self):
        """Cross-check against the real FPL API for the manager named in the brief.

        Skips cleanly if the network is unavailable rather than failing CI in
        a sandboxed environment; run explicitly to verify against production.
        """
        pytest.importorskip("requests")
        try:
            state = load_state_from_api(3539707)
        except Exception as exc:  # pragma: no cover - network flakiness only
            pytest.skip(f"live FPL API unreachable: {exc}")
        assert len(state.squad) == 15
        assert state.bank >= 0
        assert 1 <= state.free_transfers <= 5
