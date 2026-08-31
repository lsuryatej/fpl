"""Tests for fplopt.optim.objective."""

from __future__ import annotations

import pandas as pd
import pytest

from fplopt.optim.objective import (
    MiniLeagueContext,
    plain_expected_points,
    recommend_lambda,
    rank_utility,
)


def _df():
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "ep": [5.0, 4.0, 3.0],
            "sd": [1.0, 3.0, 2.0],
        }
    )


def test_plain_expected_points_copies_ep():
    out = plain_expected_points(_df())
    assert list(out["objective_score"]) == [5.0, 4.0, 3.0]


def test_rank_utility_with_lambda_zero_matches_plain_ep():
    out = rank_utility(_df(), effective_ownership={}, lam=0.0)
    plain = plain_expected_points(_df())
    assert list(out["objective_score"]) == list(plain["objective_score"])


def test_rank_utility_rewards_low_ownership_high_variance():
    df = _df()
    # Player 2 has lower ep than player 1 but much higher sd and zero EO;
    # a large enough lambda should make it score higher than the template pick.
    eo = {1: 1.0, 2: 0.0, 3: 0.5}
    out = rank_utility(df, effective_ownership=eo, lam=2.0)
    scores = dict(zip(out["player_id"], out["objective_score"]))
    # Player 1: fully owned -> no variance bonus at all, score == ep == 5.0.
    assert scores[1] == pytest.approx(5.0)
    # Player 2: ep 4.0 + 2.0*(1-0)*3.0 = 10.0.
    assert scores[2] == pytest.approx(10.0)
    assert scores[2] > scores[1]


def test_rank_utility_missing_eo_defaults_to_full_differential_credit():
    df = _df()
    out = rank_utility(df, effective_ownership={1: 1.0}, lam=1.0, default_eo=0.0)
    scores = dict(zip(out["player_id"], out["objective_score"]))
    # Player 3 has no EO entry -> default_eo=0.0 -> full (1-0)*sd bonus.
    assert scores[3] == pytest.approx(3.0 + 1.0 * (1 - 0.0) * 2.0)


def test_rank_utility_rejects_negative_lambda():
    with pytest.raises(ValueError):
        rank_utility(_df(), effective_ownership={}, lam=-0.1)


def test_rank_utility_requires_sd_column():
    df = _df().drop(columns=["sd"])
    with pytest.raises(ValueError, match="missing required columns"):
        rank_utility(df, effective_ownership={}, lam=1.0)


class TestMiniLeagueContext:
    def test_urgency_is_points_per_remaining_gw(self):
        ctx = MiniLeagueContext(rank=45, n_managers=45, points_behind_leader=88, gws_remaining=36)
        assert ctx.urgency == pytest.approx(88 / 36)
        assert ctx.is_trailing

    def test_leader_is_not_trailing(self):
        ctx = MiniLeagueContext(rank=1, n_managers=45, points_behind_leader=0, gws_remaining=36)
        assert not ctx.is_trailing
        assert ctx.urgency == 0.0

    def test_recommend_lambda_matches_the_documented_worked_example(self):
        ctx = MiniLeagueContext(rank=45, n_managers=45, points_behind_leader=88, gws_remaining=36)
        lam = recommend_lambda(ctx)
        assert lam == pytest.approx(0.15 + 0.08 * (88 / 36), abs=1e-9)

    def test_recommend_lambda_caps_out_for_extreme_urgency(self):
        ctx = MiniLeagueContext(rank=45, n_managers=45, points_behind_leader=88, gws_remaining=3)
        lam = recommend_lambda(ctx, base=0.15)
        assert lam == pytest.approx(0.15 + 1.0)

    def test_recommend_lambda_is_base_when_not_trailing(self):
        ctx = MiniLeagueContext(rank=1, n_managers=45, points_behind_leader=0, gws_remaining=36)
        assert recommend_lambda(ctx, base=0.2) == 0.2
