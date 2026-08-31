"""Playing-time model: probability of playing, probability of starting, and
expected minutes, per player per gameweek.

Design
------
FPL minutes are close to bimodal (a fit outfield player either plays ~60-90
or 0, with a smaller mass of late sub cameos), and this early in a season
there are only 1-2 real gameweeks of current-season signal per player. A
single population-average prior (e.g. "the average rostered player starts
41% of gameweeks") would be the wrong shrinkage target: that average is
dragged down by a large population of players who essentially never feature
at all, so blending it against a genuine 2-for-2 start record would (and, in
an earlier version of this module, did) pull an obvious first-team regular's
estimate down toward a mediocre 55 expected minutes.

The fix is to condition the prior on the same thing we are about to condition
the *player* on: whether they appeared at all in the early part of the season.
:func:`fit_position_priors` fits three separate priors per position from
history -- for players who **started**, **only came on as a sub**, or were
**unused** in the season's first ``early_gws`` gameweeks -- measured against
that player's own subsequent playing time for the rest of the season. This
is deliberately the same shape as our actual deployment: predicting from GW3
onward having observed GW1-2. A team's specific rotation pattern beyond what
the player's own recent appearances already encode is not modelled further;
that is a scope cut given how little current-season signal exists this early
(flagged in the project report).

On top of the shrunk playing-time estimate, the API's own availability signal
(``status`` and ``chance_of_playing_next_round``) is applied as a hard
override multiplier. ``status`` codes, per bootstrap-static: ``a`` available,
``d`` doubtful, ``i`` injured, ``s`` suspended, ``u`` unavailable (left the
club / out on loan elsewhere), ``n`` not eligible (e.g. cup-tied, ineligible
to face a parent club). ``u``/``n`` are a hard zero; ``i``/``s``/``d`` are
scaled by the reported ``chance_of_playing_next_round`` when the API gives
one, and by a conservative default otherwise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

__all__ = [
    "HARD_OUT_STATUSES",
    "UNCERTAIN_STATUSES",
    "fit_position_priors",
    "availability_multiplier",
    "appearance_bucket",
    "estimate_minutes",
]

log = logging.getLogger(__name__)

#: `status` codes that mean "will not play, full stop" regardless of any
#: chance-of-playing percentage the API also reports.
HARD_OUT_STATUSES = frozenset({"u", "n"})
#: `status` codes where a chance-of-playing percentage (if given) should scale
#: the estimate, rather than overriding it outright.
UNCERTAIN_STATUSES = frozenset({"i", "s", "d"})

#: Default scaling applied when the status is uncertain but the API has not
#: (yet) published a chance-of-playing percentage for the next round.
_DEFAULT_DOUBT_MULTIPLIER = {"d": 0.75, "i": 0.10, "s": 0.10}

_MIN_SEASONS_FOR_PRIORS = ("2022-23", "2023-24", "2024-25", "2025-26")
_POSITION_CODE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

_BUCKETS = ("started", "subbed", "unused")

# Position/bucket fallback, used only if `fit_position_priors` cannot be run
# (e.g. a unit test with a tiny synthetic history frame). Numbers are
# deliberately conservative for "unused" and generous for "started".
_FALLBACK_PRIORS: dict[tuple[int, str], dict[str, float]] = {}
for _pos, (_started, _subbed, _unused) in {
    1: (0.90, 0.30, 0.03),
    2: (0.82, 0.25, 0.06),
    3: (0.75, 0.30, 0.08),
    4: (0.72, 0.32, 0.09),
}.items():
    _FALLBACK_PRIORS[(_pos, "started")] = {
        "p_play": _started, "p_start": _started * 0.92, "minutes_started": 84.0, "minutes_sub": 24.0,
    }
    _FALLBACK_PRIORS[(_pos, "subbed")] = {
        "p_play": _subbed, "p_start": _subbed * 0.35, "minutes_started": 75.0, "minutes_sub": 20.0,
    }
    _FALLBACK_PRIORS[(_pos, "unused")] = {
        "p_play": _unused, "p_start": _unused * 0.5, "minutes_started": 70.0, "minutes_sub": 18.0,
    }


def appearance_bucket(games_played: float, games_started: float) -> str:
    """Classify a player's early-season involvement into one of three regimes."""
    if games_started > 0:
        return "started"
    if games_played > 0:
        return "subbed"
    return "unused"


def fit_position_priors(
    history_df: pd.DataFrame,
    seasons: tuple[str, ...] = _MIN_SEASONS_FOR_PRIORS,
    early_gws: int = 2,
) -> pd.DataFrame:
    """Position x early-involvement-bucket priors for the rest of a season.

    For each season, every player is classified by :func:`appearance_bucket`
    using only their first ``early_gws`` gameweeks, then their ``p_play``,
    ``p_start``, ``minutes_started`` and ``minutes_sub`` are measured over
    *the remaining* gameweeks of that same season. This mirrors exactly how
    the prior is used at prediction time (bucket a player on GW1-2, predict
    GW3 onward), so the shrinkage target already has the right shape instead
    of being a single population average that fits nobody well.

    Restricted to seasons that publish ``starts`` (2022-23 onward), since
    ``started`` vs ``subbed`` cannot be told apart without it.
    """
    frame = history_df[history_df["season"].isin(seasons)].copy()
    frame = frame.dropna(subset=["position", "minutes", "gw"])
    frame["position_code"] = frame["position"].map(_POSITION_CODE)
    frame = frame.dropna(subset=["position_code"])
    frame["played"] = frame["minutes"] > 0
    frame["started"] = frame["starts"].fillna(0) > 0

    rows = []
    for (season, element), g in frame.groupby(["season", "element"]):
        early = g[g["gw"] <= early_gws]
        later = g[g["gw"] > early_gws]
        if early.empty or later.empty:
            continue
        bucket = appearance_bucket(early["played"].sum(), early["started"].sum())
        pos = int(early["position_code"].iloc[0])
        started_later = later[later["started"]]
        subbed_later = later[later["played"] & ~later["started"]]
        rows.append(
            {
                "position": pos,
                "bucket": bucket,
                "n_player_seasons": 1,
                "n_later_gws": len(later),
                "played_sum": int(later["played"].sum()),
                "started_sum": int(later["started"].sum()),
                "minutes_started_sum": float(started_later["minutes"].sum()),
                "n_started_rows": len(started_later),
                "minutes_sub_sum": float(subbed_later["minutes"].sum()),
                "n_sub_rows": len(subbed_later),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return _fallback_prior_table()

    agg = detail.groupby(["position", "bucket"]).sum(numeric_only=True).reset_index()
    agg["p_play"] = agg["played_sum"] / agg["n_later_gws"]
    agg["p_start"] = agg["started_sum"] / agg["n_later_gws"]
    agg["minutes_started"] = np.where(
        agg["n_started_rows"] > 0, agg["minutes_started_sum"] / agg["n_started_rows"], 80.0
    )
    agg["minutes_sub"] = np.where(
        agg["n_sub_rows"] > 0, agg["minutes_sub_sum"] / agg["n_sub_rows"], 22.0
    )
    out = agg[["position", "bucket", "p_play", "p_start", "minutes_started", "minutes_sub", "n_player_seasons"]]

    # Backfill any (position, bucket) combination with too little history
    # (e.g. very few keepers ever fall in "subbed") from the fallback table.
    fallback = _fallback_prior_table()
    have = set(zip(out["position"], out["bucket"]))
    missing = fallback[~fallback.apply(lambda r: (r["position"], r["bucket"]) in have, axis=1)]
    if not missing.empty:
        out = pd.concat([out, missing], ignore_index=True)
    return out.sort_values(["position", "bucket"]).reset_index(drop=True)


def _fallback_prior_table() -> pd.DataFrame:
    rows = [{"position": p, "bucket": b, "n_player_seasons": 0, **v} for (p, b), v in _FALLBACK_PRIORS.items()]
    return pd.DataFrame(rows)


def availability_multiplier(status: str | None, chance_next: float | None) -> float:
    """Map an FPL ``status`` (+ optional chance-of-playing %) to a [0, 1] scalar.

    This is a hard override applied on top of the data-driven playing-time
    estimate, not a replacement for it: a nailed starter marked 75% (returning
    from a knock) still uses their own start-rate as the base, just scaled
    down by 0.75.
    """
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return 1.0
    status = str(status)
    if status in HARD_OUT_STATUSES:
        return 0.0
    if status in UNCERTAIN_STATUSES:
        if chance_next is not None and not (isinstance(chance_next, float) and np.isnan(chance_next)):
            return float(np.clip(chance_next, 0.0, 100.0) / 100.0)
        return _DEFAULT_DOUBT_MULTIPLIER.get(status, 0.5)
    return 1.0  # status == "a", or an unrecognised future code: assume available.


def estimate_minutes(
    elements: pd.DataFrame,
    gw_history: pd.DataFrame,
    n_gws_so_far: int,
    *,
    priors: pd.DataFrame | None = None,
    start_pseudo_n: float = 2.0,
    play_pseudo_n: float = 2.0,
    minutes_pseudo_n: float = 2.0,
) -> pd.DataFrame:
    """Estimate p_play, p_start and expected minutes for every player.

    Parameters
    ----------
    elements:
        One row per player, as ``fplopt.data.fpl_api.elements_frame`` returns:
        needs ``id``, ``element_type``, ``team``, ``now_cost``, ``status``,
        ``chance_of_playing_next_round``.
    gw_history:
        Long frame, one row per (player, gameweek so far) with ``element``,
        ``minutes``, ``starts`` -- e.g. the concatenation of
        ``fplopt.data.fpl_api.live_points_frame(payload, gw)`` across every
        gameweek played so far this season (data-checked or not; the live
        endpoint's per-fixture ``starts``/``minutes`` are reliable as soon as
        a match kicks off).
    n_gws_so_far:
        How many gameweeks' worth of deadlines have already passed. Also used
        as the ``early_gws`` window when fitting ``priors`` if the caller
        didn't already fit them at a fixed window; a player with zero
        appearances across all ``n_gws_so_far`` gameweeks is exactly the
        "unused" prior bucket.
    priors:
        Output of :func:`fit_position_priors`, bucketed by
        (position, "started"/"subbed"/"unused"). Falls back to a static table
        when omitted (mainly for tests).

    Returns
    -------
    DataFrame with one row per player: ``player_id``, ``position``, ``team``,
    ``price``, ``status``, ``p_play``, ``p_start``, ``p_sub``,
    ``exp_minutes``, ``minutes_if_started``, ``minutes_if_sub``.
    """
    if priors is None:
        priors = _fallback_prior_table()
    prior_idx = priors.set_index(["position", "bucket"])

    gh = gw_history.copy()
    if gh.empty:
        gh = pd.DataFrame(columns=["element", "minutes", "starts"])
    gh["played"] = gh["minutes"].fillna(0) > 0
    gh["started"] = gh["starts"].fillna(0) > 0

    games_played = gh.groupby("element")["played"].sum().rename("games_played")
    games_started = gh.groupby("element")["started"].sum().rename("games_started")
    started_minutes = gh[gh["started"]].groupby("element")["minutes"].agg(["mean", "count"])
    started_minutes.columns = ["avg_minutes_started", "n_started"]
    sub_minutes = gh[gh["played"] & ~gh["started"]].groupby("element")["minutes"].agg(["mean", "count"])
    sub_minutes.columns = ["avg_minutes_sub", "n_sub"]

    by_player = pd.DataFrame(index=elements["id"].astype(int))
    by_player.index.name = "element"
    by_player = by_player.join(games_played).join(games_started).join(started_minutes).join(sub_minutes)
    by_player = by_player.fillna(0.0)

    out = elements[
        ["id", "element_type", "team", "now_cost", "status", "chance_of_playing_next_round"]
    ].copy()
    out = out.rename(columns={"id": "player_id", "element_type": "position", "now_cost": "price"})
    out = out.merge(by_player, left_on="player_id", right_index=True, how="left").fillna(0.0)
    out["position"] = out["position"].astype(int)
    out["bucket"] = [
        appearance_bucket(gp, gs) for gp, gs in zip(out["games_played"], out["games_started"])
    ]

    prior_cols = ["p_play", "p_start", "minutes_started", "minutes_sub"]
    prior_vals = prior_idx.reindex(zip(out["position"], out["bucket"]))[prior_cols].reset_index(drop=True)
    out = pd.concat([out.reset_index(drop=True), prior_vals.add_suffix("_prior")], axis=1)

    n = max(int(n_gws_so_far), 0)
    if n > 0:
        p_play_raw = (out["games_played"] + play_pseudo_n * out["p_play_prior"]) / (n + play_pseudo_n)
        p_start_raw = (out["games_started"] + start_pseudo_n * out["p_start_prior"]) / (n + start_pseudo_n)
    else:
        p_play_raw = out["p_play_prior"]
        p_start_raw = out["p_start_prior"]
    p_start_raw = np.minimum(p_start_raw, p_play_raw)

    minutes_started_est = (
        out["n_started"] * out["avg_minutes_started"] + minutes_pseudo_n * out["minutes_started_prior"]
    ) / (out["n_started"] + minutes_pseudo_n)
    minutes_sub_est = (
        out["n_sub"] * out["avg_minutes_sub"] + minutes_pseudo_n * out["minutes_sub_prior"]
    ) / (out["n_sub"] + minutes_pseudo_n)

    mult = np.array(
        [
            availability_multiplier(s, c)
            for s, c in zip(out["status"], out["chance_of_playing_next_round"])
        ]
    )

    p_play = np.clip(p_play_raw.to_numpy() * mult, 0.0, 1.0)
    p_start = np.clip(p_start_raw.to_numpy() * mult, 0.0, None)
    p_start = np.minimum(p_start, p_play)
    p_sub = np.clip(p_play - p_start, 0.0, None)
    exp_minutes = p_start * minutes_started_est.to_numpy() + p_sub * minutes_sub_est.to_numpy()

    result = pd.DataFrame(
        {
            "player_id": out["player_id"].astype(int),
            "position": out["position"],
            "team": out["team"].astype(int),
            "price": out["price"].astype(int),
            "status": out["status"],
            "p_play": p_play,
            "p_start": p_start,
            "p_sub": p_sub,
            "exp_minutes": exp_minutes,
            "minutes_if_started": minutes_started_est.to_numpy(),
            "minutes_if_sub": minutes_sub_est.to_numpy(),
        }
    )
    return result
