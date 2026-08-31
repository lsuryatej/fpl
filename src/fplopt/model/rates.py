"""Per-90 rate model: goals, assists, defensive actions and saves, by player.

Every rate is estimated with empirical-Bayes shrinkage toward a
(position, price-tier) prior fitted on history, using the standard
Poisson-Gamma conjugate update expressed in exposure units of 90 minutes:

    posterior_rate = (n0 * mu + observed_events) / (n0 + observed_exposure_90s)

``mu`` (the prior mean rate) and ``n0`` (the prior's pseudo-exposure, in
90-minute equivalents) are fit per (position, price tier, stat) in
:func:`fit_rate_priors` by method-of-moments on a Gamma-Poisson mixture: the
between-player variance of the empirical rate, net of the Poisson sampling
variance that shrinks with exposure, pins down the shape parameter, which is
exactly ``n0``. Two gameweeks of exposure (``observed_exposure_90s`` around
1.5-2) against an ``n0`` of 6-20 for most buckets means the prior dominates
early, and fades out smoothly as ``observed_exposure_90s`` grows across the
season -- which is the whole point.

Where Understat has a current-season match for a player, goals and assists
are blended against xG/xA before the shrinkage step (``observed_events =
0.5 * actual + 0.5 * xG_or_xA``), since over 1-2 matches realised goals are
extremely noisy relative to the underlying chance quality xG measures.
Defensive actions (CBI, tackles, recoveries) and saves have no Understat
analogue and use FPL's own season-to-date counts only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._teamnames import normalize_name, normalize_team

__all__ = [
    "RATE_STATS",
    "DEFENSIVE_STATS_MIN_SEASON",
    "fit_rate_priors",
    "match_understat",
    "estimate_rates",
]

log = logging.getLogger(__name__)

#: (stat column on the bootstrap `elements` frame, human label)
RATE_STATS: tuple[str, ...] = (
    "goals_scored",
    "assists",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "saves",
)
#: Defensive-action columns were dropped from the public feed for several
#: seasons and only reappeared in 2025-26 (see fplopt.data.history docstring
#: and fplopt.scoring.rules) -- fitting their priors on earlier, all-null
#: seasons would silently produce a prior of exactly zero.
DEFENSIVE_STATS_MIN_SEASON = "2025-26"
_ATTACKING_STATS = ("goals_scored", "assists")
_DEFENSIVE_STATS = ("clearances_blocks_interceptions", "tackles", "recoveries")
_GK_STATS = ("saves",)

_POSITION_CODE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
_N_PRICE_BUCKETS = 3
_MIN_MINUTES_FOR_MOMENT = 180  # >= 2 full-90 equivalents before a player's own rate counts

_FALLBACK_MU: dict[tuple[int, str], float] = {
    (1, "saves"): 1.0,
    (2, "goals_scored"): 0.05, (2, "assists"): 0.05,
    (2, "clearances_blocks_interceptions"): 1.3, (2, "tackles"): 0.6, (2, "recoveries"): 2.2,
    (3, "goals_scored"): 0.12, (3, "assists"): 0.10,
    (3, "clearances_blocks_interceptions"): 0.5, (3, "tackles"): 0.5, (3, "recoveries"): 1.8,
    (4, "goals_scored"): 0.30, (4, "assists"): 0.10,
    (4, "clearances_blocks_interceptions"): 0.15, (4, "tackles"): 0.3, (4, "recoveries"): 1.2,
}


def _price_bucket_edges(prices: pd.Series, n_buckets: int = _N_PRICE_BUCKETS) -> np.ndarray:
    """Quantile edges for bucketing a position's price distribution."""
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(prices.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        return np.array([prices.min() - 1, prices.max() + 1])
    return edges


def _bucket_index(price: float, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges, price, side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


def fit_rate_priors(history_df: pd.DataFrame, seasons: tuple[str, ...] | None = None) -> dict:
    """Fit (position, price-tier) Gamma-Poisson priors for every stat in :data:`RATE_STATS`.

    Returns a dict with:
      ``table`` -- DataFrame of (position, bucket, stat, mu, n0, n_players).
      ``edges`` -- {position: price-bucket edge array}, needed to bucket a new
        player's current price the same way the priors were built.
    """
    frame = history_df.dropna(subset=["position", "value", "minutes"]).copy()
    frame["position_code"] = frame["position"].map(_POSITION_CODE)
    frame = frame.dropna(subset=["position_code"])
    frame["position_code"] = frame["position_code"].astype(int)

    edges: dict[int, np.ndarray] = {}
    for pos, g in frame.groupby("position_code"):
        priced = g[g["minutes"] > 0]
        edges[pos] = _price_bucket_edges(priced["value"]) if len(priced) else np.array([0, 1000])

    rows = []
    for stat in RATE_STATS:
        stat_seasons = seasons
        if stat_seasons is None:
            if stat in _DEFENSIVE_STATS:
                stat_seasons = tuple(s for s in history_df["season"].unique() if s >= DEFENSIVE_STATS_MIN_SEASON)
            else:
                stat_seasons = tuple(sorted(history_df["season"].unique())[-6:])
        stat_frame = frame[frame["season"].isin(stat_seasons)]
        stat_frame = stat_frame.dropna(subset=[stat])

        # Aggregate to player-season level: total events, total minutes.
        agg = (
            stat_frame.groupby(["season", "element", "position_code"])
            .agg(events=(stat, "sum"), minutes=("minutes", "sum"), price=("value", "mean"))
            .reset_index()
        )
        agg = agg[agg["minutes"] > 0]
        agg["exposure"] = agg["minutes"] / 90.0
        agg["rate"] = agg["events"] / agg["exposure"]

        for pos in (1, 2, 3, 4):
            if (pos == 1) != (stat in _GK_STATS):
                continue  # saves is GK-only; everything else is outfield-only
            pos_agg = agg[agg["position_code"] == pos]
            pos_edges = edges.get(pos, np.array([0, 1000]))
            if pos_agg.empty:
                for b in range(len(pos_edges) - 1):
                    rows.append(_fallback_row(pos, b, stat))
                continue
            bucket_idx = pos_agg["price"].map(lambda p: _bucket_index(p, pos_edges))
            for b in range(len(pos_edges) - 1):
                bucket_rows = pos_agg[bucket_idx == b]
                rows.append(_moment_match(pos, b, stat, bucket_rows))

    table = pd.DataFrame(rows)
    return {"table": table, "edges": edges}


def _fallback_row(position: int, bucket: int, stat: str) -> dict:
    mu = _FALLBACK_MU.get((position, stat), 0.2)
    return {"position": position, "bucket": bucket, "stat": stat, "mu": mu, "n0": 4.0, "n_players": 0}


def _moment_match(position: int, bucket: int, stat: str, rows: pd.DataFrame) -> dict:
    if rows.empty or rows["exposure"].sum() == 0:
        return _fallback_row(position, bucket, stat)
    mu = float(rows["events"].sum() / rows["exposure"].sum())
    mu = max(mu, 1e-4)

    moment_rows = rows[rows["minutes"] >= _MIN_MINUTES_FOR_MOMENT]
    if len(moment_rows) < 8:
        n0 = 6.0  # not enough player-seasons to estimate dispersion; moderate default shrinkage
    else:
        weights = moment_rows["exposure"].to_numpy()
        rate = moment_rows["rate"].to_numpy()
        weighted_mean = np.average(rate, weights=weights)
        weighted_var = np.average((rate - weighted_mean) ** 2, weights=weights)
        sampling_component = mu * np.average(1.0 / moment_rows["exposure"].to_numpy(), weights=weights)
        true_var = weighted_var - sampling_component
        true_var = max(true_var, (mu**2) / 50.0)
        n0 = mu / true_var
        n0 = float(np.clip(n0, 2.0, 30.0))
    return {
        "position": position, "bucket": bucket, "stat": stat,
        "mu": mu, "n0": n0, "n_players": int(rows["element"].nunique()) if "element" in rows else len(rows),
    }


def match_understat(elements: pd.DataFrame, understat_players: pd.DataFrame) -> pd.DataFrame:
    """Join current-season Understat per-player aggregates onto FPL player ids.

    Matches on normalised full name, with a normalised-team tiebreak when a
    name collides across two players (rare, but happens with common
    surnames). Returns one row per FPL player id that matched, with
    ``xg``, ``xa``, ``shots``, ``key_passes``, ``understat_minutes``.
    Players who did not match (new signings Understat hasn't indexed yet,
    name-normalisation misses) are simply absent -- callers fall back to
    FPL-only data for them.
    """
    fpl = elements[["id", "full_name", "team_name"]].copy()
    fpl["name_key"] = fpl["full_name"].map(normalize_name)
    fpl["team_key"] = fpl["team_name"].map(normalize_team)

    u = understat_players.copy()
    u = u[~u["team_title"].astype(str).str.contains(",", na=False)]  # drop mid-season-transfer rows
    u["name_key"] = u["player_name"].map(normalize_name)
    u["team_key"] = u["team_title"].map(normalize_team)

    merged = fpl.merge(u, on=["name_key", "team_key"], how="inner", suffixes=("", "_u"))
    matched_ids = set(merged["id"])

    # Fallback pass: name-only match for FPL rows not yet matched, guarded to
    # require the name key to be unique on both sides (otherwise ambiguous).
    remaining_fpl = fpl[~fpl["id"].isin(matched_ids)]
    name_counts_fpl = remaining_fpl["name_key"].value_counts()
    name_counts_u = u["name_key"].value_counts()
    unique_names = set(name_counts_fpl[name_counts_fpl == 1].index) & set(
        name_counts_u[name_counts_u == 1].index
    )
    extra = remaining_fpl[remaining_fpl["name_key"].isin(unique_names)].merge(
        u, on="name_key", how="inner", suffixes=("", "_u")
    )

    both = pd.concat([merged, extra], ignore_index=True)
    both = both.drop_duplicates(subset=["id"])
    return pd.DataFrame(
        {
            "player_id": both["id"].astype(int),
            "xg": pd.to_numeric(both["xG"], errors="coerce"),
            "xa": pd.to_numeric(both["xA"], errors="coerce"),
            "shots": pd.to_numeric(both["shots"], errors="coerce"),
            "key_passes": pd.to_numeric(both["key_passes"], errors="coerce"),
            "understat_minutes": pd.to_numeric(both["time"], errors="coerce"),
        }
    )


def estimate_rates(
    elements: pd.DataFrame,
    priors: dict,
    understat_matched: pd.DataFrame | None = None,
    *,
    xg_blend_weight: float = 0.5,
) -> pd.DataFrame:
    """Posterior per-90 rates for every player, EB-shrunk toward position/price priors.

    Parameters
    ----------
    elements:
        ``fplopt.data.fpl_api.elements_frame`` output: needs ``id``,
        ``element_type``, ``now_cost``, ``minutes``, and the current-season
        cumulative counting stats in :data:`RATE_STATS`.
    priors:
        Output of :func:`fit_rate_priors`.
    understat_matched:
        Output of :func:`match_understat`, or ``None`` to skip the xG/xA blend
        entirely (FPL-only rates for goals/assists).

    Returns
    -------
    DataFrame with one row per player: ``player_id`` plus ``<stat>_p90`` for
    every stat in :data:`RATE_STATS`.
    """
    table = priors["table"]
    edges = priors["edges"]
    prior_lookup = table.set_index(["position", "bucket", "stat"])[["mu", "n0"]]

    frame = elements.copy()
    frame["position"] = frame["element_type"].astype(int)
    frame["exposure"] = frame["minutes"].clip(lower=0) / 90.0

    if understat_matched is not None and not understat_matched.empty:
        frame = frame.merge(understat_matched, left_on="id", right_on="player_id", how="left")
    else:
        frame["xg"] = np.nan
        frame["xa"] = np.nan

    out = pd.DataFrame({"player_id": frame["id"].astype(int)})
    for stat in RATE_STATS:
        blended_events = frame[stat].astype(float).clip(lower=0)
        if stat == "goals_scored":
            has_xg = frame["xg"].notna()
            blended_events = np.where(
                has_xg, (1 - xg_blend_weight) * blended_events + xg_blend_weight * frame["xg"], blended_events
            )
        elif stat == "assists":
            has_xa = frame["xa"].notna()
            blended_events = np.where(
                has_xa, (1 - xg_blend_weight) * blended_events + xg_blend_weight * frame["xa"], blended_events
            )
        blended_events = pd.Series(blended_events, index=frame.index).clip(lower=0)

        mu = np.empty(len(frame))
        n0 = np.empty(len(frame))
        for i, (pos, price) in enumerate(zip(frame["position"], frame["now_cost"])):
            pos_edges = edges.get(int(pos), np.array([0, 1000]))
            bucket = _bucket_index(price, pos_edges)
            key = (int(pos), bucket, stat)
            if key in prior_lookup.index:
                mu[i], n0[i] = prior_lookup.loc[key]
            else:
                mu[i], n0[i] = _FALLBACK_MU.get((int(pos), stat), 0.2), 6.0

        posterior = (n0 * mu + blended_events.to_numpy()) / (n0 + frame["exposure"].to_numpy())
        # Goalkeepers never accrue outfield defensive/attacking stats worth
        # modelling as a rate; zero them rather than reporting a noisy prior.
        if stat != "saves":
            posterior = np.where(frame["position"].to_numpy() == 1, 0.0, posterior)
        else:
            posterior = np.where(frame["position"].to_numpy() == 1, posterior, 0.0)
        out[f"{stat}_p90"] = posterior

    return out
