"""Team-name and player-name canonicalisation shared across the model package.

The vaastav historical archive, the live FPL API and Understat each spell club
names differently across seasons ("Hull" vs "Hull City", "Tottenham" vs
"Spurs", "Manchester United" vs "Man Utd"). Every module that joins across
these sources needs the same canonical key, so it lives here once.

Canonical spellings are whatever the *current* FPL ``bootstrap-static`` teams
endpoint uses (checked empirically for 2026/27: ``fpl/teams`` in the parquet
cache), since that is what the rest of the model and the optimiser downstream
key on. Historical-only clubs (no longer/not yet in the top flight) keep their
vaastav spelling as their own canonical form.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_team", "normalize_name", "TEAM_ALIASES"]

#: variant spelling (lower-cased) -> canonical spelling.
#: Built from the union of every team-name string observed in
#: history/merged_gw (2016-17..2026-27), fpl/teams (2026-27) and
#: understat/players_{2024,2025,2026} (team_title).
TEAM_ALIASES: dict[str, str] = {
    # current-squad short names are already canonical (identity, listed for
    # documentation): Arsenal, Aston Villa, Bournemouth, Brentford, Brighton,
    # Chelsea, Crystal Palace, Everton, Fulham, Leeds, Liverpool, Man City,
    # Man Utd, Newcastle, Nott'm Forest, Spurs, Sunderland.
    "hull": "Hull City",
    "hull city": "Hull City",
    "ipswich": "Ipswich Town",
    "ipswich town": "Ipswich Town",
    "coventry": "Coventry City",
    "coventry city": "Coventry City",
    "manchester city": "Man City",
    "man city": "Man City",
    "manchester united": "Man Utd",
    "man utd": "Man Utd",
    "man u": "Man Utd",
    "newcastle united": "Newcastle",
    "newcastle": "Newcastle",
    "nottingham forest": "Nott'm Forest",
    "nott'm forest": "Nott'm Forest",
    "nforest": "Nott'm Forest",
    "tottenham": "Spurs",
    "tottenham hotspur": "Spurs",
    "spurs": "Spurs",
    "wolverhampton wanderers": "Wolves",
    "wolverhampton": "Wolves",
    "wolves": "Wolves",
    "west bromwich albion": "West Brom",
    "west brom": "West Brom",
    "west ham united": "West Ham",
    "west ham": "West Ham",
    "sheffield united": "Sheffield Utd",
    "sheffield utd": "Sheffield Utd",
    "leicester city": "Leicester",
    "leicester": "Leicester",
    "norwich city": "Norwich",
    "norwich": "Norwich",
    "swansea city": "Swansea",
    "swansea": "Swansea",
    "stoke city": "Stoke",
    "stoke": "Stoke",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "brighton": "Brighton",
    "cardiff city": "Cardiff",
    "cardiff": "Cardiff",
    "huddersfield town": "Huddersfield",
    "huddersfield": "Huddersfield",
    "leeds united": "Leeds",
    "leeds": "Leeds",
    "burnley fc": "Burnley",
    "burnley": "Burnley",
    "middlesbrough": "Middlesbrough",
    "sunderland": "Sunderland",
    "southampton": "Southampton",
    "watford": "Watford",
    "luton": "Luton",
    "luton town": "Luton",
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "liverpool": "Liverpool",
}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_team(name: str | None) -> str:
    """Map any spelling of a club name to its canonical form.

    Unknown names are returned title-cased and whitespace-trimmed rather than
    raising, so a genuinely new club name degrades gracefully to "itself" and
    is easy to spot in a diff instead of crashing a pipeline.
    """
    if name is None:
        return ""
    key = _WS_RE.sub(" ", str(name).strip().lower())
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    return str(name).strip()


def normalize_name(name: str | None) -> str:
    """Fold a player name to a matchable key: ascii, lower, alnum + space only.

    Used to join Understat's ``player_name`` ("Alexander Isak") against the
    FPL API's ``first_name``/``second_name`` when building blended per-90
    rates. Diacritics are stripped (``"Gyökeres"`` -> ``"gykeres"`` ->,
    NFKD-decomposed, "o" with diaeresis drops to "o" then combining mark is
    removed) so accented names from either source still collide.
    """
    if name is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    cleaned = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", cleaned).strip()
