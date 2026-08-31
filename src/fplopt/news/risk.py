"""Detect availability and rotation risk the projection model cannot see.

The FPL ``news`` field is thin and late. In GW1-2 of 2026/27 it correctly
reported that Curtis Jones had left the league, but said nothing at all about
Ollie Watkins recording zero minutes across two gameweeks while flagged fully
available. A model fed only ``status`` would have kept projecting him.

So this module derives risk from behaviour rather than announcements:

minutes anomaly
    An expensive, nominally available player who is not actually playing.
declining trend
    Minutes share falling gameweek on gameweek.
market signal
    A collapsing price. The transfer market usually knows about an injury or a
    dropped player before the API news field is updated, so a sharp negative
    ``price_change_hourly_rate`` is treated as independent evidence.

Each risk carries a ``needs_human_check`` flag. Anything true there is a
question for a web/news pass, because no amount of API data settles it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# FPL status codes. Anything other than "a" is FPL telling us something.
STATUS_MEANING = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
    "n": "not in squad / ineligible",
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Risk:
    """One detected problem with one player."""

    player_id: int
    name: str
    team: str
    position: int
    price: float
    severity: str
    kind: str
    detail: str
    needs_human_check: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        flag = " [CHECK NEWS]" if self.needs_human_check else ""
        return (
            f"{self.severity.upper():8} {self.kind:18} {self.name:16} "
            f"GBP{self.price:>5.1f}m  {self.detail}{flag}"
        )


def _minutes_by_gw(summary: dict[str, Any]) -> list[tuple[int, int]]:
    return [(h["round"], h["minutes"]) for h in summary.get("history", [])]


def scan(
    elements: list[dict[str, Any]],
    teams: dict[int, str],
    *,
    gws_played: int,
    summaries: dict[int, dict[str, Any]] | None = None,
    price_frame: Any = None,
    only: set[int] | None = None,
    intended_bench: set[int] | None = None,
) -> list[Risk]:
    """Return every detected risk, most severe first.

    ``only`` restricts the scan to a set of player ids, which is what you want
    when checking your own squad rather than the whole game.

    ``intended_bench`` lists players who are supposed to record no minutes -- a
    backup goalkeeper being the standard case. Flagging them as a risk is a
    false positive that buries the real ones.
    """
    intended_bench = intended_bench or set()
    risks: list[Risk] = []
    available_minutes = gws_played * 90

    for element in elements:
        pid = element["id"]
        if only is not None and pid not in only:
            continue

        name = element["web_name"]
        price = element["now_cost"] / 10
        team = teams.get(element["team"], "???")
        position = element["element_type"]
        minutes = element.get("minutes", 0)
        status = element.get("status", "a")
        common = {
            "player_id": pid,
            "name": name,
            "team": team,
            "position": position,
            "price": price,
        }

        # 1. FPL has flagged the player explicitly.
        if status != "a":
            chance = element.get("chance_of_playing_next_round")
            permanent = status == "u" or chance == 0
            risks.append(
                Risk(
                    **common,
                    severity="critical" if permanent else "high",
                    kind="flagged",
                    detail=(
                        f"status={status} ({STATUS_MEANING.get(status, 'unknown')}), "
                        f"chance_next={chance}. {(element.get('news') or 'no news text').strip()}"
                    ),
                    needs_human_check=not permanent,
                    evidence={"status": status, "chance": chance},
                )
            )

        # 2. Available on paper, absent in practice. The Watkins case.
        # A designated backup recording no minutes is the plan working, not a risk.
        if (
            status == "a"
            and gws_played >= 2
            and minutes == 0
            and pid not in intended_bench
        ):
            risks.append(
                Risk(
                    **common,
                    severity="critical" if price >= 6.0 else "high",
                    kind="zero-minutes",
                    detail=(
                        f"flagged available but 0 minutes in {gws_played} GWs. "
                        f"No news explains it, so the reason is off-API."
                    ),
                    needs_human_check=True,
                    evidence={"minutes": 0, "gws": gws_played},
                )
            )
        elif (
            status == "a"
            and available_minutes
            and minutes < 0.5 * available_minutes
            and pid not in intended_bench
        ):
            share = minutes / available_minutes
            risks.append(
                Risk(
                    **common,
                    severity="high" if price >= 5.5 else "medium",
                    kind="low-minutes",
                    detail=(
                        f"only {minutes} of {available_minutes} possible minutes "
                        f"({share:.0%}). Rotation or fitness risk."
                    ),
                    needs_human_check=price >= 5.5,
                    evidence={"minutes": minutes, "share": share},
                )
            )

        # 3. Minutes trending down across gameweeks.
        if summaries and pid in summaries:
            series = _minutes_by_gw(summaries[pid])
            if len(series) >= 2:
                first, last = series[0][1], series[-1][1]
                if first >= 45 and last == 0:
                    risks.append(
                        Risk(
                            **common,
                            severity="high",
                            kind="dropped",
                            detail=(
                                f"started at {first} minutes, now 0. "
                                f"Series: {[m for _, m in series]}"
                            ),
                            needs_human_check=True,
                            evidence={"series": [m for _, m in series]},
                        )
                    )

        # 4. The market is selling hard. Often the earliest signal available.
        if price_frame is not None:
            row = price_frame[price_frame["player_id"] == pid]
            if not row.empty:
                rate = row.iloc[0].get("hourly_rate")
                pct = row.iloc[0].get("proj_pct_today")
                if rate is not None and rate < -500:
                    risks.append(
                        Risk(
                            **common,
                            severity="medium",
                            kind="market-selling",
                            detail=(
                                f"hourly rate {rate:,}, projection {pct}. The market "
                                f"is dumping him, which usually precedes the news."
                            ),
                            needs_human_check=True,
                            evidence={"hourly_rate": rate, "proj_pct": pct},
                        )
                    )

    risks.sort(key=lambda r: (SEVERITY_ORDER.get(r.severity, 9), -r.price))
    return risks


def squad_risks(
    picks: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    teams: dict[int, str],
    *,
    gws_played: int,
    **kwargs: Any,
) -> list[Risk]:
    """Risks restricted to the players actually in a squad."""
    owned = {p["element"] for p in picks}
    by_id = {e["id"]: e for e in elements}

    # The second goalkeeper is a deliberate non-player. Whichever keeper in the
    # squad has fewer minutes is the backup.
    keepers = sorted(
        (by_id[pid] for pid in owned if by_id.get(pid, {}).get("element_type") == 1),
        key=lambda e: e.get("minutes", 0),
        reverse=True,
    )
    intended_bench = {e["id"] for e in keepers[1:]}

    return scan(
        elements,
        teams,
        gws_played=gws_played,
        only=owned,
        intended_bench=intended_bench,
        **kwargs,
    )


def open_questions(risks: list[Risk]) -> list[str]:
    """The questions a news pass has to answer. API data cannot settle these."""
    questions = []
    for risk in risks:
        if not risk.needs_human_check:
            continue
        if risk.kind == "zero-minutes":
            questions.append(
                f"{risk.name} ({risk.team}): why zero minutes? Injury, transfer "
                f"saga, bust-up, or tactical? Is he expected to start next?"
            )
        elif risk.kind == "dropped":
            questions.append(
                f"{risk.name} ({risk.team}): dropped from the XI. Temporary "
                f"rotation or has he lost his place?"
            )
        elif risk.kind == "flagged":
            questions.append(
                f"{risk.name} ({risk.team}): {risk.detail[:70]}. Expected return date?"
            )
        elif risk.kind == "market-selling":
            questions.append(
                f"{risk.name} ({risk.team}): heavy sell-off. What do managers know?"
            )
        elif risk.kind == "low-minutes":
            questions.append(
                f"{risk.name} ({risk.team}): limited minutes. Is he a starter?"
            )
    return questions
