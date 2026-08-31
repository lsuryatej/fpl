"""Render a Brief as a self-contained HTML page.

Written as an artifact fragment: a title, a style block and the body content,
with no document skeleton, because the publishing step supplies that. Browsers
render it directly from disk too.

Design notes. This is an instrument panel rather than a document, so severity
is encoded in form (a stripe and a pill) as well as colour, every numeric
column uses tabular figures, and the summary sits above the detail. Neutrals
carry a slight green bias so they read as chosen rather than defaulted.
"""

from __future__ import annotations

import datetime as dt
import html
from typing import Any

import pandas as pd

from fplopt.brief.assemble import Brief

POSITION_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

STYLE = """
:root {
  --ground:#f6f8f6; --surface:#ffffff; --surface-2:#f0f4f1;
  --ink:#111917; --ink-2:#3a4a45; --muted:#5b6b67; --line:#dce4e0;
  --accent:#0e6b57; --accent-soft:#e3f0eb;
  --critical:#a93a2c; --critical-soft:#f8e7e4;
  --warn:#8d6413; --warn-soft:#f8f0dd;
  --good:#2c7a55; --good-soft:#e4f2ea;
  --shadow:0 1px 2px rgba(17,25,23,.05), 0 8px 24px -16px rgba(17,25,23,.25);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#0d1211; --surface:#141b19; --surface-2:#1a2321;
    --ink:#e7edea; --ink-2:#b8c6c1; --muted:#87968f; --line:#242e2b;
    --accent:#45c4a2; --accent-soft:#12302a;
    --critical:#e88274; --critical-soft:#33201c;
    --warn:#d6a55a; --warn-soft:#302819;
    --good:#5cc48d; --good-soft:#16301f;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"] {
  --ground:#0d1211; --surface:#141b19; --surface-2:#1a2321;
  --ink:#e7edea; --ink-2:#b8c6c1; --muted:#87968f; --line:#242e2b;
  --accent:#45c4a2; --accent-soft:#12302a;
  --critical:#e88274; --critical-soft:#33201c;
  --warn:#d6a55a; --warn-soft:#302819;
  --good:#5cc48d; --good-soft:#16301f;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Archivo","Helvetica Neue",Arial,sans-serif;
  font-size:15px; line-height:1.5;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1360px; margin:0 auto; padding:28px 22px 80px; }
h1,h2,h3 { margin:0; text-wrap:balance; font-weight:700; letter-spacing:-.02em; }
.num, td.n, th.n { font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums; }

header.top { display:flex; flex-wrap:wrap; gap:18px; align-items:flex-end;
  justify-content:space-between; padding-bottom:18px; border-bottom:2px solid var(--ink); }
.eyebrow { font-size:11px; text-transform:uppercase; letter-spacing:.14em;
  color:var(--muted); font-weight:600; }
h1 { font-size:clamp(26px,3.6vw,40px); line-height:1.02; }
.deadline { text-align:right; }
.deadline .big { font-size:clamp(20px,2.4vw,28px); font-weight:700; color:var(--accent); }

.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line); margin:22px 0 26px; }
.kpi { background:var(--surface); padding:14px 16px; }
.kpi .label { font-size:11px; text-transform:uppercase; letter-spacing:.1em;
  color:var(--muted); font-weight:600; }
.kpi .value { font-size:26px; font-weight:700; margin-top:4px; }
.kpi .sub { font-size:12px; color:var(--muted); }
.kpi.bad .value { color:var(--critical); }
.kpi.good .value { color:var(--good); }

section { margin:30px 0; }
.sec-head { display:flex; align-items:baseline; gap:12px; margin-bottom:12px;
  padding-bottom:8px; border-bottom:1px solid var(--line); }
.sec-head h2 { font-size:15px; text-transform:uppercase; letter-spacing:.1em; }
.sec-head .note { font-size:12.5px; color:var(--muted); }

.verdict { background:var(--surface); border:1px solid var(--line);
  border-left:4px solid var(--accent); padding:20px 22px; box-shadow:var(--shadow); }
.verdict.pending { border-left-color:var(--warn); }
.verdict h2 { font-size:19px; margin-bottom:6px; }
.verdict p { margin:6px 0 0; color:var(--ink-2); max-width:72ch; }

.grid { display:grid; gap:22px; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); }

.scroll { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th { text-align:left; font-size:10.5px; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); font-weight:700; padding:6px 9px; border-bottom:1px solid var(--line);
  white-space:nowrap; }
td { padding:6px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }
th.n, td.n { text-align:right; }
tbody tr:hover { background:var(--surface-2); }
.name { font-weight:600; }
.dim { color:var(--muted); }
.pos-neg { color:var(--critical); font-weight:600; }
.pos-pos { color:var(--good); font-weight:600; }

.pill { display:inline-block; padding:1px 7px; border-radius:3px; font-size:10.5px;
  font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.pill.critical { background:var(--critical-soft); color:var(--critical); }
.pill.high { background:var(--warn-soft); color:var(--warn); }
.pill.medium { background:var(--surface-2); color:var(--muted); }
.pill.low { background:var(--surface-2); color:var(--muted); }
.pill.own { background:var(--accent-soft); color:var(--accent); }

.risk { display:flex; gap:12px; padding:11px 14px; background:var(--surface);
  border:1px solid var(--line); border-left:3px solid var(--muted); margin-bottom:8px; }
.risk.critical { border-left-color:var(--critical); }
.risk.high { border-left-color:var(--warn); }
.risk .body { flex:1; min-width:0; }
.risk .who { font-weight:700; }
.risk .detail { font-size:12.5px; color:var(--ink-2); margin-top:2px; }

ul.qs { margin:0; padding-left:20px; }
ul.qs li { margin-bottom:7px; color:var(--ink-2); font-size:13.5px; }

.panel { background:var(--surface); border:1px solid var(--line); padding:16px 18px; }
.empty { color:var(--muted); font-size:13.5px; font-style:italic; padding:10px 0; }

footer { margin-top:44px; padding-top:16px; border-top:1px solid var(--line);
  font-size:12px; color:var(--muted); display:flex; justify-content:space-between;
  flex-wrap:wrap; gap:10px; }
@media (max-width:640px) { .grid { grid-template-columns:1fr; } .wrap { padding:18px 14px 60px; } }
"""


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _fmt_delta(value: float, digits: int = 1) -> str:
    cls = "pos-pos" if value > 0 else ("pos-neg" if value < 0 else "dim")
    return f'<span class="{cls} num">{value:+.{digits}f}</span>'


def _countdown(deadline: dt.datetime | None, now: dt.datetime) -> str:
    if deadline is None:
        return "unknown"
    delta = deadline - now
    if delta.total_seconds() <= 0:
        return "passed"
    days, rem = divmod(int(delta.total_seconds()), 86400)
    hours, minutes = divmod(rem // 60, 60)
    if days:
        return f"{days}d {hours}h"
    return f"{hours}h {minutes}m"


def _table(headers: list[tuple[str, bool]], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="empty">No rows.</p>'
    head = "".join(
        f'<th class="{"n" if numeric else ""}">{esc(label)}</th>'
        for label, numeric in headers
    )
    body = "".join(f"<tr>{''.join(cells)}</tr>" for cells in rows)
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _section(title: str, note: str, content: str) -> str:
    return (
        f'<section><div class="sec-head"><h2>{esc(title)}</h2>'
        f'<span class="note">{esc(note)}</span></div>{content}</section>'
    )


# ---------------------------------------------------------------- panels


def _verdict(brief: Brief) -> str:
    if brief.recommendation:
        rec = brief.recommendation
        lines = "".join(f"<p>{esc(line)}</p>" for line in rec.get("summary", []))
        return f'<div class="verdict"><h2>{esc(rec.get("headline", "Recommendation"))}</h2>{lines}</div>'
    waiting = ", ".join(brief.pending)
    return (
        '<div class="verdict pending"><h2>Recommendation pending</h2>'
        f"<p>The projection model and optimizer have not produced output yet "
        f"(waiting on: {esc(waiting)}). Everything below is measured from live "
        f"data and stands on its own.</p></div>"
    )


def _kpis(brief: Brief) -> str:
    standing = brief.standing
    gap = (standing.get("leader") or 0) - (standing.get("total") or 0)
    cards = [
        ("Rank", f"{standing.get('rank')}/{brief.manager['n_managers']}", "in league", "bad"),
        ("Points", f"{standing.get('total')}", f"leader {standing.get('leader')}", ""),
        ("Gap to 1st", f"{gap}", f"{gap / max(brief.remaining_gws,1):.2f} per GW", "bad"),
        ("Net vs field", f"{standing.get('net_swing', 0):+.0f}", "points swung", "bad"),
        ("Squad risks", f"{len(brief.risks)}", "flagged players", "bad" if brief.risks else "good"),
        ("GWs left", f"{brief.remaining_gws}", "to close it", ""),
    ]
    cells = "".join(
        f'<div class="kpi {cls}"><div class="label">{esc(label)}</div>'
        f'<div class="value num">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
        for label, value, sub, cls in cards
    )
    return f'<div class="kpis">{cells}</div>'


def _risk_panel(brief: Brief) -> str:
    if not brief.risks:
        return '<p class="empty">No availability risks detected in your squad.</p>'
    blocks = []
    for risk in brief.risks:
        blocks.append(
            f'<div class="risk {esc(risk.severity)}">'
            f'<div><span class="pill {esc(risk.severity)}">{esc(risk.severity)}</span></div>'
            f'<div class="body"><div class="who">{esc(risk.name)} '
            f'<span class="dim">{esc(risk.team)} &middot; '
            f'{esc(POSITION_NAME.get(risk.position,""))} &middot; '
            f'<span class="num">{risk.price:.1f}</span>m</span></div>'
            f'<div class="detail">{esc(risk.detail)}</div></div></div>'
        )
    return "".join(blocks)


def _squad_panel(brief: Brief) -> str:
    rows = []
    for row in brief.squad:
        marks = []
        if row["is_captain"]:
            marks.append('<span class="pill own">C</span>')
        if row["is_vice"]:
            marks.append('<span class="pill medium">V</span>')
        if row["benched"]:
            marks.append('<span class="pill medium">bench</span>')
        if row["status"] != "a":
            marks.append(f'<span class="pill critical">{esc(row["status"])}</span>')
        rows.append(
            [
                f'<td class="name">{esc(row["name"])}</td>',
                f'<td class="dim">{esc(POSITION_NAME.get(row["position"],""))}</td>',
                f'<td class="dim">{esc(row["team"])}</td>',
                f'<td class="n num">{row["price"]:.1f}</td>',
                f'<td class="n num">{row["points"]}</td>',
                f'<td class="n num">{row["minutes"]}</td>',
                f'<td class="n num">{row["selected_by"]:.1f}</td>',
                f"<td>{' '.join(marks)}</td>",
            ]
        )
    return _table(
        [
            ("Player", False),
            ("Pos", False),
            ("Team", False),
            ("Price", True),
            ("Pts", True),
            ("Mins", True),
            ("Own %", True),
            ("", False),
        ],
        rows,
    )


def _swing_table(swings: list[Any], own_label: str) -> str:
    rows = []
    for swing in swings:
        cost = getattr(swing, "swing", None)
        if cost is None:
            cost = getattr(swing, "weighted_swing", 0.0)
        rows.append(
            [
                f'<td class="name">{esc(getattr(swing, "name", ""))}</td>',
                f'<td class="dim">{esc(getattr(swing, "team", ""))}</td>',
                f'<td class="n num">{getattr(swing, "price", 0):.1f}</td>',
                f'<td class="n num">{getattr(swing, "owned_pct", 0):.0f}</td>',
                f'<td class="n num">{getattr(swing, "points", 0)}</td>',
                f'<td class="n">{_fmt_delta(cost)}</td>',
            ]
        )
    return _table(
        [
            ("Player", False),
            ("Team", False),
            ("Price", True),
            (own_label, True),
            ("Pts", True),
            ("Swing", True),
        ],
        rows,
    )


def _ownership_panel(brief: Brief) -> str:
    rows = []
    for row in brief.ownership:
        rows.append(
            [
                f'<td class="name">{esc(row.name)}</td>',
                f'<td class="dim">{esc(row.position)} {esc(row.team)}</td>',
                f'<td class="n num">{row.price:.1f}</td>',
                f'<td class="n num">{row.owned_pct:.0f}</td>',
                f'<td class="n num">{row.eo_pct:.0f}</td>',
                f'<td class="n">{_fmt_delta(row.league_vs_global, 0)}</td>',
                f'<td class="n num">{row.points}</td>',
                f'<td>{"<span class=\'pill own\'>own</span>" if row.user_owns else ""}</td>',
            ]
        )
    return _table(
        [
            ("Player", False),
            ("Pos", False),
            ("Price", True),
            ("League %", True),
            ("EO %", True),
            ("vs global", True),
            ("Pts", True),
            ("", False),
        ],
        rows,
    )


def _gap_panel(brief: Brief) -> str:
    rows = [
        [
            f'<td class="name">{esc(target.label)}</td>',
            f'<td class="n num">{target.total}</td>',
            f'<td class="n num">{target.gap}</td>',
            f'<td class="n num">{target.per_gw:.2f}</td>',
        ]
        for target in brief.gap_targets
    ]
    return _table(
        [("Target", False), ("Their total", True), ("Gap", True), ("Edge / GW", True)],
        rows,
    )


def _price_panel(brief: Brief) -> str:
    frame: pd.DataFrame = brief.price_movers
    if frame is None or frame.empty:
        return '<p class="empty">No price movement data yet.</p>'
    owned = frame[frame.get("owned", False)] if "owned" in frame else frame.head(0)
    others = frame[~frame.get("owned", False)] if "owned" in frame else frame
    selection = pd.concat([owned, others.head(12)])
    rows = []
    for _, row in selection.iterrows():
        direction = "rise" if row["proj_pct_today"] > 0 else "fall"
        pill = "good" if direction == "rise" else "critical"
        rows.append(
            [
                f'<td class="name">{esc(row["web_name"])}</td>',
                f'<td class="n num">{row["now_cost"]/10:.1f}</td>',
                f'<td><span class="pill {pill}">{direction}</span></td>',
                f'<td class="n num">{row["proj_pct_today"]:.0f}</td>',
                f'<td class="n num">{int(row["hourly_rate"]) if pd.notna(row["hourly_rate"]) else 0:,}</td>',
                f'<td>{"<span class=\'pill own\'>yours</span>" if row.get("owned") else ""}</td>',
            ]
        )
    return _table(
        [
            ("Player", False),
            ("Price", True),
            ("Move", False),
            ("Progress %", True),
            ("Per hour", True),
            ("", False),
        ],
        rows,
    )


def _fixtures_panel(brief: Brief) -> str:
    rows = [
        [
            f'<td class="dim num">{esc(fixture["kickoff"])}</td>',
            f'<td class="name">{esc(fixture["home"])}</td>',
            f'<td class="dim">v</td>',
            f'<td class="name">{esc(fixture["away"])}</td>',
            f'<td class="n num">{esc(fixture["home_fdr"])}</td>',
            f'<td class="n num">{esc(fixture["away_fdr"])}</td>',
        ]
        for fixture in brief.fixtures
    ]
    return _table(
        [
            ("Kickoff", False),
            ("Home", False),
            ("", False),
            ("Away", False),
            ("H FDR", True),
            ("A FDR", True),
        ],
        rows,
    )


def _questions_panel(brief: Brief) -> str:
    if not brief.open_questions:
        return '<p class="empty">Nothing needs a news check.</p>'
    items = "".join(f"<li>{esc(question)}</li>" for question in brief.open_questions)
    return f'<div class="panel"><ul class="qs">{items}</ul></div>'


def _projection_panel(brief: Brief) -> str:
    frame = brief.projections
    if frame is None or frame.empty:
        return (
            '<p class="empty">Projections not generated yet. This panel fills in '
            "once the model writes to data/projections/.</p>"
        )
    target = frame[frame["gw"] == brief.target_gw].nlargest(25, "ep")
    rows = []
    for _, row in target.iterrows():
        rows.append(
            [
                f'<td class="name">{esc(row.get("web_name", row["player_id"]))}</td>',
                f'<td class="dim">{esc(POSITION_NAME.get(int(row["position"]),""))}</td>',
                f'<td class="n num">{row["price"]/10:.1f}</td>',
                f'<td class="n num">{row["ep"]:.2f}</td>',
                f'<td class="n num">{row["p10"]:.1f}</td>',
                f'<td class="n num">{row["p90"]:.1f}</td>',
                f'<td class="n num">{row["exp_minutes"]:.0f}</td>',
            ]
        )
    return _table(
        [
            ("Player", False),
            ("Pos", False),
            ("Price", True),
            ("xPts", True),
            ("p10", True),
            ("p90", True),
            ("Mins", True),
        ],
        rows,
    )


def render(brief: Brief) -> str:
    """Produce the complete HTML fragment for this brief."""
    now = brief.generated_at
    countdown = _countdown(brief.deadline, now)
    deadline_text = (
        brief.deadline.strftime("%a %d %b, %H:%M UTC") if brief.deadline else "unknown"
    )
    notes = "".join(f"<p>{esc(note)}</p>" for note in brief.notes)

    parts = [
        "<title>ECPP War Room</title>",
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Archivo:wght@400;500;600;700;800&"
        'family=JetBrains+Mono:wght@400;500;700&display=swap">',
        f"<style>{STYLE}</style>",
        '<div class="wrap">',
        '<header class="top"><div>'
        f'<div class="eyebrow">{esc(brief.manager["league"])} &middot; '
        f'{esc(brief.manager["team_name"] or "your squad")}</div>'
        f"<h1>Gameweek {brief.target_gw} decision brief</h1></div>"
        f'<div class="deadline"><div class="eyebrow">Deadline</div>'
        f'<div class="big num">{esc(countdown)}</div>'
        f'<div class="eyebrow">{esc(deadline_text)}</div></div></header>',
        _kpis(brief),
        _verdict(brief),
    ]

    if notes:
        parts.append(f'<section><div class="panel">{notes}</div></section>')

    parts.append(
        _section(
            "Risk board",
            "availability the model cannot see",
            _risk_panel(brief),
        )
    )
    parts.append(
        _section(
            "Needs a news check",
            "no API field settles these",
            _questions_panel(brief),
        )
    )
    parts.append(
        '<div class="grid">'
        + _section("Your squad", f"as picked in GW{brief.target_gw - 1}", _squad_panel(brief))
        + _section("Gameweek fixtures", f"GW{brief.target_gw}", _fixtures_panel(brief))
        + "</div>"
    )
    parts.append(
        '<div class="grid">'
        + _section(
            "Liabilities",
            "the league owns them, you do not",
            _swing_table(brief.liabilities, "League %"),
        )
        + _section(
            "Your differentials",
            "you own them, the league mostly does not",
            _swing_table(brief.differentials, "League %"),
        )
        + "</div>"
    )
    parts.append(
        _section(
            "League ownership",
            "effective ownership inside this league, not the global game",
            _ownership_panel(brief),
        )
    )
    parts.append(
        '<div class="grid">'
        + _section("Gap analysis", f"{brief.remaining_gws} gameweeks remain", _gap_panel(brief))
        + _section("Price watch", "changes apply around 01:30 UK", _price_panel(brief))
        + "</div>"
    )
    parts.append(
        _section(
            "Projections",
            f"expected points, GW{brief.target_gw}",
            _projection_panel(brief),
        )
    )

    pending = ", ".join(brief.pending) if brief.pending else "none"
    parts.append(
        "<footer>"
        f'<span>Generated {esc(now.strftime("%Y-%m-%d %H:%M UTC"))} '
        f'&middot; entry {brief.manager["entry"]}</span>'
        f"<span>Pending: {esc(pending)}</span>"
        "</footer>"
    )
    parts.append("</div>")
    return "\n".join(parts)
