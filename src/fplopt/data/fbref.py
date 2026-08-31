"""FBref scraper -- currently blocked by Cloudflare, and says so loudly.

Status as verified against the live site (2026-08-31): every fbref.com URL,
including ``/robots.txt``, answers **HTTP 403** with Cloudflare's
``<title>Just a moment...</title>`` interstitial. That is the managed
JavaScript challenge, not a User-Agent filter. It was reproduced with:

* three distinct realistic browser User-Agents (Chrome/macOS, Firefox/Windows,
  Safari/macOS) plus full ``Accept`` / ``Accept-Language`` / ``Sec-Fetch-*``
  header sets;
* a session warm-up on ``https://fbref.com/`` before the target page (the
  ``__cf_bm`` cookie is issued but does not unlock anything);
* both ``fbref.com`` and ``www.fbref.com``;
* request spacing of 3.5 to 4 seconds.

Solving the challenge needs a real JS runtime (Playwright, curl-impersonate,
FlareSolverr). Rather than pretend, :func:`fetch` raises
:class:`FBrefUnavailable` whenever a challenge page comes back.

The parsing side is real and tested offline: FBref hides most of its secondary
tables inside HTML comments, and :func:`uncomment_tables` strips the comment
markers so ``pandas.read_html`` can see them. If the block is ever lifted, or a
caller supplies HTML fetched by other means, :func:`parse_tables` and
:func:`read_tables_from_html` work today.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .http import FetchError, HTTPStatusError, RateLimitedSession

__all__ = [
    "BASE_URL",
    "FBrefUnavailable",
    "FBrefClient",
    "uncomment_tables",
    "parse_tables",
    "read_tables_from_html",
    "flatten_columns",
    "is_challenge_page",
    "probe",
]

log = logging.getLogger(__name__)

BASE_URL = "https://fbref.com"

#: FBref asks scrapers for one request per three seconds; we use a little more.
MIN_INTERVAL = 3.5

_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)

_CHALLENGE_MARKERS = (
    "just a moment",
    "cf-browser-verification",
    "cf_chl_opt",
    "challenge-platform",
    "enable javascript and cookies to continue",
)

_BROWSER_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


class FBrefUnavailable(FetchError):
    """FBref could not be read.

    Raised for the Cloudflare JS challenge, for outright 403/429 blocks, and for
    pages that come back with no parseable tables. The message always says which
    of those happened so a caller does not have to guess.
    """


# ----------------------------------------------------------------------
# detection and parsing (pure, offline-testable)
# ----------------------------------------------------------------------
def is_challenge_page(html: str) -> bool:
    """Return whether ``html`` is a Cloudflare interstitial rather than content."""
    head = html[:4000].lower()
    return any(marker in head for marker in _CHALLENGE_MARKERS)


def uncomment_tables(html: str) -> str:
    """Strip HTML comment markers so commented-out tables become parseable.

    FBref wraps every table except the first in ``<!-- ... -->``. Only comments
    that actually contain a ``<table`` are unwrapped, so genuine comments (and
    conditional-comment cruft) are left alone.
    """

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        return inner if "<table" in inner.lower() else match.group(0)

    return _COMMENT_RE.sub(_replace, html)


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse FBref's two-row MultiIndex headers into single strings.

    The top level is a spanner label like ``"Unnamed: 0_level_0"`` for plain
    columns and ``"Expected"`` for grouped ones; the former is dropped.
    """
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    out = frame.copy()
    out.columns = [
        str(lower) if str(upper).startswith("Unnamed") else f"{upper}_{lower}"
        for upper, lower in frame.columns
    ]
    return out


def parse_tables(html: str, flatten: bool = True) -> list[pd.DataFrame]:
    """Parse every table in an FBref page, including the commented-out ones.

    Raises
    ------
    FBrefUnavailable
        If the HTML is a Cloudflare challenge page or contains no tables.
    """
    if is_challenge_page(html):
        raise FBrefUnavailable(
            "Received a Cloudflare challenge page instead of content. "
            "FBref requires a JavaScript-capable client; see the module docstring."
        )
    cleaned = uncomment_tables(html)
    try:
        tables = pd.read_html(io.StringIO(cleaned), flavor="lxml")
    except ValueError as exc:
        raise FBrefUnavailable(f"No tables found in FBref HTML: {exc}") from exc
    if not tables:
        raise FBrefUnavailable("FBref HTML parsed but contained zero tables")
    return [flatten_columns(t) for t in tables] if flatten else tables


def read_tables_from_html(path_or_html: str, flatten: bool = True) -> list[pd.DataFrame]:
    """Parse tables from a local HTML file path or a raw HTML string."""
    text = path_or_html
    if "<" not in path_or_html[:200]:
        with open(path_or_html, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    return parse_tables(text, flatten=flatten)


# ----------------------------------------------------------------------
# network
# ----------------------------------------------------------------------
@dataclass
class FBrefClient:
    """Attempts to read FBref with browser-shaped headers and >=3.5s spacing.

    Every public method raises :class:`FBrefUnavailable` while the Cloudflare
    challenge is in force. No method ever returns fabricated or placeholder
    data.
    """

    min_interval: float = MIN_INTERVAL
    cache_enabled: bool = True
    warm_up: bool = True
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.session = RateLimitedSession(
            min_interval=self.min_interval,
            cache_subdir="fbref",
            cache_enabled=self.cache_enabled,
            max_retries=1,
            extra_headers=_BROWSER_HEADERS,
        )
        self._warmed = False

    # ------------------------------------------------------------------
    def fetch(self, path: str) -> str:
        """Fetch one FBref page and return its HTML.

        Raises
        ------
        FBrefUnavailable
            On a Cloudflare challenge, a 403/429, or any transport failure.
        """
        url = path if path.startswith("http") else f"{BASE_URL}/{path.lstrip('/')}"
        if self.warm_up and not self._warmed:
            self._warmed = True
            try:
                self.session.get_text(f"{BASE_URL}/", suffix=".html")
            except FetchError as exc:
                log.info("FBref warm-up request failed: %s", exc)

        try:
            html = self.session.get_text(url, suffix=".html")
        except HTTPStatusError as exc:
            message = (
                f"FBref returned HTTP {exc.status_code} for {url}. "
                "Confirmed 2026-08-31: fbref.com serves a Cloudflare JS challenge "
                "to every plain HTTP client, including /robots.txt. A "
                "JavaScript-capable fetcher is required."
            )
            self.errors.append(message)
            raise FBrefUnavailable(message) from exc
        except FetchError as exc:
            message = f"FBref transport failure for {url}: {exc}"
            self.errors.append(message)
            raise FBrefUnavailable(message) from exc

        if is_challenge_page(html):
            message = (
                f"FBref returned a Cloudflare challenge page for {url} "
                "(HTTP 200 but no content). A JavaScript-capable fetcher is required."
            )
            self.errors.append(message)
            raise FBrefUnavailable(message)
        return html

    def tables(self, path: str) -> list[pd.DataFrame]:
        """Fetch a page and return every table on it, comments included."""
        return parse_tables(self.fetch(path))

    def league_stats(self, comp_id: int = 9, season: str | None = None) -> list[pd.DataFrame]:
        """Fetch the Premier League standard-stats page (comp 9).

        ``season`` is FBref's ``YYYY-YYYY`` form, e.g. ``"2025-2026"``; omit it
        for the current campaign.
        """
        if season:
            path = f"en/comps/{comp_id}/{season}/stats/{season}-Premier-League-Stats"
        else:
            path = f"en/comps/{comp_id}/stats/Premier-League-Stats"
        return self.tables(path)

    def match_report(self, match_url_path: str) -> list[pd.DataFrame]:
        """Fetch a single match report page and return its tables."""
        return self.tables(match_url_path)

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self.session.close()


def probe(paths: tuple[str, ...] = ("robots.txt", "en/comps/9/Premier-League-Stats")) -> dict[str, Any]:
    """Check whether FBref is reachable right now.

    Returns a diagnostic dict rather than raising, so :mod:`fplopt.data.build`
    can report the block without failing the run.
    """
    client = FBrefClient(cache_enabled=False)
    results: dict[str, Any] = {"available": False, "checks": []}
    try:
        for path in paths:
            entry: dict[str, Any] = {"path": path}
            try:
                html = client.fetch(path)
                entry.update(status="ok", bytes=len(html))
                results["available"] = True
            except FBrefUnavailable as exc:
                entry.update(status="blocked", detail=str(exc)[:200])
            results["checks"].append(entry)
    finally:
        client.close()
    return results
