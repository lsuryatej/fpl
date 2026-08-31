"""Shared HTTP plumbing: rate limiting, retries with backoff, disk caching.

Every network client in :mod:`fplopt.data` goes through :class:`RateLimitedSession`
so that politeness policy lives in exactly one place. Failures are surfaced as
:class:`FetchError` rather than swallowed -- a caller that wants to tolerate a
missing resource has to say so explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import requests

from .paths import cache_dir, ensure_dir

__all__ = [
    "FetchError",
    "HTTPStatusError",
    "RateLimitedSession",
    "cache_key",
]

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Status codes worth retrying: rate limiting plus transient upstream failures.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """A network fetch failed in a way the caller must handle."""


class HTTPStatusError(FetchError):
    """The server answered with a non-2xx status that we will not retry past."""

    def __init__(self, url: str, status_code: int, body_excerpt: str = "") -> None:
        self.url = url
        self.status_code = status_code
        self.body_excerpt = body_excerpt
        detail = f" body={body_excerpt!r}" if body_excerpt else ""
        super().__init__(f"GET {url} returned HTTP {status_code}.{detail}")


def cache_key(url: str, params: Mapping[str, Any] | None = None) -> str:
    """Return a stable filename stem for a URL plus query parameters.

    The digest keeps paths short and filesystem-safe while a readable prefix
    keeps the cache directory browsable by eye.
    """
    canonical = url
    if params:
        canonical += "?" + json.dumps(dict(sorted(params.items())), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    readable = "".join(c if c.isalnum() else "-" for c in url.split("//", 1)[-1])
    return f"{readable[:80].strip('-')}-{digest}"


@dataclass
class RateLimitedSession:
    """A ``requests`` session with politeness, retries and on-disk caching.

    Parameters
    ----------
    min_interval:
        Minimum number of seconds between two outbound requests from this
        session. Enforced with a lock so concurrent callers cannot beat it.
    max_retries:
        How many additional attempts to make after the first one fails with a
        retryable status or a transport error.
    user_agent:
        Value for the ``User-Agent`` header. Defaults to a real browser string
        because several of these hosts reject obvious bot agents.
    cache_subdir:
        Directory beneath ``data/cache/`` for this session's responses.
    cache_enabled:
        Set ``False`` to bypass the disk cache entirely (used by live checks).
    timeout:
        Per-request socket timeout in seconds.
    """

    min_interval: float = 0.4
    max_retries: int = 4
    user_agent: str = DEFAULT_USER_AGENT
    cache_subdir: str = "misc"
    cache_enabled: bool = True
    timeout: float = 30.0
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    _session: requests.Session = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)
    _last_request_at: float = field(default=0.0, init=False, repr=False)
    request_count: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._lock = threading.Lock()
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }
        headers.update(self.extra_headers)
        self._session.headers.update(headers)

    # ------------------------------------------------------------------
    # cache helpers
    # ------------------------------------------------------------------
    def _cache_path(self, url: str, params: Mapping[str, Any] | None, suffix: str) -> Path:
        # Keying on the date means a cached body is reused within a day and
        # naturally refreshed the next day, which matches how often these
        # sources actually change.
        day = date.today().isoformat()
        directory = ensure_dir(cache_dir() / self.cache_subdir / day)
        return directory / f"{cache_key(url, params)}{suffix}"

    def _read_cache(self, path: Path) -> bytes | None:
        if not self.cache_enabled or not path.exists():
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            log.warning("Could not read cache file %s: %s", path, exc)
            return None

    def _write_cache(self, path: Path, payload: bytes) -> None:
        if not self.cache_enabled:
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_bytes(payload)
            tmp.replace(path)
        except OSError as exc:
            log.warning("Could not write cache file %s: %s", path, exc)

    # ------------------------------------------------------------------
    # request path
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self.min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _request(
        self,
        url: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        method: str,
        data: Mapping[str, Any] | None,
    ) -> requests.Response:
        """Issue the request, retrying retryable failures with backoff."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self._session.request(
                    method,
                    url,
                    params=dict(params) if params else None,
                    data=dict(data) if data else None,
                    headers=dict(headers) if headers else None,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise FetchError(f"GET {url} failed after {attempt + 1} attempts: {exc}") from exc
                self._sleep_backoff(attempt, None)
                continue

            self.request_count += 1
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                log.info(
                    "HTTP %s from %s (attempt %d/%d), backing off",
                    response.status_code,
                    url,
                    attempt + 1,
                    self.max_retries + 1,
                )
                self._sleep_backoff(attempt, retry_after)
                continue

            if not response.ok:
                raise HTTPStatusError(url, response.status_code, response.text[:200])
            return response

        # Only reachable if every attempt was a retryable status.
        raise FetchError(f"GET {url} exhausted {self.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def _sleep_backoff(attempt: int, retry_after: str | None) -> None:
        """Sleep for an exponential backoff, honouring ``Retry-After`` if sane."""
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 0.0
            if 0 < delay <= 120:
                time.sleep(delay)
                return
        time.sleep(min(2.0**attempt, 30.0) + random.uniform(0, 0.5))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_bytes(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        suffix: str = ".bin",
        use_cache: bool = True,
    ) -> bytes:
        """Fetch ``url`` and return the raw body, using the disk cache when possible."""
        path = self._cache_path(url, params, suffix)
        if use_cache:
            cached = self._read_cache(path)
            if cached is not None:
                self.cache_hits += 1
                return cached
        response = self._request(url, params, headers, "GET", None)
        payload = response.content
        if use_cache:
            self._write_cache(path, payload)
        return payload

    def get_text(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        suffix: str = ".txt",
        use_cache: bool = True,
    ) -> str:
        """Fetch ``url`` and return the decoded body text."""
        raw = self.get_bytes(url, params, headers, suffix=suffix, use_cache=use_cache)
        return raw.decode("utf-8", errors="replace")

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        use_cache: bool = True,
    ) -> Any:
        """Fetch ``url`` and parse the body as JSON.

        Raises
        ------
        FetchError
            If the transport failed or the body is not valid JSON.
        """
        raw = self.get_bytes(url, params, headers, suffix=".json", use_cache=use_cache)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # A poisoned cache entry is worse than a slow one; drop it.
            path = self._cache_path(url, params, ".json")
            if path.exists():
                path.unlink(missing_ok=True)
            raise FetchError(f"GET {url} did not return valid JSON: {exc}") from exc

    def post_json(
        self,
        url: str,
        data: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """POST form-encoded ``data`` to ``url`` and parse the JSON response.

        POST responses are never cached.
        """
        response = self._request(url, None, headers, "POST", data)
        try:
            return response.json()
        except ValueError as exc:
            raise FetchError(f"POST {url} did not return valid JSON: {exc}") from exc

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._session.close()
