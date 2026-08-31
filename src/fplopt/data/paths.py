"""Filesystem layout for the data layer.

Everything the ingestion layer writes lives under a single project-relative
``data/`` root so the whole cache can be inspected or deleted in one move.
The root can be relocated with the ``FPLOPT_DATA_DIR`` environment variable,
which is what the test-suite uses to keep fixtures out of the real cache.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "project_root",
    "data_dir",
    "cache_dir",
    "history_dir",
    "understat_dir",
    "fbref_dir",
    "ensure_dir",
]

_ENV_VAR = "FPLOPT_DATA_DIR"


def project_root() -> Path:
    """Return the repository root (the parent of ``src/``)."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Return the root directory for all persisted data.

    Honours the ``FPLOPT_DATA_DIR`` environment variable when set, otherwise
    defaults to ``<project_root>/data``.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "data"


def cache_dir() -> Path:
    """Return the directory holding raw HTTP response bodies."""
    return data_dir() / "cache"


def history_dir() -> Path:
    """Return the directory holding the vaastav historical archive parquet."""
    return data_dir() / "history"


def understat_dir() -> Path:
    """Return the directory holding Understat parquet extracts."""
    return data_dir() / "understat"


def fbref_dir() -> Path:
    """Return the directory holding FBref parquet extracts."""
    return data_dir() / "fbref"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
