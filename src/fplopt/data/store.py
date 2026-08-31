"""Thin parquet cache over the ``data/`` directory.

Names are slash-separated logical paths, e.g. ``"history/merged_gw"`` or
``"understat/shots_2026"``. They map onto ``data/<name>.parquet``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from .paths import data_dir, ensure_dir

__all__ = ["path_for", "save", "load", "exists", "age_days", "list_saved", "describe"]

log = logging.getLogger(__name__)


def path_for(name: str) -> Path:
    """Return the parquet path backing the logical dataset ``name``."""
    if not name or name.startswith("/") or ".." in name.split("/"):
        raise ValueError(f"Invalid dataset name: {name!r}")
    return data_dir() / f"{name}.parquet"


def save(df: pd.DataFrame, name: str) -> Path:
    """Write ``df`` to ``data/<name>.parquet`` and return the path.

    The write goes to a temporary file first so an interrupted run cannot leave
    a half-written parquet behind that later reads would choke on.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"save() expects a DataFrame, got {type(df).__name__}")
    target = path_for(name)
    ensure_dir(target.parent)
    tmp = target.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False)
    except (ValueError, TypeError, pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        # Object columns holding mixed types cannot round-trip through Arrow;
        # coerce them to string rather than lose the whole frame.
        log.warning("Direct parquet write of %s failed (%s); stringifying object columns", name, exc)
        coerced = _stringify_objects(df)
        coerced.to_parquet(tmp, index=False)
    tmp.replace(target)
    return target


def _stringify_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with un-serialisable object columns cast to str."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    return out


def load(name: str) -> pd.DataFrame:
    """Read ``data/<name>.parquet``.

    Raises
    ------
    FileNotFoundError
        If the dataset has not been saved yet.
    """
    target = path_for(name)
    if not target.exists():
        raise FileNotFoundError(f"No cached dataset {name!r} at {target}")
    return pd.read_parquet(target)


def exists(name: str) -> bool:
    """Return whether ``data/<name>.parquet`` is present."""
    return path_for(name).exists()


def age_days(name: str) -> float | None:
    """Return the age of the cached dataset in days, or ``None`` if absent."""
    target = path_for(name)
    if not target.exists():
        return None
    return (time.time() - target.stat().st_mtime) / 86400.0


def list_saved() -> list[str]:
    """Return every logical dataset name currently stored, sorted."""
    root = data_dir()
    if not root.exists():
        return []
    names = [
        str(p.relative_to(root).with_suffix("")).replace("\\", "/")
        for p in root.rglob("*.parquet")
    ]
    return sorted(names)


def describe(name: str) -> dict[str, object]:
    """Return a small summary dict for a saved dataset.

    Never raises for a missing dataset; ``{"exists": False}`` is returned instead
    so summary tables can render partial state.
    """
    if not exists(name):
        return {"name": name, "exists": False, "rows": 0, "cols": 0, "age_days": None}
    frame = load(name)
    return {
        "name": name,
        "exists": True,
        "rows": int(len(frame)),
        "cols": int(frame.shape[1]),
        "age_days": age_days(name),
    }
