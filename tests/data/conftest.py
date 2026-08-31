"""Shared fixtures for the data-layer tests.

Every test that touches the parquet store is redirected to a tmp directory via
``FPLOPT_DATA_DIR`` so the real ``data/`` cache is never written or read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture_json(name: str) -> Any:
    """Read a JSON fixture from ``tests/data/fixtures/``."""
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def load_fixture_csv(name: str) -> pd.DataFrame:
    """Read a CSV fixture from ``tests/data/fixtures/``."""
    return pd.read_csv(FIXTURE_DIR / name, low_memory=False)


@pytest.fixture
def bootstrap_small() -> dict[str, Any]:
    """A 12-player slice of ``/bootstrap-static/`` covering all four positions."""
    return load_fixture_json("bootstrap_small.json")


@pytest.fixture
def fixtures_small() -> list[dict[str, Any]]:
    """Four real gameweek-1 fixtures including their nested ``stats`` blocks."""
    return load_fixture_json("fixtures_small.json")


@pytest.fixture
def live_gw1_small() -> dict[str, Any]:
    """A slice of ``/event/1/live/`` matching ``bootstrap_small``."""
    return load_fixture_json("live_gw1_small.json")


@pytest.fixture
def understat_match() -> dict[str, Any]:
    """A full real ``getMatchData`` payload (Arsenal vs Coventry, 2026/27)."""
    return load_fixture_json("understat_match_31180.json")


@pytest.fixture
def understat_league() -> dict[str, Any]:
    """A trimmed real ``getLeagueData`` payload for EPL 2026."""
    return load_fixture_json("understat_league_2026_small.json")


@pytest.fixture
def merged_gw_2016() -> pd.DataFrame:
    """Raw ``merged_gw.csv`` rows in the 2016-17 schema (no position/team/xP)."""
    return load_fixture_csv("merged_gw_2016-17.csv")


@pytest.fixture
def merged_gw_2026() -> pd.DataFrame:
    """Raw ``merged_gw.csv`` rows in the 2026-27 schema (defensive stats present)."""
    return load_fixture_csv("merged_gw_2026-27.csv")


@pytest.fixture
def players_raw_2016() -> pd.DataFrame:
    """Raw ``players_raw.csv`` rows for 2016-17, used for position backfill."""
    return load_fixture_csv("players_raw_2016-17.csv")


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the whole data layer at a throwaway directory for one test."""
    monkeypatch.setenv("FPLOPT_DATA_DIR", str(tmp_path))
    return tmp_path
