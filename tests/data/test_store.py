"""Tests for the parquet cache helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from fplopt.data import paths, store


def test_save_load_roundtrip(tmp_data_dir: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    written = store.save(frame, "unit/roundtrip")
    assert written.exists()
    assert written == tmp_data_dir / "unit" / "roundtrip.parquet"
    pd.testing.assert_frame_equal(store.load("unit/roundtrip"), frame)


def test_exists_and_age(tmp_data_dir: Path) -> None:
    assert store.exists("unit/absent") is False
    assert store.age_days("unit/absent") is None

    store.save(pd.DataFrame({"a": [1]}), "unit/present")
    assert store.exists("unit/present") is True
    age = store.age_days("unit/present")
    assert age is not None and 0 <= age < 1


def test_load_missing_raises(tmp_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.load("unit/never-written")


def test_save_rejects_non_dataframe(tmp_data_dir: Path) -> None:
    with pytest.raises(TypeError):
        store.save({"a": [1]}, "unit/bad")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "/absolute", "../escape", "a/../../b"])
def test_path_for_rejects_traversal(name: str) -> None:
    with pytest.raises(ValueError):
        store.path_for(name)


def test_save_coerces_unserialisable_objects(tmp_data_dir: Path) -> None:
    """A column of mixed dicts/lists must not lose the whole frame."""
    frame = pd.DataFrame({"id": [1, 2], "blob": [{"k": 1}, [1, 2, 3]]})
    store.save(frame, "unit/objects")
    out = store.load("unit/objects")
    assert len(out) == 2
    assert out["blob"].map(type).eq(str).all()


def test_list_saved_and_describe(tmp_data_dir: Path) -> None:
    store.save(pd.DataFrame({"a": [1, 2]}), "unit/one")
    store.save(pd.DataFrame({"a": [1]}), "unit/two")
    assert store.list_saved() == ["unit/one", "unit/two"]

    described = store.describe("unit/one")
    assert described["exists"] is True
    assert described["rows"] == 2
    assert described["cols"] == 1

    missing = store.describe("unit/nope")
    assert missing["exists"] is False
    assert missing["rows"] == 0


def test_data_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLOPT_DATA_DIR", str(tmp_path))
    assert paths.data_dir() == tmp_path.resolve()
    assert paths.cache_dir() == tmp_path.resolve() / "cache"
    monkeypatch.delenv("FPLOPT_DATA_DIR")
    assert paths.data_dir() == paths.project_root() / "data"


def test_save_is_atomic_no_tmp_left(tmp_data_dir: Path) -> None:
    store.save(pd.DataFrame({"a": [1]}), "unit/atomic")
    leftovers = list((tmp_data_dir / "unit").glob("*.tmp"))
    assert leftovers == []
