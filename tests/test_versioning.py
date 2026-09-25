"""Release-version allocation and resolution."""

from pathlib import Path

import pytest

from hepml.adapters.releases import latest_final_version, next_version, resolve_model_dir


def test_next_version_starts_at_v1(tmp_path):
    assert next_version(tmp_path, "sig200").name == "sig200_v1"


def test_next_version_increments_past_gaps(tmp_path):
    (tmp_path / "sig200_v1").mkdir()
    (tmp_path / "sig200_v3").mkdir()
    assert next_version(tmp_path, "sig200").name == "sig200_v4"


def test_next_version_tolerates_missing_base(tmp_path):
    assert next_version(tmp_path / "does-not-exist", "sig200").name == "sig200_v1"


def test_latest_final_version(tmp_path):
    assert latest_final_version(tmp_path, 200) is None
    (tmp_path / "sig200_v1").mkdir()
    (tmp_path / "sig200_v2").mkdir()
    (tmp_path / "sig400_v9").mkdir()  # other mass must not interfere
    assert latest_final_version(tmp_path, 200) == 2


def test_latest_final_version_missing_root(tmp_path):
    assert latest_final_version(tmp_path / "nope", 200) is None


def test_resolve_latest_falls_back_to_working_dir(tmp_path):
    resolved = resolve_model_dir(tmp_path / "final", 200, "latest", working_root=tmp_path / "work")
    assert resolved == tmp_path / "work" / "sig200"


def test_resolve_latest_picks_highest(tmp_path):
    (tmp_path / "sig200_v1").mkdir()
    (tmp_path / "sig200_v7").mkdir()
    assert resolve_model_dir(tmp_path, 200, "latest") == tmp_path / "sig200_v7"


def test_resolve_explicit_version(tmp_path):
    assert resolve_model_dir(tmp_path, 200, 3) == tmp_path / "sig200_v3"
    assert resolve_model_dir(tmp_path, 200, " 3 ") == tmp_path / "sig200_v3"


def test_resolve_rejects_garbage():
    with pytest.raises(SystemExit):
        resolve_model_dir(Path("x"), 200, "newest")
