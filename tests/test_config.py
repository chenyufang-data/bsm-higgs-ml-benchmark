"""Config loading: .env seeding, sample manifest, output paths, analysis."""

import os
from pathlib import Path

import pytest
from hepml_compact.config import load_compact_sample

from hepml.adapters.configuration import AnalysisConfig, fit_parameters, load_analysis, load_dotenv, output_paths


# ---------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------
def test_load_dotenv_parses_and_applies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # record prior (absent) state so teardown removes what load_dotenv sets
    monkeypatch.delenv("HEPML_TEST_DOTENV", raising=False)
    (tmp_path / ".env").write_text(
        "# comment\n\nHEPML_TEST_DOTENV='quoted value'\nnot a key value line\n",
        encoding="utf-8",
    )
    applied = load_dotenv()
    assert applied == {"HEPML_TEST_DOTENV": "quoted value"}
    assert os.environ["HEPML_TEST_DOTENV"] == "quoted value"


def test_load_dotenv_does_not_override_real_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPML_TEST_KEEP", "from-environment")
    (tmp_path / ".env").write_text("HEPML_TEST_KEEP=from-dotenv\n", encoding="utf-8")
    applied = load_dotenv()
    assert "HEPML_TEST_KEEP" not in applied
    assert os.environ["HEPML_TEST_KEEP"] == "from-environment"


def test_load_dotenv_missing_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_dotenv() == {}


# ---------------------------------------------------------------
# output_paths
# ---------------------------------------------------------------
def test_output_paths_default_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEPML_OUTPUT_ROOT", raising=False)
    out = output_paths()
    assert out.base == Path("outputs")
    assert out.root_parquet == Path("outputs/cg_bbc/compact")
    assert out.splits == Path("outputs/cg_bbc/datasets/splits")
    assert out.models_work == Path("outputs/cg_bbc/runs")
    assert out.models_final == Path("outputs/cg_bbc/releases")
    assert out.ablation == Path("outputs/cg_bbc/ablation")


def test_output_paths_env_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPML_OUTPUT_ROOT", str(tmp_path / "elsewhere"))
    out = output_paths()
    assert out.base == tmp_path / "elsewhere"
    assert out.splits == tmp_path / "elsewhere" / "cg_bbc" / "datasets" / "splits"


def test_output_paths_explicit_base_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HEPML_OUTPUT_ROOT", str(tmp_path / "env-root"))
    out = output_paths(tmp_path / "explicit")
    assert out.base == tmp_path / "explicit"


# ---------------------------------------------------------------
# load_analysis
# ---------------------------------------------------------------
def test_load_analysis_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_analysis()
    assert cfg == AnalysisConfig()
    assert cfg.lumi == 3000.0
    assert cfg.training.n_estimators == 1200


def test_load_analysis_yaml_overrides(tmp_path):
    p = tmp_path / "analysis.yaml"
    p.write_text(
        "physics:\n  lumi: 300\ntraining:\n  n_estimators: 10\n  scan:\n    min_B: 1.5\n",
        encoding="utf-8",
    )
    cfg = load_analysis(p)
    assert cfg.lumi == 300.0
    assert cfg.training.n_estimators == 10
    assert cfg.training.scan.min_B == 1.5
    # untouched values keep their defaults
    assert cfg.training.max_depth == 4


def test_load_analysis_explicit_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_analysis(tmp_path / "nope.yaml")


def test_compact_sample_roots_and_new_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "samples.yaml"
    manifest.write_text("data_root: production\nsamples:\n  signal:\n    kind: signal\n    files: [new.root]\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEPML_DATA_ROOT", raising=False)
    inputs, sample = load_compact_sample(manifest, "signal")
    assert inputs == [tmp_path / "production" / "new.root"]
    assert sample["kind"] == "signal"
    monkeypatch.setenv("HEPML_DATA_ROOT", str(tmp_path / "environment"))
    assert load_compact_sample(manifest, "signal")[0] == [tmp_path / "environment" / "new.root"]
    assert load_compact_sample(manifest, "signal", str(tmp_path / "cli"))[0] == [tmp_path / "cli" / "new.root"]


def test_empty_or_invalid_production_manifest_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "samples.yaml"
    manifest.write_text("samples: {}\n")
    with pytest.raises(ValueError, match="not configured"):
        load_compact_sample(manifest, "sig_200")
    manifest.write_text("samples:\n  signal:\n    kind: signal\n    files: []\n")
    with pytest.raises(ValueError, match="nonempty files"):
        load_compact_sample(manifest, "signal")


def test_selection_is_not_duplicated_in_analysis_config(tmp_path):
    path = tmp_path / "analysis.yaml"
    path.write_text("selection: {b_pt: 30}\n")
    with pytest.raises(ValueError, match="extraction.yaml"):
        load_analysis(path)


def test_evaluation_config_is_typed_and_validation_only(tmp_path):
    path = tmp_path / "analysis.yaml"
    path.write_text("evaluation:\n  lumi_pb_inv: 500000\n  min_background_neff: 200\n")
    cfg = load_analysis(path)
    assert cfg.evaluation.lumi_pb_inv == 500000
    assert cfg.evaluation.min_background_neff == 200
    path.write_text("evaluation: {threshold_source: test}\n")
    with pytest.raises(ValueError, match="validation"):
        load_analysis(path)
    path.write_text("evaluation: {typo: 1}\n")
    with pytest.raises(TypeError):
        load_analysis(path)


def test_fit_parameters_apply_only_recorded_typed_overrides():
    analysis = AnalysisConfig()
    base = fit_parameters(analysis)
    assert "seed" not in base and "scan" not in base and base["max_depth"] == analysis.training.max_depth
    deeper = fit_parameters(analysis, ["max_depth=5", "learning_rate=0.1"])
    assert deeper == dict(base, max_depth=5, learning_rate=0.1) and isinstance(deeper["learning_rate"], float)
    assert fit_parameters(analysis, ["min_child_weight=2"])["min_child_weight"] == 2.0
    for bad in ["seed=7", "scan=1", "depth=5", "max_depth", "max_depth=5.5", "max_depth=true", "gamma=x"]:
        with pytest.raises(ValueError):
            fit_parameters(analysis, [bad])
