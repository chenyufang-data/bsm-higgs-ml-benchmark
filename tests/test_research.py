"""Inspect real synthetic exports and execute research notebooks in fresh kernels."""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from hepml.adapters.research import (
    check_splits,
    compare_runs,
    feature_preview,
    inspect_inputs,
    read_settings,
    verify_stage,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def research_settings(tmp_path, prepared_dataset):
    compact = prepared_dataset.parent / "compact"
    signal, background = tmp_path / "signal", tmp_path / "background"
    signal.mkdir()
    background.mkdir()
    for path in compact.iterdir():
        shutil.copy2(path, (signal if path.name.startswith("signal_") else background) / path.name)
    settings = {
        "study": str(REPO / "studies/cg_bbc"), "signal_dir": str(signal), "background_dir": str(background),
        "workspace": str(tmp_path / "research"), "mass": 200, "benchmark_id": "benchmark", "run_id": "xgb",
        "preview_rows_per_sample": 35,
        "training_args": ["--n-estimators", "12", "--early-stopping-rounds", "3"],
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    return path


def test_inventory_and_bounded_preview(research_settings):
    config = read_settings(research_settings)
    inventory, records = inspect_inputs(config)
    assert inventory.n_root.tolist() == [2000, 2000]
    assert inventory.n_selected.tolist() == [1600, 1600]
    assert inventory.selected_yield.tolist() == pytest.approx([4800, 48000])
    preview = feature_preview(config, records)
    assert preview.groupby("sample").size().tolist() == [35, 35]
    assert {"mcb1", "mcb2"}.issubset(preview.columns)
    assert preview.sample_weight.unique().tolist() == pytest.approx([3.0, 30.0])
    pd.testing.assert_frame_equal(preview, feature_preview(config, records))
    shard = next(Path(config["background_dir"]).glob("*.parquet"))
    shard.write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        inspect_inputs(config)


def test_split_assignment_mismatch_is_rejected(prepared_dataset):
    directory = prepared_dataset / "splits"
    assert len(check_splits(directory, 200)) == 6
    path = directory / "assignments_sig200.parquet"
    assignment = pd.read_parquet(path)
    assignment.loc[0, "event_id"] = assignment.loc[1, "event_id"]
    assignment.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="overlapping"):
        check_splits(directory, 200)


@pytest.mark.slow
def test_notebooks_execute_in_fresh_kernels(research_settings, tmp_path, monkeypatch, prepared_dataset):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    # Use the current test interpreter, not a global/user kernel installation.
    kernel = tmp_path / "kernels/hepml-test"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "hepml-test", "language": "python",
    }))
    monkeypatch.setenv("HEPML_RESEARCH_CONFIG", str(research_settings))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    for path in sorted((REPO / "notebooks").glob("*.ipynb")):
        if path.name in {"02_bdt_baselines.ipynb", "03_parameterized_bdt.ipynb", "04_physics_results.ipynb",
                         "05_truth_tagging.ipynb", "06_jet_networks.ipynb"}:
            # The study integration tests create their saved results and
            # execute these read-only reports in separate fresh kernels.
            continue
        notebook = nbformat.read(path, as_version=4)
        assert all(not cell.get("outputs") and cell.get("execution_count") is None for cell in notebook.cells)
        for cell in notebook.cells:
            if "actions" in cell.metadata.get("tags", []):
                cell.source = "START_PREPARATION = True\nSTART_TRAINING = True\nWAIT_FOR_COMPLETION = True"
        manager = KernelManager(kernel_name="hepml-test", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
        nbclient.NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(REPO)}}).execute()
        nbformat.write(notebook, tmp_path / path.name)
    config = read_settings(research_settings)
    # Separating input folders must not change event assignments or weights.
    for split in ("train", "val", "test"):
        original = pd.read_parquet(prepared_dataset / "splits" / f"{split}_sig200.parquet")
        separated = pd.read_parquet(Path(config["workspace"]) / "datasets/benchmark/data/splits" / f"{split}_sig200.parquet")
        pd.testing.assert_frame_equal(original, separated)
    run = Path(config["workspace"]) / "runs/xgb"
    assert verify_stage(run)["status"] == "complete"
    comparison = compare_runs([run], split="test")
    assert comparison.threshold_from.tolist() == ["validation"]
    assert comparison.split.tolist() == ["test"]
    metrics = json.loads((run / "models/sig200/metrics.json").read_text())
    for split in ("train", "val", "test"):
        row = compare_runs([run], split=split).iloc[0]
        result = metrics[split]["at_validation_threshold"]
        assert row.signal_yield_full == pytest.approx(result["signal_yield"])
        assert row.background_yield_full == pytest.approx(result["background_yield"])
        assert row.significance_full == pytest.approx(result["significance"])
        assert row.auc_weighted == pytest.approx(metrics[split]["auc_weighted"])
    provenance = json.loads((run / "provenance.json").read_text())
    assert provenance["sources"] and provenance["packages"] and provenance["settings"]["benchmark_artifacts"]
    with pytest.raises(ValueError, match="different benchmark"):
        other = tmp_path / "different-run"
        shutil.copytree(run, other)
        other_provenance = json.loads((other / "provenance.json").read_text())
        other_provenance["settings"]["benchmark_artifacts"] = {"other": "hash"}
        (other / "provenance.json").write_text(json.dumps(other_provenance))
        # Update only the status hash so comparison reaches the benchmark check.
        from hepml_compact.parquet_writer import sha256_file

        status = json.loads((other / "status.json").read_text())
        status["artifacts"]["provenance.json"] = sha256_file(other / "provenance.json")
        (other / "status.json").write_text(json.dumps(status))
        compare_runs([run, other])
