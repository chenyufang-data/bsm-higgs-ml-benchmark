"""Small saved study -> fresh-kernel HTML report -> multi-objective release."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hepml_compact.parquet_writer import sha256_file

from hepml.cli import main
from hepml.commands.shape_audit import load_anchor

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_evaluation_study_report_and_frozen_objectives(tmp_path, prepared_dataset, monkeypatch):
    pytest.importorskip("nbclient")
    pytest.importorskip("nbconvert")
    import nbformat

    monkeypatch.setenv("MPLBACKEND", "Agg")
    study = tmp_path / "study"
    shutil.copytree(REPO / "studies/cg_bbc", study)
    analysis = yaml.safe_load((study / "analysis.yaml").read_text())
    analysis["training"].update(n_estimators=12, early_stopping_rounds=3)
    analysis["evaluation"]["min_background_neff"] = 2
    (study / "analysis.yaml").write_text(yaml.safe_dump(analysis))
    config = yaml.safe_load((study / "validation.yaml").read_text())
    config["evaluation_study"].update(prior_audit_masses=[200], seeds=[42], bootstrap_replicates=4)
    (study / "validation.yaml").write_text(yaml.safe_dump(config))
    compact = tmp_path / "compact"
    shutil.copytree(prepared_dataset.parent / "compact", compact / "rho01")
    registry = tmp_path / "registry"
    registry.mkdir()
    load_anchor(prepared_dataset, 200).to_parquet(registry / "assignments.parquet", index=False)
    (registry / "registry.json").write_text(json.dumps({"assignments_sha256": sha256_file(registry / "assignments.parquet")}))
    output, runs = tmp_path / "report", tmp_path / "runs"
    main(["evaluation-study", "--study", str(study), "--benchmark", str(prepared_dataset),
          "--registry-dir", str(registry), "--compact-root", str(compact), "--runs-dir", str(runs),
          "--run-prefix", "study", "--outdir", str(output)])
    assert json.loads((output / "provenance/status.json").read_text())["status"] == "complete"
    assert (output / "report.ipynb").is_file() and (output / "report.html").is_file()
    assert "data:image/png;base64" in (output / "report.html").read_text(encoding="utf-8")
    notebook = nbformat.read(output / "report.ipynb", as_version=4)
    figures = [item for cell in notebook.cells for item in cell.get("outputs", [])
               if "image/png" in item.get("data", {})]
    assert len(figures) == 4  # Curves and stability for both fitting policies.
    assert (output / "provenance/sources.zip").is_file()
    assert not (output / "provenance/sources").exists()
    points = pd.read_csv(output / "tables/operating_points.csv")
    assert len(points) == 6
    assert points.eligible.all()
    assert set(points.objective) == {"Z_Asimov", "Z_syst_5pct", "Z_syst_10pct"}
    models = runs / "study-none-seed42/models"
    final = tmp_path / "releases"
    with pytest.raises(ValueError, match="test selection"):
        main(["freeze", "--workdir", str(models), "--finaldir", str(final), "--use-threshold-from", "test"])
    main(["freeze", "--workdir", str(models), "--finaldir", str(final)])
    release = final / "sig200_v1"
    record = json.loads((release / "threshold.json").read_text())
    assert len(record["evaluation"]["operating_points"]) == 3
    main(["predict", "--model-dir", str(final), "--input", str(prepared_dataset / "splits/val_sig200.parquet"),
          "--split", "val", "--objective", "Z_syst_10pct"])
    prediction = pd.read_parquet(release / "preds_infer_val.parquet")
    scores = pd.read_parquet(models / "sig200/preds_val.parquet")
    np.testing.assert_allclose(prediction.bdt_score, scores.bdt_score, atol=2e-6)
    for objective, point in record["evaluation"]["operating_points"].items():
        np.testing.assert_array_equal(prediction[f"bdt_pred_{objective}"], prediction.bdt_score >= point["threshold"])
    pred_meta_path = release / "preds_infer_val.meta.json"
    pred_meta = json.loads(pred_meta_path.read_text())
    pred_meta["split_meta"]["signal"]["xs_by_rho"] = {"0.1": 1., "0.2": 4.}
    pred_meta_path.write_text(json.dumps(pred_meta))
    main(["summarize", "--model-dir", str(final), "--split", "val"])
    report = json.loads((release / "report_infer_val.json").read_text())
    assert report['yields_weighted']['k_factors'] == {'signal': 1.95, 'background': 1.26}
    assert report["objective"] == "Z_syst_10pct"
    assert report["threshold"] == record["evaluation"]["operating_points"]["Z_syst_10pct"]["threshold"]
    assert "rho_scan" not in report  # Version 2 must never run the legacy unconstrained rescan.
    assert "Frozen validation operating points" in (release / "report_infer_val.md").read_text()
    assert len(report["operating_points"]) == 3
    for point in report["operating_points"]:
        reference = record["evaluation"]["operating_points"][point["objective"]]["metrics"]
        assert point["S"] == pytest.approx(reference["S"])
        assert point["B"] == pytest.approx(reference["B"])
    assert not (models / "sig200/preds_test.parquet").exists()
