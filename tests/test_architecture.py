"""Keep the dependency boundary and model-independent evaluation usable."""

import ast
import json
import shutil
from pathlib import Path

import numpy as np

from hepml import cli
from hepml.adapters.configuration import output_paths
from hepml.application.training import evaluate_scores, fit_and_score
from hepml.domain.metrics import weighted_significance_scan

PACKAGE = Path(__file__).parents[1] / "src" / "hepml"


def test_inner_layers_do_not_import_infrastructure_or_a_specific_study():
    allowed = {"hepml.domain", "hepml.application", "hepml.ports", "hepml.log"}
    paths = [PACKAGE / "ports.py", *(PACKAGE / "domain").glob("*.py"), *(PACKAGE / "application").glob("*.py")]
    forbidden = {"uproot", "pyarrow", "xgboost", "torch", "yaml", "joblib", "argparse"}
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for module in modules:
                assert module.split(".")[0] not in forbidden | {"studies"}, (path, module)
                if module.startswith("hepml."):
                    assert any(module == name or module.startswith(name + ".") for name in allowed), (path, module)


def test_model_port_preserves_input_and_fit_weights_and_shared_evaluation():
    X = np.array([[0.2], [0.4], [0.6], [0.8]], dtype=np.float32)
    y = np.array([0, 1, 0, 1])
    weights = np.array([100.0, 10.0, 100.0, 10.0])

    class FixedScoreModel:
        def predict_proba(self, values):
            return np.column_stack([1 - values[:, 0], values[:, 0]])

    def trainer(train, target, val, val_target, **kwargs):
        assert train is X and val is X and target is y and val_target is y
        assert kwargs["fit_weights_train"] is weights
        return FixedScoreModel()

    _, (p_train, p_val, p_test) = fit_and_score(trainer, X, y, X, y, X, fit_weights_train=weights)
    np.testing.assert_array_equal(p_train, p_val)
    np.testing.assert_array_equal(p_val, p_test)
    result = evaluate_scores(
        y, p_test, weights, lumi=1.0, frac_sig=0.5, frac_bkg=0.5, scan=dict(n_steps=4, min_nS=0, min_nB=0, min_B=1.0)
    )
    assert result["auc_weighted"] == 0.75
    scan = result["weighted_significance_scan"]
    assert scan["best_nS"] == 40.0 and scan["best_nB"] == 200.0


def test_shared_scan_matches_numerical_reference():
    reference = json.loads((Path(__file__).parent / "fixtures" / "significance_reference.json").read_text())
    for case in reference:
        inputs = case["inputs"]
        result = weighted_significance_scan(
            np.array(inputs["y"]), np.array(inputs["score"]), np.array(inputs["weight"]), **case["options"]
        )
        assert result == case["result"]


def test_cli_study_paths_are_consistent_and_context_does_not_leak(tmp_path, monkeypatch, capsys):
    study = tmp_path / "custom"
    study.mkdir()
    shutil.copy(PACKAGE.parents[1] / "studies/cg_bbc/extraction.yaml", study / "extraction.yaml")
    path = study / "extraction.yaml"
    path.write_text(path.read_text().replace("name: cg_bbc", "name: alternate"))
    (study / "plugin.py").write_text(
        "FEATURES = ['x']\nRETAINED_COLUMNS = []\nREQUIRED_BRANCHES = []\nOPTIONAL_BRANCHES = []\n"
        "def derive_features(frame): return frame\n"
    )
    monkeypatch.setenv("HEPML_OUTPUT_ROOT", str(tmp_path / "outputs"))
    cli.main(["paths", "--study", str(study)])
    result = json.loads(capsys.readouterr().out)
    assert Path(result["models_work"]) == tmp_path / "outputs" / "alternate" / "runs"
    assert output_paths().models_work == tmp_path / "outputs" / "cg_bbc" / "runs"
