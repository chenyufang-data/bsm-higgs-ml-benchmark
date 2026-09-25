"""Synthetic compact -> prepare -> train -> release/inference regression tests."""

import json

import numpy as np
import pandas as pd
import pytest

from hepml.commands import freeze as freeze_final
from hepml.commands import plots, predict, summarize
from hepml.commands import train as train_bdt

pytestmark = pytest.mark.slow


def test_pipeline_end_to_end(tmp_path, prepared_dataset):
    ml_out = prepared_dataset
    models = tmp_path / "models"
    final = tmp_path / "final"
    splits = ml_out / "splits"

    assert (ml_out / "dataset_sig200_vs_bkg.parquet").exists()
    assert (splits / "test_sig200.parquet").exists()
    split_meta = json.loads((splits / "split_sig200.meta.json").read_text())
    assert split_meta["signal"]["xs_pb"] == pytest.approx(2.0)
    assert "synthetic_bkg" in split_meta["backgrounds"]

    # weights are uniform per sample: yield = lumi * xs * n_passed / n_total
    df = pd.read_parquet(ml_out / "dataset_sig200_vs_bkg.parquet")
    sig = df[df["target"] == 1]
    assert sig["sample_weight"].nunique() == 1
    expected_yield = split_meta["lumi"] * 2.0 * len(sig) / split_meta["signal"]["n_events_total"]
    assert sig["sample_weight"].sum() == pytest.approx(expected_yield, rel=1e-9)

    # 3) train (small model so the test stays fast; early stopping active)
    train_bdt.main(
        [
            "--splits-dir",
            str(splits),
            "--outdir",
            str(models),
            "--mass",
            "200",
            "--n-estimators",
            "60",
            "--early-stopping-rounds",
            "10",
            "--lumi",
            "6000",
        ]
    )
    work = models / "sig200"
    assert (work / "model.ubj").exists()
    assert (work / "metrics.json").exists()
    metrics = json.loads((work / "metrics.json").read_text())
    from hepml.domain.metrics import normalized_weights

    for split in ("train", "val", "test"):
        predictions = pd.read_parquet(work / f"preds_{split}.parquet")
        original = pd.read_parquet(splits / f"{split}_sig200.parquet")
        pd.testing.assert_series_equal(predictions.sample_weight, original.sample_weight)
        normalization = metrics[split]["normalization"]
        expected = predictions.assign(w=normalized_weights(predictions, normalization)).groupby("sample").w.sum()
        assert expected.to_dict() == pytest.approx(df.groupby("sample").sample_weight.sum().to_dict())
        assert metrics[split]["at_validation_threshold"]["threshold"] == metrics["val"]["weighted_significance_scan"]["best_thr"]

    plots.main(["--modeldir", str(models), "--mass", "200"])

    # 4) freeze release v1
    freeze_final.main(
        [
            "--workdir",
            str(models),
            "--finaldir",
            str(final),
            "--mass",
            "200",
        ]
    )
    release = final / "sig200_v1"
    assert (release / "features.txt").exists()
    thr_json = (release / "threshold.json").read_text()
    assert '"best_iteration"' in thr_json

    # 5) predict on the test split with the frozen model
    predict.main(
        [
            "--model-dir",
            str(final),
            "--mass",
            "200",
            "--input",
            str(splits / "test_sig200.parquet"),
            "--split",
            "test",
        ]
    )
    preds_path = release / "preds_infer_test.parquet"
    assert preds_path.exists()

    # 6) SKEW REGRESSION: frozen-model scores must match training-time scores
    infer = pd.read_parquet(preds_path)
    trained = pd.read_parquet(work / "preds_test.parquet")
    assert len(infer) == len(trained)
    assert np.allclose(
        infer["bdt_score"].to_numpy(),
        trained["bdt_score"].to_numpy(),
        rtol=0,
        atol=2e-6,
    ), "frozen-model inference diverged from training-time scores (iteration_range regression)"

    summarize.main(["--model-dir", str(final), "--mass", "200"])
    assert (release / "report_infer_test.json").exists()
    assert (release / "report_infer_test.md").exists()
    report = json.loads((release / "report_infer_test.json").read_text())
    yields = report["yields_weighted"]
    assert yields["normalization"]["method"] == "per_sample_full_prepared_v1"
    assert yields["S_pass_full"] == pytest.approx(metrics["test"]["at_validation_threshold"]["signal_yield"])
    assert yields["B_pass_full"] == pytest.approx(metrics["test"]["at_validation_threshold"]["background_yield"])
    lumi_scale = 6000 / split_meta["lumi"]
    assert yields["evaluation_lumi_pb_inv"] == 6000
    assert yields["S_all_full"] == pytest.approx(df.loc[df.target == 1, "sample_weight"].sum() * lumi_scale)
    assert yields["B_all_full"] == pytest.approx(df.loc[df.target == 0, "sample_weight"].sum() * lumi_scale)


def test_balanced_mixture_training(tmp_path, prepared_dataset):
    """--weight-mode balanced-mixture trains and records its mode."""
    ml_out = prepared_dataset
    models = tmp_path / "models"

    train_bdt.main(
        [
            "--splits-dir",
            str(ml_out / "splits"),
            "--outdir",
            str(models),
            "--mass",
            "200",
            "--n-estimators",
            "30",
            "--early-stopping-rounds",
            "10",
            "--weight-mode",
            "balanced-mixture",
        ]
    )
    metrics = json.loads((models / "sig200" / "metrics.json").read_text())
    assert metrics["xgb"]["weight_mode"] == "balanced-mixture"
