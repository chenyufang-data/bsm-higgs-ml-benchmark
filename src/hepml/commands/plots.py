#!/usr/bin/env python3
"""Generate diagnostics for a working XGBoost model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hepml.adapters.plots import (
    _plot_permutation_importance,
    _plot_roc,
    _plot_score_hists,
    _plot_shap_summary,
    _plot_significance,
    _read_preds,
)

try:
    import shap
except ImportError:
    shap = None

from hepml.adapters.configuration import output_paths
from hepml.log import get_logger

log = get_logger(__name__)


# --------------------------------------
# Main
# --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Diagnostic plots for a trained BDT model.")
    ap.add_argument("--mass", type=int, default=200, help="e.g. 200")
    ap.add_argument(
        "--modeldir", default=None, help="Base directory where sig{mass}/ exists (default: <output-root>/<study>/runs/)"
    )
    ap.add_argument("--features", nargs="+", default=None, help="Override feature list")
    args = ap.parse_args(argv)

    base_dir = Path(args.modeldir) if args.modeldir is not None else output_paths().models_work
    mdir = base_dir / f"sig{args.mass}"
    if not mdir.exists():
        raise FileNotFoundError(f"Missing directory: {mdir}")

    # load feature list from metrics.json if not given
    metrics_path = mdir / "metrics.json"
    if args.features is None:
        with open(metrics_path) as f:
            metrics = json.load(f)
        feature_names = metrics["features"]
    else:
        feature_names = args.features

    train_df = _read_preds(mdir / "preds_train.parquet")
    val_df = _read_preds(mdir / "preds_val.parquet")
    test_df = _read_preds(mdir / "preds_test.parquet")

    outdir = mdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    _plot_score_hists(train_df, test_df, outdir / "scores_train_vs_test.png")
    _plot_roc(val_df, outdir / "roc_val.png", "ROC (validation)")
    _plot_roc(test_df, outdir / "roc_test.png", "ROC (test)")

    best_val, _ = _plot_significance(mdir, outdir / "Zscan_val.png", "Weighted significance scan (val)", "val")
    best_tst, _ = _plot_significance(mdir, outdir / "Zscan_test.png", "Weighted significance scan (test)", "test")

    model_path = mdir / "model.ubj"

    try:
        _plot_permutation_importance(test_df, feature_names, outdir)
    except Exception as e:
        log.warning("Permutation importance skipped due to error: %s", e)

    try:
        _plot_shap_summary(model_path, test_df, feature_names, outdir / "shap_summary.png")
    except Exception as e:
        log.warning("SHAP summary skipped due to error: %s", e)

    log.info("Saved plots to: %s", outdir)
    log.info(
        "[val]  best thr=%.3f  Z=%.4f  S=%.3f  B=%.3g",
        best_val["best_thr"],
        best_val["best_Z"],
        best_val["best_S"],
        best_val["best_B"],
    )
    log.info(
        "[test] best thr=%.3f  Z=%.4f  S=%.3f  B=%.3g",
        best_tst["best_thr"],
        best_tst["best_Z"],
        best_tst["best_S"],
        best_tst["best_B"],
    )


if __name__ == "__main__":
    main()
