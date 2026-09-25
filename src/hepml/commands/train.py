#!/usr/bin/env python3
"""Compose XGBoost training with shared fitting and physics evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hepml.adapters.configuration import load_analysis, output_paths
from hepml.adapters.dataset_files import load_split
from hepml.adapters.xgboost_model import train_xgb
from hepml.application.training import (
    build_fit_weights,
    compute_scale_pos_weight,
    evaluate_scores,
    fit_and_score,
    make_xyw,
)
from hepml.domain.artifacts import dataset_filename, model_dirname, split_filename, split_meta_path
from hepml.domain.metrics import normalized_weights, sample_normalization, threshold_yields
from hepml.domain.validation import sanity_prepared_ml
from hepml.log import get_logger
from studies.cg_bbc.features import DEFAULT_FEATURES

log = get_logger(__name__)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------


# --------------------------------------
# Main
# --------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the XGBoost BDT on prepared splits.")
    ap.add_argument(
        "--splits-dir",
        default=None,
        help="Directory containing train/val/test parquet (default: <output-root>/<study>/datasets/splits/)",
    )
    ap.add_argument("--mass", type=int, default=200, help="Signal mass tag, e.g. 200")
    ap.add_argument("--outdir", default=None, help="Base output directory (default: <output-root>/<study>/runs/)")
    ap.add_argument(
        "--analysis-config",
        default=None,
        help="Analysis config YAML (default: studies/cg_bbc/analysis.yaml if present)",
    )
    ap.add_argument(
        "--study", default=None, help="Study directory: select its features and namespace default outputs by study name"
    )
    ap.add_argument("--features", nargs="+", default=None, help="Feature columns (default: selected study's features)")
    ap.add_argument("--add-features", nargs="+", default=[], help="Extra features to append")
    ap.add_argument("--drop-features", nargs="+", default=[], help="Features to drop")

    # training hyperparams (None -> value from analysis config)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--subsample", type=float, default=None)
    ap.add_argument("--colsample-bytree", type=float, default=None)
    ap.add_argument("--reg-lambda", type=float, default=None)
    ap.add_argument("--reg-alpha", type=float, default=None)
    ap.add_argument("--min-child-weight", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--early-stopping-rounds", type=int, default=None)
    ap.add_argument("--tree-method", default="hist", choices=["hist", "approx", "auto"], help="XGBoost tree method")

    # balancing & evaluation
    ap.add_argument("--class-balance", action="store_true", help="Use scale_pos_weight = Nneg/Npos")
    ap.add_argument(
        "--weight-mode",
        choices=["none", "balanced-mixture"],
        default="none",
        help="Training weights: 'none' (unweighted, default) or 'balanced-mixture' "
        "(physical process proportions within each class, classes balanced)",
    )
    ap.add_argument("--signif-steps", type=int, default=None, help="Threshold scan steps for weighted significance")
    ap.add_argument("--lumi", type=float, default=None, help="Luminosity scale factor for significance/yields")
    ap.add_argument("--min-nS", type=int, default=None)
    ap.add_argument("--min-nB", type=int, default=None)
    ap.add_argument("--min-B", type=float, default=None)
    ap.add_argument("--eps-plateau", type=float, default=None)

    args = ap.parse_args(argv)

    cfg = load_analysis(
        args.analysis_config
        or (
            str(Path(args.study) / "analysis.yaml")
            if args.study and (Path(args.study) / "analysis.yaml").exists()
            else None
        )
    )
    tr = cfg.training
    scan_cfg = tr.scan

    def opt(cli_value, cfg_value):
        return cli_value if cli_value is not None else cfg_value

    seed = opt(args.seed, tr.seed)
    lumi = opt(args.lumi, cfg.lumi)
    scan_kwargs = dict(
        n_steps=opt(args.signif_steps, scan_cfg.n_steps),
        min_nS=opt(args.min_nS, scan_cfg.min_nS),
        min_nB=opt(args.min_nB, scan_cfg.min_nB),
        min_B=opt(args.min_B, scan_cfg.min_B),
        eps_plateau=opt(args.eps_plateau, scan_cfg.eps_plateau),
    )

    out = output_paths()
    study = None
    if args.study:
        from hepml.adapters.study_loader import load_study

        study = load_study(args.study)
        out = output_paths(study=study.name)
    splits_dir = Path(args.splits_dir) if args.splits_dir is not None else out.splits
    base_outdir = Path(args.outdir) if args.outdir is not None else out.models_work
    outdir = base_outdir / model_dirname(args.mass)
    outdir.mkdir(parents=True, exist_ok=True)

    default_features = list(study.plugin.FEATURES) if study else DEFAULT_FEATURES
    features = (args.features if args.features is not None else default_features) + args.add_features
    features = [f for f in features if f not in args.drop_features]

    all_path = splits_dir.parent / dataset_filename(args.mass)
    train_path = splits_dir / split_filename("train", args.mass)
    val_path = splits_dir / split_filename("val", args.mass)
    test_path = splits_dir / split_filename("test", args.mass)

    df_all = load_split(all_path)
    df_train = load_split(train_path)
    df_val = load_split(val_path)
    df_test = load_split(test_path)

    frames = dict(train=df_train, val=df_val, test=df_test)
    normalizations = {name: sample_normalization(df_all, frame) for name, frame in frames.items()}

    if not (set(df_train.columns) == set(df_val.columns) == set(df_test.columns)):
        raise ValueError("Train/val/test column mismatch")

    # sanity_prepared_ml recomputes sample_weight from lumi; use the lumi the
    # splits were actually prepared with (recorded in the split meta), not the
    # module default, or preparing with a non-default --lumi breaks training.
    meta_path = split_meta_path(splits_dir, args.mass)
    sanity_lumi = lumi
    if meta_path.exists():
        try:
            sanity_lumi = float(json.loads(meta_path.read_text()).get("lumi", lumi))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            log.warning("Could not read lumi from %s (%s); using lumi=%s", meta_path, e, lumi)

    sanity_prepared_ml(df_train, features=features, name="train", lumi=sanity_lumi)
    sanity_prepared_ml(df_val, features=features, name="val", lumi=sanity_lumi)
    sanity_prepared_ml(df_test, features=features, name="test", lumi=sanity_lumi)

    # Stored physical weights include the preparation luminosity. Rescale only
    # the evaluation arrays if a different target luminosity was requested.
    eval_weights = {
        name: normalized_weights(frame, normalizations[name]) * lumi / sanity_lumi
        for name, frame in frames.items()
    }

    X_train, y_train, _ = make_xyw(df_train, features)
    X_val, y_val, _ = make_xyw(df_val, features)
    X_test, y_test, _ = make_xyw(df_test, features)

    spw = compute_scale_pos_weight(y_train) if args.class_balance else None
    fit_w_train = build_fit_weights(df_train, args.weight_mode)
    fit_w_val = build_fit_weights(df_val, args.weight_mode)

    model, (p_train, p_val, p_test) = fit_and_score(
        train_xgb,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        fit_weights_train=fit_w_train,
        fit_weights_val=fit_w_val,
        seed=seed,
        max_depth=opt(args.max_depth, tr.max_depth),
        n_estimators=opt(args.n_estimators, tr.n_estimators),
        learning_rate=opt(args.learning_rate, tr.learning_rate),
        subsample=opt(args.subsample, tr.subsample),
        colsample_bytree=opt(args.colsample_bytree, tr.colsample_bytree),
        reg_lambda=opt(args.reg_lambda, tr.reg_lambda),
        reg_alpha=opt(args.reg_alpha, tr.reg_alpha),
        min_child_weight=opt(args.min_child_weight, tr.min_child_weight),
        gamma=opt(args.gamma, tr.gamma),
        early_stopping_rounds=opt(args.early_stopping_rounds, tr.early_stopping_rounds),
        scale_pos_weight=spw,
        tree_method=args.tree_method,
    )

    metrics = {
        "mass": args.mass,
        "features": features,
        "lumi_pb_inv": float(lumi),
        "train": evaluate_scores(y_train, p_train, eval_weights["train"], lumi=lumi),
        "val": evaluate_scores(y_val, p_val, eval_weights["val"], lumi=lumi, scan=scan_kwargs),
        "test": evaluate_scores(y_test, p_test, eval_weights["test"], lumi=lumi, scan=scan_kwargs),
        "xgb": {
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            "params": model.get_params(),
            "scale_pos_weight": None if spw is None else float(spw),
            "weight_mode": args.weight_mode,
        },
    }

    threshold = metrics["val"]["weighted_significance_scan"]["best_thr"]
    for name, scores in zip(frames, (p_train, p_val, p_test)):
        metrics[name]["normalization"] = normalizations[name]
        metrics[name]["at_validation_threshold"] = threshold_yields(
            frames[name].target.to_numpy(), scores, eval_weights[name], threshold=threshold, lumi=lumi,
        )

    # save model
    model_path_joblib = outdir / "model.joblib"
    log.info("Saving model to %s", model_path_joblib)
    joblib.dump(model, model_path_joblib)

    model_path_ubj = outdir / "model.ubj"
    booster = model.get_booster()
    log.info("Saving native booster format to %s", model_path_ubj)
    booster.save_model(model_path_ubj.as_posix())

    # save metrics
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save preds for inspection
    def save_preds(df: pd.DataFrame, scores: np.ndarray, split: str):
        out = df.copy()
        out["bdt_score"] = scores.astype(np.float32)
        out_path = outdir / f"preds_{split}.parquet"
        out.to_parquet(out_path, index=False)
        return out_path

    pred_train_path = save_preds(df_train, p_train, "train")
    pred_val_path = save_preds(df_val, p_val, "val")
    pred_test_path = save_preds(df_test, p_test, "test")

    log.info("Saved model: %s", model_path_ubj)
    log.info("Saved metrics: %s", metrics_path)
    log.info("Saved preds: %s, %s, %s", pred_train_path, pred_val_path, pred_test_path)

    # log quick headline results
    best_val = metrics["val"]["weighted_significance_scan"]
    best_test = metrics["test"]["weighted_significance_scan"]
    log.info(
        "Validation best (weighted) Z_Asimov: Z=%.4g at thr=%.3f (S=%.4g, B=%.4g)",
        best_val["best_Z"],
        best_val["best_thr"],
        best_val["best_S"],
        best_val["best_B"],
    )
    log.info(
        "Test best (weighted) Z_Asimov:       Z=%.4g at thr=%.3f (S=%.4g, B=%.4g)",
        best_test["best_Z"],
        best_test["best_thr"],
        best_test["best_S"],
        best_test["best_B"],
    )


if __name__ == "__main__":
    main()
