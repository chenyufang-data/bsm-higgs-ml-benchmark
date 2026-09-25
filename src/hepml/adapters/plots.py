#!/usr/bin/env python3
"""Matplotlib, SHAP and XGBoost diagnostic plot adapters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import shap
except ImportError:
    shap = None

from hepml.log import get_logger

log = get_logger(__name__)


# ----------------------------
# Helpers
# ----------------------------
def _read_preds(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"target", "bdt_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
    return df


def _plot_score_hists(train_df, test_df, outpath: Path):
    """
    Score distributions train vs test, signal vs background, with KS test.
    """
    fig = plt.figure(figsize=(7, 5))

    # Split
    tr_s = train_df[train_df["target"] == 1]["bdt_score"].to_numpy()
    tr_b = train_df[train_df["target"] == 0]["bdt_score"].to_numpy()
    te_s = test_df[test_df["target"] == 1]["bdt_score"].to_numpy()
    te_b = test_df[test_df["target"] == 0]["bdt_score"].to_numpy()

    # KS tests (unweighted)
    ks_s = ks_2samp(tr_s, te_s)
    ks_b = ks_2samp(tr_b, te_b)

    bins = np.linspace(0, 1, 41)
    plt.hist(tr_s, bins=bins, density=True, histtype="step", linewidth=2, label=f"Train S (KS p={ks_s.pvalue:.3g})")
    plt.hist(tr_b, bins=bins, density=True, histtype="step", linewidth=2, label=f"Train B (KS p={ks_b.pvalue:.3g})")

    plt.hist(te_s, bins=bins, density=True, alpha=0.35, label="Test S (filled)")
    plt.hist(te_b, bins=bins, density=True, alpha=0.35, label="Test B (filled)")

    plt.xlabel("BDT score")
    plt.ylabel("Density")
    plt.title("Score distributions (overtraining check)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _plot_roc(df, outpath: Path, title: str):
    fig = plt.figure(figsize=(6, 5))

    y = df["target"].to_numpy(dtype=int)
    s = df["bdt_score"].to_numpy(dtype=float)
    w = df["sample_weight"].to_numpy(dtype=np.float64)

    # Unweighted ROC
    fpr, tpr, _ = roc_curve(y, s)
    auc_u = roc_auc_score(y, s)

    # Weighted ROC (sklearn supports sample_weight)
    fprw, tprw, _ = roc_curve(y, s, sample_weight=w)
    auc_w = roc_auc_score(y, s, sample_weight=w)

    plt.plot(fpr, tpr, label=f"Unweighted AUC = {auc_u:.3f}")
    plt.plot(fprw, tprw, label=f"Weighted AUC = {auc_w:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _plot_significance(
    model_dir: Path,
    outpath: Path,
    title: str,
    split: str,
):
    """
    Plot weighted Z_Asimov vs threshold from metrics.json.
    """
    fig = plt.figure(figsize=(7, 5))

    model_path = model_dir / "metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} does not exist")

    with open(model_path) as f:
        metrics = json.load(f)

    scan = metrics[split]["weighted_significance_scan"]["scan"]
    best = metrics[split]["weighted_significance_scan"]

    thr_all = np.asarray(scan["thr"], dtype=float)
    Z_all = np.asarray(scan["Z"], dtype=float)
    nS_all = np.asarray(scan["nS"], dtype=float)
    nB_all = np.asarray(scan["nB"], dtype=float)
    B_all = np.asarray(scan["B"], dtype=float)

    min_nS = best["constraints"]["min_nS"]
    min_nB = best["constraints"]["min_nB"]
    min_B = best["constraints"]["min_B"]
    eps_plateau = best["constraints"].get("eps_plateau", 0.02)

    mask = (nS_all >= min_nS) & (nB_all >= min_nB) & (B_all >= min_B)

    thr = thr_all[mask]
    Zs = Z_all[mask]

    if thr.size == 0:
        raise ValueError("No scan points pass constraints.")

    plateau = thr[Zs >= (1.0 - eps_plateau) * Zs.max()]
    log.info("Z plateau: [%.4f, %.4f]", plateau.min(), plateau.max())

    plt.plot(thr, Zs)
    plt.axvline(
        best["best_thr"],
        linestyle="--",
        label=(
            f"thr={best['best_thr']:.3f}, Z={best['best_Z']:.1f} (nS={best['best_nS']:.3f}, nB={best['best_nB']:.1f})"
        ),
    )

    plt.xlabel("Score threshold")
    plt.ylabel(r"Weighted $Z_\mathrm{Asimov}=\sqrt{2\left((S+B)\ln(1+\frac{S}{B})-S\right)}$")
    plt.title(title + f"\n(S={best['best_S']:.3g}, B={best['best_B']:.3g})")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

    return best, scan


def _plot_permutation_importance(test_df: pd.DataFrame, feature_names: list[str], outdir: Path, n_repeats=10, seed=42):
    """
    Permutation importance is often more interpretable than built-in gain/weight.
    Needs features present in preds_test.parquet.
    """
    missing = [f for f in feature_names if f not in test_df.columns]
    if missing:
        log.warning("Permutation importance skipped: missing features in test preds parquet: %s", missing)
        return

    X = test_df[feature_names].to_numpy(dtype=np.float32)
    y = test_df["target"].to_numpy(dtype=np.int32)
    w = test_df["sample_weight"].to_numpy(dtype=np.float64)

    # Wrap a saved XGBoost Booster as a sklearn-compatible estimator
    # so it can be used with sklearn utilities.
    from sklearn.base import BaseEstimator, ClassifierMixin

    class BoosterWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model_path: str, feature_names: list[str]):
            self.model_path = model_path
            self.feature_names = feature_names
            self.booster_ = None

        def fit(self, X, y=None):
            booster = xgb.Booster()
            booster.load_model(self.model_path)
            self.booster_ = booster
            return self

        def predict_proba(self, X):
            dm = xgb.DMatrix(X, feature_names=self.feature_names)
            p = self.booster_.predict(dm)  # shape (n,)
            p = p.reshape(-1, 1)
            return np.hstack([1 - p, p])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    # load model.ubj
    model_path = outdir.parent / "model.ubj"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} does not exist")

    est = BoosterWrapper(str(model_path), feature_names).fit(X, y)

    def auc_scorer(estimator, X_, y_):
        s_ = estimator.predict_proba(X_)[:, 1]
        return roc_auc_score(y_, s_, sample_weight=w)

    result = permutation_importance(
        est,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=auc_scorer,
    )

    imp = result.importances_mean
    err = result.importances_std
    order = np.argsort(imp)

    fig = plt.figure(figsize=(7, 5))
    plt.barh(np.array(feature_names)[order], imp[order], xerr=err[order])
    plt.xlabel("Permutation importance (#Delta weighted AUC)")
    plt.title("Permutation importance (test)")
    plt.tight_layout()
    fig.savefig(outdir / "importance_permutation.png")
    plt.close(fig)


def _plot_shap_summary(
    model_path: Path, df: pd.DataFrame, feature_names: list[str], outpath: Path, max_points=20000, seed=42
):
    if shap is None:
        log.warning("SHAP skipped: shap not installed. Install with: pip install shap")
        return

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        log.warning("SHAP skipped: missing features in preds parquet: %s", missing)
        return

    # load booster
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # sample rows to keep runtime/memory reasonable
    df_ = df.sample(n=min(len(df), max_points), random_state=seed) if len(df) > max_points else df
    X = df_[feature_names].to_numpy(dtype=np.float32)

    # compute SHAP values
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)

    # save plot
    plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
