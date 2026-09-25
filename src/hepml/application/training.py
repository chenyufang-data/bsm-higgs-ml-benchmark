"""Fit any score model on prepared inputs; evaluate with common physics metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hepml.domain.metrics import normalized_weights, safe_auc, sample_normalization, weighted_significance_scan
from hepml.ports import ModelTrainer, ScoreModel


def make_xyw(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "target",
    weight_xs_col: str = "sample_weight",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bookkeeping = {target_col, weight_xs_col, "fit_weight", "xs_pb", "n_events_total", "gen_weight", "evt_xs",
                   "event_id", "source_id", "source_entry", "sample", "sample_key", "split",
                   "generated_mass", "generated_rho_tc", "hypothesis_id", "replica_id"}
    if bookkeeping.intersection(features):
        raise ValueError("Physical/fit weights, labels and event bookkeeping cannot be ML features")
    # Compact recipes must name generator-truth columns with "truth" (enforced there).
    if any("truth" in feature for feature in features):
        raise ValueError("Generator-truth columns cannot be ML features")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    X = df[features].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.int32)

    # physics eval weight (lumi * merged cross section / generated count, see hepml.domain.weights)
    if weight_xs_col in df.columns:
        w_xs = df[weight_xs_col].to_numpy(dtype=np.float64)
    else:
        w_xs = np.ones(len(df), dtype=np.float64)

    return X, y, w_xs


def compute_scale_pos_weight(y: np.ndarray) -> float:
    # typical choice: neg/pos
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos <= 0:
        return 1.0
    return n_neg / n_pos


def build_fit_weights(df: pd.DataFrame, mode: str, *, reference: pd.DataFrame | None = None) -> np.ndarray | None:
    """Per-event weights for the FIT (not evaluation), per --weight-mode.

    'none': unweighted training (default; shapes only).
    'balanced-mixture': within each class, processes keep their PHYSICAL
    proportions (proportional to sample_weight, i.e. xs/N), but the two
    classes are rescaled to equal totals so the background's huge absolute
    cross-section cannot drown the signal. Fixes the background-cocktail
    composition (e.g. bbc being ~40% of background MC events but <1% of
    the physical background rate) without destabilizing training.
    """
    if mode == "none":
        return None
    if mode != "balanced-mixture":
        raise ValueError(f"Unknown weight mode: {mode!r}")

    w = (df["sample_weight"].to_numpy(dtype=np.float64).copy() if reference is None
         else normalized_weights(df, sample_normalization(reference, df)))
    y = df["target"].to_numpy()
    for cls in (0, 1):
        m = y == cls
        total = float(w[m].sum())
        if total <= 0:
            raise ValueError(f"Class {cls} has non-positive total sample_weight; cannot balance")
        w[m] *= (len(df) / 2.0) / total
    return w


def fit_and_score(
    trainer: ModelTrainer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    **options,
) -> tuple[ScoreModel, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model = trainer(X_train, y_train, X_val, y_val, **options)
    scores = tuple(model.predict_proba(values)[:, 1] for values in (X_train, X_val, X_test))
    return model, scores


def evaluate_scores(y, scores, weights, *, lumi, frac_sig=1.0, frac_bkg=1.0, scan=None):
    result = {"auc_unweighted": safe_auc(y, scores), "auc_weighted": safe_auc(y, scores, sample_weight=weights)}
    if scan is not None:
        result["weighted_significance_scan"] = weighted_significance_scan(
            y, scores, weights, lumi=lumi, frac_sig=frac_sig, frac_bkg=frac_bkg, **scan
        )
    return result
