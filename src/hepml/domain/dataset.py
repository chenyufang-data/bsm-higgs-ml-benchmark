"""Deterministic dataset cleaning and splitting; no paths or model dependencies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hepml.log import get_logger

log = get_logger(__name__)
REQUIRED_META = ["label", "weight", "xs"]


def _clean_df(df: pd.DataFrame, features: list[str], retained: list[str] | None = None) -> pd.DataFrame:
    identity = [c for c in ("event_id", "source_id", "source_entry", "sample_key") if c in df]
    needed = list(dict.fromkeys(REQUIRED_META + features + (retained or []) + identity))
    missing = [c for c in needed if c not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Object lists and identifiers are preserved verbatim. Reject bad flat inputs
    # instead of silently changing the event population for one representation.
    result = df[needed].copy()
    numeric = list(dict.fromkeys(REQUIRED_META + features))
    try:
        values = result[numeric].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Nonnumeric training features or weights") from exc
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite features or weights; refusing to change event membership")
    if not result["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0/1")
    result["label"] = result["label"].astype(int)
    return result


def _stratified_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """
    Split into train/val/test with stratification on 'target'.
    val_size is fraction of *full dataset* (not of train).
    """
    if test_size < 0 or val_size < 0 or (test_size + val_size) >= 1.0:
        raise ValueError(f"Invalid split sizes: test={test_size}, val={val_size}. Need test+val < 1.")

    rng = np.random.default_rng(seed)

    # indices per class
    idx_sig = df.index[df["target"] == 1].to_numpy()
    idx_bkg = df.index[df["target"] == 0].to_numpy()
    if len(idx_sig) == 0 or len(idx_bkg) == 0:
        raise ValueError(f"Need both classes to split: n_sig={len(idx_sig)}, n_bkg={len(idx_bkg)}")

    def split_class(idxs):
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(test_size * n))
        n_val = int(round(val_size * n))
        test = idxs[:n_test]
        val = idxs[n_test : n_test + n_val]
        train = idxs[n_test + n_val :]
        return train, val, test

    tr_s, va_s, te_s = split_class(idx_sig.copy())
    tr_b, va_b, te_b = split_class(idx_bkg.copy())

    train_idx = np.concatenate([tr_s, tr_b])
    val_idx = np.concatenate([va_s, va_b])
    test_idx = np.concatenate([te_s, te_b])

    # shuffle each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )
