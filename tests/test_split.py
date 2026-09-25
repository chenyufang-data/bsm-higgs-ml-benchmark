"""Stratified split: determinism, stratification, degenerate inputs."""

import numpy as np
import pandas as pd
import pytest

from hepml.domain.dataset import _stratified_split


def _dataset(n_sig=100, n_bkg=300, seed=0):
    rng = np.random.default_rng(seed)
    n = n_sig + n_bkg
    return pd.DataFrame(
        {
            "target": [1] * n_sig + [0] * n_bkg,
            "x": rng.normal(size=n),
            "sample_weight": rng.uniform(0.5, 1.5, size=n),
        }
    )


def test_split_sizes_and_stratification():
    df = _dataset()
    train, val, test = _stratified_split(df, test_size=0.1, val_size=0.1, seed=42)
    assert len(train) + len(val) + len(test) == len(df)
    # 10% per class, rounded
    assert (test["target"] == 1).sum() == 10
    assert (test["target"] == 0).sum() == 30
    assert (val["target"] == 1).sum() == 10
    # every split keeps both classes
    for part in (train, val, test):
        assert set(part["target"].unique()) == {0, 1}


def test_split_is_deterministic_for_fixed_seed_and_order():
    df = _dataset()
    a = _stratified_split(df, 0.1, 0.1, seed=7)
    b = _stratified_split(df, 0.1, 0.1, seed=7)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_split_changes_with_seed():
    df = _dataset()
    a_train, _, _ = _stratified_split(df, 0.1, 0.1, seed=1)
    b_train, _, _ = _stratified_split(df, 0.1, 0.1, seed=2)
    assert not a_train["x"].equals(b_train["x"])


def test_split_partitions_rows_without_overlap():
    df = _dataset()
    train, val, test = _stratified_split(df, 0.2, 0.1, seed=3)
    combined = pd.concat([train, val, test])["x"].sort_values().to_numpy()
    assert np.allclose(combined, df["x"].sort_values().to_numpy())


def test_split_rejects_bad_sizes():
    df = _dataset()
    with pytest.raises(ValueError):
        _stratified_split(df, 0.6, 0.5, seed=0)


def test_split_rejects_single_class():
    df = _dataset(n_sig=0, n_bkg=50)
    with pytest.raises(ValueError):
        _stratified_split(df, 0.1, 0.1, seed=0)
