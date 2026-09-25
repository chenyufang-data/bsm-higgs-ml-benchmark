"""Asimov significance and split-fraction checks."""

import numpy as np
import pandas as pd
import pytest

from hepml.domain.metrics import (
    asimov_z,
    normalized_weights,
    sample_normalization,
    split_fracs_by_class,
    split_fracs_weighted,
    threshold_yields,
)


def test_asimov_z_zero_for_degenerate_inputs():
    assert asimov_z(0.0, 100.0) == 0.0
    assert asimov_z(-5.0, 100.0) == 0.0
    assert asimov_z(10.0, 0.0) == 0.0
    assert asimov_z(10.0, -1.0) == 0.0


def test_asimov_z_known_value():
    S, B = 10.0, 100.0
    expected = np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S))
    assert asimov_z(S, B) == pytest.approx(expected)


def test_asimov_z_approximates_s_over_sqrt_b_for_small_s():
    S, B = 1.0, 10_000.0
    assert asimov_z(S, B) == pytest.approx(S / np.sqrt(B), rel=1e-3)


def _frame(n_sig, n_bkg, w_sig=1.0, w_bkg=1.0):
    return pd.DataFrame(
        {
            "target": [1] * n_sig + [0] * n_bkg,
            "sample_weight": [w_sig] * n_sig + [w_bkg] * n_bkg,
        }
    )


def test_split_fracs_by_class():
    df_all = _frame(10, 40, w_sig=2.0, w_bkg=0.5)
    df_split = _frame(1, 8, w_sig=2.0, w_bkg=0.5)
    f_sig, f_bkg = split_fracs_by_class(df_all, df_split)
    assert f_sig == pytest.approx(0.1)
    assert f_bkg == pytest.approx(0.2)


def test_split_fracs_weighted_dict_matches_tuple():
    df_all = _frame(10, 10)
    df_split = _frame(2, 5)
    d = split_fracs_weighted(df_all, df_split)
    assert (d["f_sig"], d["f_bkg"]) == split_fracs_by_class(df_all, df_split)


def test_split_fracs_raises_on_missing_class():
    df_all = _frame(0, 10)
    with pytest.raises(ValueError):
        split_fracs_by_class(df_all, _frame(0, 2))


@pytest.fixture
def process_frames():
    full = pd.DataFrame({
        "sample": ["signal"] * 10 + ["bg_a"] * 20 + ["bg_b"] * 10,
        "target": [1] * 10 + [0] * 30,
        "sample_weight": [2.] * 10 + [1.] * 20 + [10.] * 10,
    })
    # Deliberately different process fractions: 20%, 10%, 50%.
    return full, full.iloc[[0, 1, 10, 11, 30, 31, 32, 33, 34]].copy()


def test_per_sample_closure_and_selected_yields(process_frames):
    full, split = process_frames
    before = split.copy(deep=True)
    normalization = sample_normalization(full, split)
    weights = normalized_weights(split, normalization)
    sums = split.assign(evaluation_weight=weights).groupby("sample").evaluation_weight.sum()
    assert sums.to_dict() == pytest.approx({"signal": 20., "bg_a": 20., "bg_b": 100.})
    # Select one event per process: S=2*5, B=1*10 + 10*2.
    scores = np.array([.9, .1, .9, .1, .9, .1, .1, .1, .1])
    result = threshold_yields(split.target, scores, weights, threshold=.5, lumi=10.)
    assert result["signal_yield"] == pytest.approx(10.)
    assert result["background_yield"] == pytest.approx(30.)
    assert result["signal_xs_pb"] == pytest.approx(1.)
    assert result["background_xs_pb"] == pytest.approx(3.)
    assert result["significance"] == pytest.approx(asimov_z(10., 30.))
    pd.testing.assert_frame_equal(split, before)


def test_normalization_rejects_missing_or_unknown_process(process_frames):
    full, split = process_frames
    with pytest.raises(ValueError, match="every reference sample"):
        sample_normalization(full, split[split["sample"] != "bg_b"])
    split.loc[0, "sample"] = "unknown"
    with pytest.raises(ValueError, match="unknown samples"):
        sample_normalization(full, split)


@pytest.mark.parametrize("bad_weight", [0., -1., np.nan, np.inf])
def test_normalization_rejects_invalid_weights(process_frames, bad_weight):
    full, split = process_frames
    split.loc[split["sample"] == "bg_b", "sample_weight"] = bad_weight
    with pytest.raises(ValueError):
        sample_normalization(full, split)
