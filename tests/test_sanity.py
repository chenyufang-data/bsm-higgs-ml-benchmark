"""Sanity-check module: accepts a well-formed frame, rejects broken ones."""

import numpy as np
import pandas as pd
import pytest

from hepml.domain.validation import SanityError, sanity_prepared_ml, sanity_root_parquet
from hepml.domain.weights import compute_sample_weight
from studies.cg_bbc.features import DEFAULT_FEATURES

LUMI = 3000.0
XS_PB = 2.0e-2
N_TOTAL = 200_000.0


def _prepared_frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.uniform(1.0, 100.0, size=n) for f in DEFAULT_FEATURES})
    df["target"] = (rng.uniform(size=n) < 0.5).astype(int)
    # ensure both classes
    df.loc[0, "target"] = 1
    df.loc[1, "target"] = 0
    df["gen_weight"] = rng.uniform(0.5, 1.5, size=n)
    df["evt_xs"] = rng.uniform(0.1, 2.0, size=n)
    df["xs_pb"] = XS_PB
    df["n_events_total"] = N_TOTAL
    df["sample_weight"] = compute_sample_weight(df["xs_pb"], df["n_events_total"], lumi=LUMI)
    return df


def _raw_frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.uniform(1.0, 100.0, size=n) for f in DEFAULT_FEATURES})
    df["label"] = (rng.uniform(size=n) < 0.5).astype(int)
    df["weight"] = rng.uniform(0.5, 1.5, size=n)
    df["xs"] = rng.uniform(0.1, 2.0, size=n)
    return df


def test_prepared_frame_passes():
    report = sanity_prepared_ml(_prepared_frame(), features=DEFAULT_FEATURES, lumi=LUMI)
    assert report["stage"] == "prepared_ml"
    assert report["n_rows"] == 50


def test_wrong_lumi_fails():
    # the frame was built at LUMI; validating against a different lumi must fail
    with pytest.raises(SanityError):
        sanity_prepared_ml(_prepared_frame(), features=DEFAULT_FEATURES, lumi=LUMI / 10)


def test_corrupted_sample_weight_fails():
    df = _prepared_frame()
    df.loc[5, "sample_weight"] *= 1.5
    with pytest.raises(SanityError):
        sanity_prepared_ml(df, features=DEFAULT_FEATURES, lumi=LUMI)


def test_wrong_n_total_fails():
    df = _prepared_frame()
    df.loc[3, "n_events_total"] = N_TOTAL * 2  # weight no longer matches
    with pytest.raises(SanityError):
        sanity_prepared_ml(df, features=DEFAULT_FEATURES, lumi=LUMI)


def test_nan_feature_fails():
    df = _prepared_frame()
    df.loc[7, DEFAULT_FEATURES[0]] = np.nan
    with pytest.raises(SanityError):
        sanity_prepared_ml(df, features=DEFAULT_FEATURES, lumi=LUMI)


def test_feature_subset_is_respected():
    # a frame missing one feature column passes when that feature is not requested
    dropped = DEFAULT_FEATURES[0]
    df = _prepared_frame().drop(columns=[dropped])
    features = [f for f in DEFAULT_FEATURES if f != dropped]
    report = sanity_prepared_ml(df, features=features, lumi=LUMI)
    assert report["n_rows"] == 50
    with pytest.raises(SanityError):
        sanity_prepared_ml(df, features=DEFAULT_FEATURES, lumi=LUMI)  # full default list must fail


def test_raw_frame_passes_and_bad_label_fails():
    assert sanity_root_parquet(_raw_frame(), features=DEFAULT_FEATURES)["stage"] == "root_parquet"
    df = _raw_frame()
    df.loc[2, "label"] = 2
    with pytest.raises(SanityError):
        sanity_root_parquet(df, features=DEFAULT_FEATURES)
