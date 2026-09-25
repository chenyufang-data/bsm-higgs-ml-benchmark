"""The physics weight formula: lumi * xs / n_events_total."""

import numpy as np
import pandas as pd
import pytest

from hepml.domain.weights import compute_sample_weight


def test_scalar_formula():
    # yield = xs * lumi * (n_passed / n_total): per-event weight lumi*xs/n_total
    assert compute_sample_weight(0.5, 1000.0, lumi=3000.0) == pytest.approx(3000.0 * 0.5 / 1000.0)


def test_summed_weights_give_expected_yield():
    xs, n_total, lumi, n_passed = 2.0e-2, 200_000, 3000.0, 13_003
    w = compute_sample_weight(xs, n_total, lumi=lumi)
    assert n_passed * w == pytest.approx(xs * lumi * n_passed / n_total)


def test_vectorized_over_series():
    xs = pd.Series([1.0, 2.0])
    n = pd.Series([100.0, 100.0])
    w = compute_sample_weight(xs, n, lumi=10.0)
    assert np.allclose(w, [0.1, 0.2])


def test_lumi_scales_linearly():
    w1 = compute_sample_weight(1.0, 1.0, lumi=1.0)
    w2 = compute_sample_weight(1.0, 1.0, lumi=300.0)
    assert w2 == pytest.approx(300.0 * w1)
