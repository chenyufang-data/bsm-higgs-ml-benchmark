"""Audit statistics and immutable split anchors; the workflow is in test_shape_audit_cli.py."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp, wasserstein_distance

from hepml.application.shape_audit import audit_decision, bootstrap_counts, followup_reasons
from hepml.domain.metrics import binomial_interval, bootstrap_wasserstein, shape_distances
from hepml.domain.splits import extend_registry


def test_registry_preserves_anchor_and_ignores_input_order():
    anchor = pd.DataFrame(dict(event_id=[f"anchor:{i}" for i in range(20)], sample_key="anchor",
                               split=["train"] * 16 + ["val"] * 2 + ["test"] * 2,
                               assignment_method="preserved_pilot_anchor"))
    extra = pd.DataFrame(dict(event_id=[f"new:{i}" for i in range(100)], sample_key="new"))
    expected = extend_registry(anchor, extra)
    actual = extend_registry(anchor, extra.sample(frac=1, random_state=12))
    pd.testing.assert_frame_equal(expected, actual)
    pd.testing.assert_frame_equal(expected[expected.sample_key == "anchor"].sort_values("event_id").reset_index(drop=True),
                                  anchor.sort_values("event_id").reset_index(drop=True))
    assert expected[expected.sample_key == "new"].split.value_counts().to_dict() == {"train": 80, "val": 10, "test": 10}
    pd.testing.assert_frame_equal(extend_registry(expected, extra), expected)
    with pytest.raises(ValueError, match="membership changed"):
        extend_registry(expected, extra.iloc[:-1])
    with pytest.raises(ValueError, match="different samples"):
        extend_registry(anchor, extra.assign(event_id=["anchor:0"] + list(extra.event_id[1:])))


def test_shape_metrics_and_bootstrap_match_scipy_with_ties():
    x, y = np.array([0., 1., 1., 2.]), np.array([1., 1., 2., 5., 7.])
    result = shape_distances(x, y, reference_iqr=2)
    assert result["ks"] == pytest.approx(ks_2samp(x, y).statistic)
    assert result["wasserstein"] == pytest.approx(wasserstein_distance(x, y))
    assert result["wasserstein_iqr"] == pytest.approx(wasserstein_distance(x, y) / 2)
    assert shape_distances(x, y, reference_iqr=0)["wasserstein_iqr"] is None
    rng = np.random.default_rng(2)
    cx, cy = bootstrap_counts(len(x), 25, rng), bootstrap_counts(len(y), 25, rng)
    actual = bootstrap_wasserstein(x, y, cx, cy)
    expected = [wasserstein_distance(x, y, a, b) for a, b in zip(cx, cy)]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert binomial_interval(0, 10, alpha=.05)[0] == 0
    assert binomial_interval(10, 10, alpha=.05)[1] == 1


def test_inconclusive_does_not_trigger_conditional_masses():
    config = {"two_sample_auc_margin": .55, "stage_alpha": .025}
    distances = pd.DataFrame(dict(ks_followup=[False], w1_followup=[False], within_margins=[False]))
    efficiencies = pd.DataFrame(dict(followup=[False], within_margin=[False]))
    joint = pd.DataFrame(dict(auc_lower=[.48], auc_upper=[.57], permutation_p_adjusted=[1.]))
    assert not followup_reasons(distances, efficiencies, joint, config)
    assert audit_decision(distances, efficiencies, joint, production_complete=False, config=config) == "inconclusive"
    distances.loc[0, "ks_followup"] = True
    assert followup_reasons(distances, efficiencies, joint, config)
    assert audit_decision(distances, efficiencies, joint, production_complete=False, config=config) == "shape-relevant"
