"""Binned shape-fit sensitivity: significance limits, bin merging, calibration and pairing."""

import numpy as np
import pytest

from hepml.application.binned_fit import (
    asimov_significance,
    boundaries_follow_rule,
    closest_cb_mass,
    log_bins,
    match_to_partons,
    merge_bins,
    pair_mass,
    response_curve,
    response_scale,
    score_boundaries,
)
from hepml.domain.metrics import asimov_z_syst


def test_one_bin_is_the_small_signal_limit_of_the_counting_formula():
    signal, background = 5.1e5, 2.42e8
    z = asimov_significance([signal], [background], normalization=0.05)
    assert z == pytest.approx(signal / np.sqrt(background + (0.05 * background) ** 2))
    # Cowan's Eq. 20 used by the grid agrees to far better than the pre-registered 1%.
    assert z == pytest.approx(asimov_z_syst(signal, background, 0.05), rel=1e-3)
    assert z == pytest.approx(20 * signal / background, rel=1e-3)


def test_sideband_constrains_the_shared_normalization_unless_shapes_are_uncertain():
    signal, background = [5000.0, 0.0], [1.0e6, 1.0e7]
    single = asimov_significance(signal[:1], background[:1], normalization=0.05)
    results = [asimov_significance(signal, background, normalization=0.05, shape=d) for d in (0.0, 0.01, 0.02, 0.05)]
    assert single == pytest.approx(0.1, rel=1e-3)
    assert results == pytest.approx([4.767, 0.356, 0.183, 0.082], abs=1e-3)
    assert all(a > b for a, b in zip(results, results[1:]))


def test_mc_statistics_only_lower_the_significance():
    signal, background = [100.0, 400.0, 50.0], [1e5, 2e5, 1e6]
    without = asimov_significance(signal, background, normalization=0.05, shape=0.02)
    with_mc = asimov_significance(signal, background, normalization=0.05, shape=0.02, mc_variance=[1e8, 4e8, 1e9])
    assert with_mc < without
    with pytest.raises(ValueError):
        asimov_significance([1.0], [0.0], normalization=0.05)


def test_log_bins_keep_underflow_and_overflow():
    edges = log_bins(0.3, 2.0, 0.1)
    assert edges[0] == 0 and np.isinf(edges[-1]) and edges[1] == pytest.approx(0.3)
    assert edges[-2] >= 2.0 > edges[-3]
    np.testing.assert_allclose(np.diff(np.log(edges[1:-1])), 0.1)


def test_bins_merge_upwards_until_the_effective_count_is_reached():
    # Uniform unit weights: N_eff equals the event count.
    counts = np.array([3.0, 4.0, 5.0, 20.0, 2.0, 1.0])
    groups = merge_bins(counts, counts, min_neff=10)
    assert groups == [[0, 1, 2], [3, 4, 5]]  # the sparse tail joins the last full group
    assert merge_bins([1.0, 1.0], [1.0, 1.0], min_neff=10) == [[0, 1]]
    # A few heavy events need more merging than their raw count suggests.
    heavy = merge_bins([100.0, 100.0], [100.0**2 / 2, 100.0**2 / 20], min_neff=10)
    assert heavy == [[0, 1]]


def test_boundary_rule_accepts_heavy_events_and_rejects_other_thresholds():
    score = np.linspace(1.0, 0.1, 10)
    heavy = np.r_[1.0, 1.0, 5.0, np.ones(7)]  # the third event alone is a third of the total
    thresholds = score_boundaries(score, heavy, [0.5, 0.1])
    assert boundaries_follow_rule(score, heavy, thresholds, [0.5, 0.1])
    assert abs((heavy[score >= thresholds[1]].sum() / heavy.sum()) - 0.1) > 0.01  # far from 10%, by the rule
    assert not boundaries_follow_rule(score, heavy, [score[4], thresholds[1]], [0.5, 0.1])  # could go lower
    assert not boundaries_follow_rule(score, heavy, [score[8], thresholds[1]], [0.5, 0.1])  # keeps too much


def test_score_boundaries_follow_the_weighted_background_fraction():
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    weight = np.ones(10)
    thresholds = score_boundaries(score, weight, [0.5, 0.2])
    np.testing.assert_allclose(thresholds, [0.5, 0.8])  # scores >= 0.5 keep 50%, >= 0.8 keep 20%
    # Heavy events count by weight, and a fraction is never exceeded.
    heavy = score_boundaries(score, np.r_[1.0, 3.0, np.ones(8)], [0.5, 0.1])
    np.testing.assert_allclose(heavy, [0.6, 0.9])
    # Ties are accepted together.
    tied = score_boundaries(np.array([0.9, 0.9, 0.5, 0.1]), np.ones(4), [0.6])
    np.testing.assert_allclose(tied, [0.9])
    categories = np.searchsorted(thresholds, score, side="right")
    assert categories.tolist() == [2, 2, 1, 1, 1, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="decrease"):
        score_boundaries(score, weight, [0.2, 0.5])
    with pytest.raises(ValueError, match="alone exceeds"):
        score_boundaries(score, np.r_[10.0, np.ones(9)], [0.2])
    with pytest.raises(ValueError, match="separate"):
        score_boundaries(np.array([0.9, 0.9, 0.5, 0.1]), np.ones(4), [0.6, 0.55])


def test_response_curve_and_interpolation():
    rng = np.random.default_rng(1)
    reco = rng.uniform(25, 500, 20000)
    parton = reco * np.where(reco < 100, 1.2, 1.05)
    nodes, values, counts = response_curve(reco, parton, [25, 50, 100, 200])
    assert values == pytest.approx([1.2, 1.2, 1.05, 1.05]) and counts.sum() == len(reco)
    scale = response_scale([10.0, nodes[0], 1000.0], nodes, values)
    assert scale == pytest.approx([1.2, 1.2, 1.05])  # constant beyond the outer nodes
    between = response_scale([np.sqrt(nodes[1] * nodes[2])], nodes, values)[0]
    assert between == pytest.approx((1.2 + 1.05) / 2)  # linear in log(pT)
    with pytest.raises(ValueError, match="No calibration jets"):
        response_curve(reco, parton, [25, 1000, 2000])


def test_closest_pairing_and_mass():
    # Two back-to-back massless jets of 100 GeV make 200 GeV.
    c1 = (np.array([100.0, 100.0]), np.zeros(2), np.zeros(2), np.zeros(2))
    b1 = (np.array([100.0, 50.0]), np.zeros(2), np.full(2, np.pi), np.zeros(2))
    b2 = (np.array([30.0, 100.0]), np.zeros(2), np.full(2, np.pi), np.zeros(2))
    assert pair_mass(b1, c1)[0] == pytest.approx(200.0)
    masses, chosen = closest_cb_mass(b1, b2, c1, 200.0)
    np.testing.assert_allclose(masses, [200.0, 200.0])
    assert chosen.tolist() == [1, 2]
    # Scaling both jets by 1.1 scales the mass by 1.1.
    scaled, _ = closest_cb_mass(b1, b2, c1, 220.0, scale_b1=1.1, scale_b2=1.1, scale_c1=1.1)
    assert scaled[0] == pytest.approx(220.0)


def test_parton_matching_uses_flavour_status_and_distance():
    pid, status = np.array([5, -5, 4, 21]), np.array([23, 23, 23, 21])
    eta, phi = np.array([0.0, 1.0, 0.1, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])
    assert match_to_partons(0.05, 0.0, pid, status, eta, phi, flavour=5, max_delta_r=0.4) == 0
    assert match_to_partons(0.05, 0.0, pid, status, eta, phi, flavour=4, max_delta_r=0.4) == 2
    assert match_to_partons(2.0, 2.0, pid, status, eta, phi, flavour=5, max_delta_r=0.4) == -1


def test_calibration_jets_match_tagged_jets_to_partons_of_their_flavour(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hepml.commands.binned_fit_study import calibration_jets

    floats, ints = pa.list_(pa.float32()), pa.list_(pa.int32())
    table = pa.table({
        "event_id": ["train-event", "other-event"],
        "jets_pt": pa.array([[100.0, 80.0, 60.0], [50.0, 40.0, 30.0]], floats),
        "jets_eta": pa.array([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]], floats),
        "jets_phi": pa.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], floats),
        "role_b1": [0, 0], "role_b2": [1, 1], "role_c1": [2, 2],
        # b parton near jet 0, a c parton near jet 1 (a mistag), c parton near jet 2.
        "truth_partons_pid": pa.array([[5, 4, 4, 21], [5, 4, 4, 21]], ints),
        "truth_partons_status": pa.array([[23, 23, 23, 21]] * 2, ints),
        "truth_partons_eta": pa.array([[0.05, 1.0, -1.0, 0.0]] * 2, floats),
        "truth_partons_phi": pa.array([[0.0, 1.0, 2.0, 0.0]] * 2, floats),
        "truth_partons_pt": pa.array([[110.0, 90.0, 75.0, 0.0]] * 2, floats),
    })
    path = tmp_path / "signal.part00000.parquet"
    pq.write_table(table, path)
    jets = calibration_jets([path], {"train-event"}, 0.4)
    assert jets.kind.tolist() == ["b", "b", "c"]
    assert jets.parton_pt.iloc[0] == pytest.approx(110.0) and np.isnan(jets.parton_pt.iloc[1])
    assert jets.parton_pt.iloc[2] == pytest.approx(75.0)
    pq.write_table(table.drop_columns(["truth_partons_pt"]), path)
    with pytest.raises(ValueError, match="no parton truth"):
        calibration_jets([path], {"train-event"}, 0.4)
