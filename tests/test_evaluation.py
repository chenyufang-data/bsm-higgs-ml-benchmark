"""Independent counting-formula references, MC support and frozen thresholds."""

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from hepml.application.evaluation import evaluate_frozen, evaluate_validation
from hepml.application.training import build_fit_weights, make_xyw
from hepml.domain.config import EvaluationConfig
from hepml.domain.metrics import (
    asimov_z,
    asimov_z_syst,
    operating_curve,
    select_operating_points,
    yield_metrics,
)
from hepml.domain.weights import balanced_hypothesis_weights


def decimal_reference(s, b, delta):
    with localcontext() as context:
        context.prec = 160
        s, b, delta = map(lambda x: Decimal(str(x)), (s, b, delta))
        variance = (delta*b)**2
        first = (s+b)*(((s+b)*(b+variance))/(b*b+(s+b)*variance)).ln()
        second = b*b/variance*(1+variance*s/(b*(b+variance))).ln()
        return float((2*(first-second)).sqrt())


@pytest.mark.parametrize("s,b,delta", [
    (10, 100, .05), (1, 1e12, .1), (1e-7, 1e12, .1), (100, 1e-12, .1),
    (10, 100, 1e-16), (1, 1e-100, .05), (1e5, 1e6, .1), (.0099, 1, .1), (.0101, 1, .1),
])
def test_systematic_formula_matches_independent_high_precision(s, b, delta):
    assert asimov_z_syst(s, b, delta) == pytest.approx(decimal_reference(s, b, delta), rel=2e-9, abs=1e-30)


def test_metrics_boundaries_and_ordering():
    for s, b in ((0., 100.), (0., 0.), (10., 0.), (1e-7, 1e12), (30., 200.)):
        assert asimov_z_syst(s, b, 0) == asimov_z(s, b)
        assert asimov_z_syst(s, b, .1) <= asimov_z_syst(s, b, .05) * (1+1e-10) + 1e-30
        assert asimov_z_syst(s, b, .05) <= asimov_z(s, b) * (1+1e-10) + 1e-30
    assert yield_metrics(1, 0)["S_over_B"] is None
    assert not yield_metrics(1, 0)["valid"]
    assert yield_metrics(0, 1)["S_over_B"] == 0
    for invalid in (-1., np.nan, np.inf):
        with pytest.raises(ValueError):
            asimov_z_syst(1, 1, invalid)
        with pytest.raises(ValueError):
            yield_metrics(invalid, 1)


def frame():
    return pd.DataFrame(dict(sample=["s"]*4+["a"]*4+["b"]*4, target=[1]*4+[0]*8,
                             event_id=[str(i) for i in range(12)],
                             sample_weight=[2.]*4+[1.]*4+[10.]*4,
                             bdt_score=[.9,.8,.5,.1,.8,.6,.3,.1,.8,.7,.2,.1]))


def test_cumulative_scan_matches_direct_cuts_and_has_endpoints():
    data = frame()
    config = EvaluationConfig(min_background_neff=1)
    curve, processes = operating_curve(data, data.sample_weight, config)
    assert curve.iloc[0].threshold == 0
    assert curve.iloc[-1].threshold > data.bdt_score.max()
    assert not curve.iloc[-1].eligible
    for row in curve.itertuples():
        selected = data[data.bdt_score >= row.threshold]
        assert row.S == pytest.approx(selected.loc[selected.target == 1, "sample_weight"].sum())
        assert row.B == pytest.approx(selected.loc[selected.target == 0, "sample_weight"].sum())
        assert row.background_mc == len(selected[selected.target == 0])
        assert row.Z_syst_5pct == pytest.approx(asimov_z_syst(row.S, row.B, .05))
    assert len(processes) == len(curve)*3
    # Removing all MC from one process must not look like confirmed zero yield.
    assert not curve.loc[curve.threshold == .9, "all_background_processes_supported"].iloc[0]
    points = select_operating_points(curve)
    for name, point in points.items():
        assert point["metrics"][name] == curve.loc[curve.eligible, name].max()
    with pytest.raises(ValueError, match="validation"):
        select_operating_points(curve, threshold_source="test")
    with pytest.raises(ValueError, match="replicas"):
        operating_curve(pd.concat([data, data]), np.tile(data.sample_weight, 2), config)


def test_no_valid_threshold_and_deterministic_tie():
    curve, _ = operating_curve(frame(), frame().sample_weight, EvaluationConfig(min_background_neff=1000))
    assert all(p["threshold"] is None for p in select_operating_points(curve).values())
    curve["eligible"] = True
    curve["Z_Asimov"] = 1.
    assert select_operating_points(curve)["Z_Asimov"]["threshold"] == 0


def test_luminosity_fit_isolation_and_frozen_roundtrip():
    import json

    data = frame()
    original = data.copy(deep=True)
    cfg = EvaluationConfig(lumi_pb_inv=100, min_background_neff=1)
    first, curve, _, _ = evaluate_validation(data, data, cfg, stored_lumi=10)
    double, doubled, _, _ = evaluate_validation(data, data, EvaluationConfig(lumi_pb_inv=200, min_background_neff=1),
                                               stored_lumi=10)
    np.testing.assert_allclose(doubled.S, 2*curve.S)
    np.testing.assert_allclose(doubled.B, 2*curve.B)
    np.testing.assert_allclose(doubled.S_over_B, curve.S_over_B, equal_nan=True)
    np.testing.assert_allclose(doubled.signal_efficiency, curve.signal_efficiency)
    assert first["auc_weighted"] == double["auc_weighted"]
    assert build_fit_weights(data, "none") is None
    fit = build_fit_weights(data, "balanced-mixture", reference=data)
    assert fit[data.target == 1].sum() == pytest.approx(fit[data.target == 0].sum())
    after, _, _, _ = evaluate_validation(data, data, cfg, stored_lumi=10)
    assert first == after
    frozen = json.loads(json.dumps(first))
    result = evaluate_frozen(data, data, frozen, cfg, stored_lumi=10, split="test")
    for row in result.itertuples():
        point = first["operating_points"][row.objective]
        assert row.threshold == point["threshold"]
        assert row.S == pytest.approx(point["metrics"]["S"])
    pd.testing.assert_frame_equal(data, original)


def test_hypothesis_priors_and_replication_conserve_fit_weight():
    y = np.array([1,1,1,0,0,0,0])
    h = np.array([200,400,400,200,200,400,400])
    physical = np.array([10.,1.,1.,1.,20.,1.,20.])
    ids = np.array(["s1","s2","s3","b1","b2","b1","b2"])
    before = physical.copy()
    weights = balanced_hypothesis_weights(y, physical, h, {200:.5,400:.5}, event_ids=ids)
    assert weights.sum() == pytest.approx(5.)
    for target in (0,1):
        for mass in (200,400):
            assert weights[(y==target)&(h==mass)].sum() == pytest.approx(1.25)
    assert weights[4]/weights[3] == pytest.approx(20.)
    np.testing.assert_array_equal(physical, before)


def test_k_factors_scale_yields_once_and_survive_frozen_reload():
    import json

    data = frame()
    before = data.copy(deep=True)
    raw_cfg = EvaluationConfig(lumi_pb_inv=100, min_background_neff=1)
    cfg = EvaluationConfig(lumi_pb_inv=100, min_background_neff=1,
                           signal_k_factor=1.95, background_k_factor=1.26)
    old, raw, _, _ = evaluate_validation(data, data, raw_cfg, stored_lumi=10)
    metadata, corrected, _, _ = evaluate_validation(data, data, cfg, stored_lumi=10)
    np.testing.assert_allclose(corrected.S, raw.S * 1.95)
    np.testing.assert_allclose(corrected.B, raw.B * 1.26)
    np.testing.assert_allclose(corrected.background_sumw2, raw.background_sumw2 * 1.26**2)
    np.testing.assert_allclose(corrected.background_neff, raw.background_neff)
    np.testing.assert_allclose(corrected.signal_efficiency, raw.signal_efficiency)
    np.testing.assert_allclose(corrected.background_efficiency, raw.background_efficiency)
    assert metadata['auc_weighted'] == pytest.approx(old['auc_weighted'])
    frozen = json.loads(json.dumps(metadata))
    applied = evaluate_frozen(data, data, frozen, cfg, stored_lumi=10, split='val')
    for row in applied.itertuples():
        keep = data.bdt_score >= row.threshold
        assert row.S == pytest.approx(data.loc[keep & (data.target == 1), 'sample_weight'].sum() * 10 * 1.95)
        assert row.B == pytest.approx(data.loc[keep & (data.target == 0), 'sample_weight'].sum() * 10 * 1.26)
    with pytest.raises(ValueError, match='configuration changed'):
        evaluate_frozen(data, data, frozen, raw_cfg, stored_lumi=10, split='val')
    # Old releases keep their original uncorrected convention.
    del old['config']['signal_k_factor'], old['config']['background_k_factor']
    legacy = evaluate_frozen(data, data, old, raw_cfg, stored_lumi=10, split='val')
    for row in legacy.itertuples():
        assert row.S == pytest.approx(old['operating_points'][row.objective]['metrics']['S'])
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize('factor', [0, -1, np.nan, np.inf])
@pytest.mark.parametrize('field', ['signal_k_factor', 'background_k_factor'])
def test_invalid_k_factors_are_rejected(field, factor):
    with pytest.raises(ValueError, match='k-factors'):
        EvaluationConfig(**{field: factor})


@pytest.mark.parametrize("feature", ["sample_weight", "fit_weight", "target", "xs_pb", "event_id"])
def test_fit_and_physics_bookkeeping_cannot_be_features(feature):
    with pytest.raises(ValueError, match="cannot be ML features"):
        make_xyw(frame(), [feature])
