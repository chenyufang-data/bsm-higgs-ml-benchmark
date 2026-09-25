"""Paired Phase 5 comparisons: alignment, fixed-background working points and paired bootstrap."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve

from hepml.application.comparison import (
    align_predictions,
    combine_outcomes,
    joint_paired_bootstrap,
    noninferiority,
    paired_bootstrap,
    paired_metrics,
    resample_counts,
    selection_support,
    working_point,
)


def predictions(seed=0, n=400, shift=0.0):
    rng = np.random.default_rng(seed)
    target = np.r_[np.ones(n // 4, dtype=int), np.zeros(n - n // 4, dtype=int)]
    sample = np.where(target == 1, "sig", np.where(np.arange(n) % 3 == 0, "bkg_rare", "bkg_common"))
    weight = np.where(sample == "sig", 1.0, np.where(sample == "bkg_rare", 50.0, 2.0))
    score = np.clip(rng.normal(0.35 + 0.3 * target + shift, 0.15), 0, 1)
    return pd.DataFrame(dict(event_id=[f"e{i:04d}" for i in range(n)], sample=sample, target=target,
                             sample_weight=weight, bdt_score=score))


def test_alignment_matches_events_by_identity_not_position():
    a = predictions()
    b = a.assign(bdt_score=a.bdt_score * 0.9).sample(frac=1, random_state=4)
    frame = align_predictions({"reference": a, "candidate": b})
    np.testing.assert_allclose(frame.candidate, frame.reference * 0.9)
    with pytest.raises(ValueError, match="different set"):
        align_predictions({"reference": a, "candidate": b.iloc[1:]})
    with pytest.raises(ValueError, match="sample_weight differs"):
        align_predictions({"reference": a, "candidate": b.assign(sample_weight=b.sample_weight * 2)})
    with pytest.raises(ValueError, match="duplicate"):
        align_predictions({"reference": a, "candidate": pd.concat([b, b.iloc[:1]])})


def test_working_point_interpolates_the_weighted_roc():
    frame = predictions()
    point = working_point(frame.target, frame.bdt_score, frame.sample_weight, background_efficiency=0.25)
    fpr, tpr, thresholds = roc_curve(frame.target, frame.bdt_score, sample_weight=frame.sample_weight,
                                     drop_intermediate=False)
    assert point["signal_efficiency"] == pytest.approx(np.interp(0.25, fpr, tpr), abs=1e-12)
    kept = frame.bdt_score >= point["threshold"]
    background = frame.sample_weight * (frame.target == 0)
    assert background[kept].sum() / background.sum() == pytest.approx(point["achieved_background_efficiency"])
    assert point["achieved_background_efficiency"] <= 0.25


def test_interpolation_follows_the_roc_path_not_its_upper_envelope():
    # Path: (0, 1/3) at 0.9 -> (0.5, 1/3) at 0.7 -> (0.5, 2/3) at 0.6 -> (1, 1) at 0.5.
    # At 0.25 the path runs flat at 1/3; the later signal-only step must not leak in.
    target = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.7, 0.6, 0.5, 0.5])
    point = working_point(target, scores, np.ones(5), background_efficiency=0.25)
    assert point["signal_efficiency"] == pytest.approx(1 / 3)
    assert point["threshold"] == 0.9 and point["achieved_background_efficiency"] == 0.0
    assert point["next_background_efficiency"] == 0.5
    assert working_point(target, scores, np.ones(5), background_efficiency=0.5)["signal_efficiency"] == pytest.approx(2 / 3)


def test_resampling_keeps_every_process_size():
    frame = predictions()
    counts = resample_counts(frame["sample"].to_numpy(), np.random.default_rng(1))
    assert frame.assign(c=counts).groupby("sample").c.sum().to_dict() == frame.groupby("sample").size().to_dict()


def test_paired_bootstrap_shares_resamples_across_models():
    a = predictions()
    frame = align_predictions({"reference": a, "same": a, "candidate": predictions(seed=0, shift=-0.05)})
    weights = frame.sample_weight.to_numpy()
    replicas = paired_bootstrap(frame, ["reference", "same", "candidate"], weights,
                                background_efficiencies=[0.25], replicates=20, seed=7)
    wide = replicas.pivot(index="replicate", columns="model", values="signal_efficiency_at_0.25")
    np.testing.assert_array_equal(wide["reference"], wide["same"])  # identical models: zero difference
    again = paired_bootstrap(frame, ["reference", "candidate"], weights, background_efficiencies=[0.25],
                             replicates=20, seed=7)
    pd.testing.assert_frame_equal(
        again.pivot(index="replicate", columns="model", values="auc_weighted")[["reference", "candidate"]],
        replicas.pivot(index="replicate", columns="model", values="auc_weighted")[["reference", "candidate"]])
    # Replica 0 equals the metrics recomputed with the same multiplicities.
    counts = resample_counts(frame["sample"].to_numpy(), np.random.default_rng(7))
    direct = paired_metrics(frame, ["candidate"], weights, background_efficiencies=[0.25], counts=counts)
    assert direct["signal_efficiency_at_0.25"].iloc[0] == pytest.approx(wide["candidate"].iloc[0])


@pytest.mark.parametrize("candidate_shift, expected", [(0.0, "non-inferior"), (-0.2, "inferior"), (-0.05, "inconclusive")])
def test_noninferiority_outcomes(candidate_shift, expected):
    reference = np.full(1000, 0.80) + np.random.default_rng(3).normal(0, 0.02, 1000)
    candidate = reference * (1 + candidate_shift) + np.random.default_rng(4).normal(0, 0.005, 1000)
    result = noninferiority(0.80, 0.80 * (1 + candidate_shift), reference, candidate, margin=0.05, relative=True,
                            confidence=0.95)
    assert result["outcome"] == expected
    assert result["loss"] == pytest.approx(-candidate_shift)
    absolute = noninferiority(0.90, 0.905, np.full(5, 0.9), np.full(5, 0.905), margin=0.01, relative=False,
                              confidence=0.95)
    assert absolute["loss"] == pytest.approx(-0.005) and absolute["outcome"] == "non-inferior"


def test_joint_bootstrap_shares_common_background_draws_across_hypotheses():
    first = predictions(seed=1)
    second = predictions(seed=2)
    # The same background events appear at both hypotheses; signal events differ.
    signal = second.target == 1
    second.loc[signal, "event_id"] = "s" + second.loc[signal, "event_id"]
    second.loc[signal, "sample"] = "sig_other"
    groups = {mass: (align_predictions({"a": frame, "b": frame.assign(bdt_score=frame.bdt_score ** 2)}), ["a", "b"],
                     frame.sort_values("event_id").sample_weight.to_numpy())
              for mass, frame in [(200, first), (400, second)]}
    replicas = joint_paired_bootstrap(groups, background_efficiencies=[0.25], replicates=5, seed=3)
    assert set(replicas.group) == {200, 400} and len(replicas) == 5 * 2 * 2
    # Tuple names, such as (mass, seed), are kept whole.
    keyed = joint_paired_bootstrap({(mass, 42): group for mass, group in groups.items()},
                                   background_efficiencies=[0.25], replicates=5, seed=3)
    assert set(keyed.group) == {(200, 42), (400, 42)}
    np.testing.assert_array_equal(keyed.auc_weighted, replicas.auc_weighted)
    # Rebuild the draws: common background events receive one multiplicity in both groups.
    union = pd.concat([groups[200][0][["event_id", "sample"]], groups[400][0][["event_id", "sample"]]]).drop_duplicates()
    counts = pd.Series(resample_counts(union["sample"].to_numpy(), np.random.default_rng(3)), index=union.event_id)
    for mass, (frame, models, weights) in groups.items():
        direct = paired_metrics(frame, models, weights, background_efficiencies=[0.25],
                                counts=counts.loc[frame.event_id].to_numpy())
        saved = replicas[(replicas.group == mass) & (replicas.replicate == 0)].reset_index(drop=True)
        np.testing.assert_allclose(saved.auc_weighted, direct.auc_weighted)
    with pytest.raises(ValueError, match="different processes"):
        clash = first.assign(sample=np.where(first.target == 0, "renamed", first["sample"]))
        joint_paired_bootstrap({1: groups[200], 2: (align_predictions({"a": clash, "b": clash}), ["a"],
                                                    clash.sort_values("event_id").sample_weight.to_numpy())},
                               background_efficiencies=[0.25], replicates=2, seed=0)


def test_selection_support_counts_background_effective_events():
    support = selection_support([1, 0, 0, 0], [0.9, 0.8, 0.7, 0.1], [1.0, 2.0, 2.0, 5.0], 0.5)
    assert support == dict(signal_mc=1, background_mc=2, background_neff=pytest.approx(2.0))


def test_combined_gate_needs_every_criterion():
    assert combine_outcomes(["non-inferior", "non-inferior"]) == "non-inferior"
    assert combine_outcomes(["non-inferior", "inconclusive"]) == "inconclusive"
    assert combine_outcomes(["inconclusive", "inferior"]) == "inferior"
    with pytest.raises(ValueError):
        combine_outcomes([])
