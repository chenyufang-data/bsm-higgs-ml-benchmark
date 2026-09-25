"""Audit statistics, immutable split anchors, and a small executable workflow."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import ks_2samp, wasserstein_distance

from hepml.application.shape_audit import audit_decision, bootstrap_counts, followup_reasons
from hepml.domain.metrics import binomial_interval, bootstrap_wasserstein, shape_distances
from hepml.domain.splits import extend_registry

REPO = Path(__file__).resolve().parents[1]


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


@pytest.mark.slow
@pytest.mark.parametrize("expand", [False, True])
def test_shape_audit_cli_keeps_test_events_out(tmp_path, prepared_dataset, root_factory, monkeypatch, expand,
                                               jets_recipe):
    from hepml.cli import main

    study = REPO / "studies/cg_bbc"
    compact = tmp_path / "audit_compact"
    # One mass / two couplings keeps this integration fixture small. Production
    # defaults remain the preregistered two-mass, three-coupling design.
    for mass in ([400, 300] if expand else [400]):
        for rho, seed in ((.1, 42), (.5, 51)):
            path = compact / f"rho0{round(rho * 10)}"
            main(["compact", "--extraction", str(jets_recipe), "--input", str(root_factory(seed=seed, entries=100)),
                  "--sample", f"audit{mass}_rho{round(rho * 10)}", "--kind", "signal", "--mass", str(mass),
                  "--rho-tc", str(rho), "--xs-pb", "2", "--outdir", str(path)])
    if expand:
        # Inject a known, large coupling-associated feature shift to exercise
        # the real statistics -> decision -> conditional loading path.
        from hepml.commands import shape_audit

        original = shape_audit.load_sample_frame

        def shifted(directory, meta, *args, **kwargs):
            result = original(directory, meta, *args, **kwargs)
            if meta["mass"] == 400 and meta["rho_tc"] == .5:
                result["mcb1"] += 100000
            return result

        monkeypatch.setattr(shape_audit, "load_sample_frame", shifted)
    config = yaml.safe_load((study / "validation.yaml").read_text())
    config.update(initial_masses=[400], conditional_masses=[300] if expand else [], couplings=[.1, .5],
                  bootstrap_replicates=12, two_sample_permutations=1, histogram_bins=8,
                  reference_cut_features=["mcb1"], reference_cut_quantiles=[.5])
    config_path = tmp_path / "validation.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output, registry = tmp_path / "audit", tmp_path / "registry"
    argv = ["shape-audit", "--study", str(study), "--config", str(config_path),
            "--compact-root", str(compact), "--anchor-dataset", str(prepared_dataset),
            "--outdir", str(output), "--registry-dir", str(registry)]
    main(argv)
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    decision = json.loads((output / "decision.json").read_text())
    assert len(decision["stages"]) == (2 if expand else 1)
    if expand:
        assert decision["decision"] == "shape-relevant"
        assert (output / "shape_distances_conditional.csv").is_file()
    assignment = pd.read_parquet(registry / "assignments.parquet")
    oof = pd.read_parquet(output / "joint_oof_m400.parquet")
    assert set(oof.event_id).issubset(set(assignment.loc[assignment.split == "train", "event_id"]))
    assert set(oof.event_id).isdisjoint(set(assignment.loc[assignment.split != "train", "event_id"]))
    for filename in ("rho_decision.md", "shape_overlays_initial.pdf", "shape_ratios_initial.pdf",
                     "shape_distances_initial.csv", "production.json", "preregistration.json"):
        assert (output / filename).is_file()
    with pytest.raises(FileExistsError):
        main(argv)
