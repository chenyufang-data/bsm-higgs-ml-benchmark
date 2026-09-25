"""Truth tagging: efficiency rules, outcome enumeration, closure statistics and the closure command."""

import itertools
import json
from pathlib import Path

import awkward as ak
import nbformat
import numpy as np
import pandas as pd
import pytest
from hepml_compact.config import load_profile
from hepml_compact.parquet_writer import sha256_file

from hepml.cli import main
from hepml.commands.truth_tag_closure import ShapeAccumulator, TagRateAccumulator
from hepml.domain.splits import extend_registry, extend_sample_membership
from hepml.domain.truth_tagging import (
    TaggingModel,
    compile_efficiency,
    direct_roles,
    draw_outcomes,
    enumerate_outcomes,
    event_uniforms,
    outcomes,
    sampling_chi2,
    state_probabilities,
)
from tests.truth_tag_world import MODEL, SETTINGS, STUDY, build_world, simulate


def test_efficiency_expressions_are_safe_and_bounded():
    b = compile_efficiency(SETTINGS["efficiencies"]["b_tag"]["b"])
    assert b([100.0])[0] == pytest.approx(0.80 * np.tanh(0.3) * 30 / (1 + 8.6))
    assert compile_efficiency("0.22")([30.0, 300.0]).tolist() == [0.22, 0.22]
    for text in ("__import__('os')", "pt.real", "open('x')", "eta * 0.1", "tanh(pt, pt)", "[pt]"):
        with pytest.raises(ValueError):
            compile_efficiency(text)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compile_efficiency("0.01*pt")


def test_model_picks_efficiencies_by_absolute_flavour_label():
    eps_b, eps_c = MODEL.efficiencies(np.array([5, -5, 4, -4, 21, 1, 0]), np.full(7, 50.0))
    assert eps_c.tolist() == [0.01, 0.01, 0.22, 0.22, 0.001, 0.001, 0.001]
    assert eps_b[0] == eps_b[1] and eps_b[2] == eps_b[3] and eps_b[4] == pytest.approx(0.002 + 7.3e-6 * 50)
    states = state_probabilities(eps_b, eps_c)
    np.testing.assert_allclose(states.sum(axis=1), 1.0)
    assert MODEL.observed_states(np.array([0, 1, 16, 17])).tolist() == [0, 1, 2, 3]


def brute_force(pt, flavour, model):
    """Every joint tag state of the acceptance jets: pass probability and probability per (b1, b2, c1)."""
    eps_b, eps_c = model.efficiencies(flavour, pt)
    states = state_probabilities(eps_b, eps_c)
    per_outcome, total = {}, 0.0
    for joint in itertools.product(range(4), repeat=len(pt)):
        probability = np.prod([states[j, s] for j, s in enumerate(joint)])
        b_only = [j for j, s in enumerate(joint) if s == 1]
        c_only = [j for j, s in enumerate(joint) if s == 2]
        if len(b_only) == 2 and len(c_only) == 1:
            b1, b2 = sorted(b_only, key=lambda j: -pt[j])  # stable: ties keep jet order
            per_outcome[(b1, b2, c_only[0])] = per_outcome.get((b1, b2, c_only[0]), 0.0) + probability
            total += probability
    return total, per_outcome


def test_enumeration_equals_the_sum_over_every_joint_tag_state():
    rng = np.random.default_rng(3)
    events = []
    for n in (3, 4, 5, 6):
        pt = rng.uniform(26, 300, n)
        pt[1] = pt[0] if n > 4 else pt[1]  # a pT tie
        events.append((pt, rng.choice([5, 4, 21, 1], n), rng.integers(0, 4, n) * 0 + 1))
    # An extra jet outside acceptance in every event must be ignored.
    pts = [np.r_[pt, 20.0] for pt, _, _ in events]
    flavours = [np.r_[f, 5] for _, f, _ in events]
    offsets = np.r_[0, np.cumsum([len(p) for p in pts])]
    flat_pt, flat_flavour = np.concatenate(pts), np.concatenate(flavours)
    flat_eta = np.zeros_like(flat_pt)
    entries, per_event = enumerate_outcomes(offsets, flat_pt, flat_eta, flat_flavour, np.zeros(len(flat_pt), int), MODEL)
    for e, (pt, flavour) in enumerate(zip(pts, flavours)):
        inside = pt > 25
        total, expected = brute_force(pt[inside], flavour[inside], MODEL)
        assert per_event["pass_probability"][e] == pytest.approx(total, rel=1e-12)
        mine = entries["event"] == e
        got = {(b1 - offsets[e], b2 - offsets[e], c1 - offsets[e]): p for b1, b2, c1, p in
               zip(entries["b1"][mine], entries["b2"][mine], entries["c1"][mine], entries["probability"][mine])}
        assert set(got) == set(expected)
        for key, value in expected.items():
            assert got[key] == pytest.approx(value, rel=1e-12)
    assert len(outcomes(5)) == 30 and not per_event["direct_pass"].any()


def test_direct_outcome_reproduces_the_compactor_selection(jets_recipe):
    rows = [
        ([50.0, 50.0, 40.0, 15.0], [0.0, 0.0, 0.0, 3.0], [1, 1, 16, 0]),  # b pT tie, extra jet outside
        ([50.0, 50.0, 25.0], [0.0, 0.0, 0.0], [1, 1, 16]),  # strict pT cut
        ([50.0, 50.0, 40.0], [0.0, 0.0, 2.5], [1, 1, 16]),  # strict eta cut
        ([50.0, 50.0, 40.0], [0.0, 0.0, 0.0], [1, 1, 17]),  # both bits count as neither
        ([50.0, 50.0, 40.0, 30.0], [0.0, 0.0, 0.0, 0.0], [1, 1, 16, 1]),  # a third b
        ([30.0, 80.0, 40.0, 60.0], [0.0, 0.0, 0.0, 1.0], [1, 1, 16, 17]),  # reordered b, neutral both-bit jet
    ]
    arrays = ak.Array({"Jet/Jet.PT": [r[0] for r in rows], "Jet/Jet.Eta": [r[1] for r in rows],
                       "Jet/Jet.Phi": [[0.0] * len(r[0]) for r in rows], "Jet/Jet.Mass": [[0.0] * len(r[0]) for r in rows],
                       "Jet/Jet.BTag": [r[2] for r in rows]})
    selected = load_profile(jets_recipe).process_chunk(arrays, 1).frame
    offsets = np.r_[0, np.cumsum([len(r[0]) for r in rows])]
    pt, eta, btag = (np.concatenate([r[k] for r in rows]) for k in range(3))
    passes, roles = direct_roles(offsets, pt, eta, btag.astype(int), MODEL)
    assert np.flatnonzero(passes).tolist() == [0, 5]
    for role in ("b1", "b2", "c1"):
        assert roles[role][passes].tolist() == selected[f"role_{role}"].tolist()
    entries, per_event = enumerate_outcomes(offsets, pt, eta, np.full(len(pt), 5), btag.astype(int), MODEL)
    assert per_event["direct_pass"].tolist() == passes.tolist()
    chosen = entries["direct"]
    assert (entries["b1"][chosen] - offsets[entries["event"][chosen]]).tolist() == roles["b1"][passes].tolist()
    assert (entries["c1"][chosen] - offsets[entries["event"][chosen]]).tolist() == roles["c1"][passes].tolist()


def test_closure_statistics_accept_the_true_model_and_reject_a_wrong_one():
    wrong = TaggingModel.from_settings({**SETTINGS, "efficiencies": {
        **SETTINGS["efficiencies"], "c_tag": {**SETTINGS["efficiencies"]["c_tag"], "c": "0.30"}}})
    for truth, expect_pass in ((MODEL, True), (wrong, False)):
        offsets, pt, eta, flavour, btag = simulate(MODEL, 40000, 11, efficiency_model=truth)
        rates = TagRateAccumulator(MODEL, SETTINGS["closure"]["tag_rates"]["pt_edges"], 10)
        inside = MODEL.in_acceptance(pt, eta)
        rates.add(pt[inside], flavour[inside], btag[inside])
        _, tests = rates.results(0.001)
        assert tests.set_index("flavour").loc["c", "passed"] == expect_pass
        entries, per_event = enumerate_outcomes(offsets, pt, eta, flavour, btag, MODEL)
        p = per_event["pass_probability"]
        pull = (per_event["direct_pass"].sum() - p.sum()) / np.sqrt((p * (1 - p)).sum())
        assert (abs(pull) < 3) == expect_pass
        values = pt[entries["c1"]]
        shape = ShapeAccumulator(np.quantile(values[entries["direct"]], np.linspace(0, 1, 11)[1:-1]))
        shape.add(values, entries["event"], entries["probability"], entries["direct"], len(p))
        chi2, dof, p_value, sigma = shape.result()
        assert dof == 10 and (sigma > 0).all()
        if expect_pass:
            assert p_value > 0.001


def test_drawn_outcomes_follow_their_probabilities():
    pt, flavour = np.array([60.0, 50.0, 40.0, 30.0]), np.array([5, 4, 4, 21])
    n = 20000
    offsets = np.arange(n + 1) * 4
    tiled = dict(pt=np.tile(pt, n), eta=np.zeros(4 * n), flavour=np.tile(flavour, n), btag=np.zeros(4 * n, int))
    entries, per_event = enumerate_outcomes(offsets[:2], pt, np.zeros(4), flavour, np.zeros(4, int), MODEL)
    share = entries["probability"] / per_event["pass_probability"][0]
    # Evenly spaced uniforms: every outcome is drawn in proportion to its probability.
    roles, pass_probability = draw_outcomes(offsets, tiled["pt"], tiled["eta"], tiled["flavour"], tiled["btag"], MODEL,
                                            (np.arange(n) + 0.5) / n)
    np.testing.assert_allclose(pass_probability, per_event["pass_probability"][0])
    local = [(b1 % 4, b2 % 4, c1 % 4) for b1, b2, c1 in zip(roles["b1"], roles["b2"], roles["c1"])]
    for index, key in enumerate(zip(entries["b1"], entries["b2"], entries["c1"])):
        assert abs(local.count(tuple(int(k) for k in key)) / n - share[index]) <= 1 / n + 1e-12
    uniforms = event_uniforms(["a:1", "a:2", "b:1"], 7)
    assert ((uniforms >= 0) & (uniforms < 1)).all() and uniforms.tolist() == event_uniforms(["a:1", "a:2", "b:1"], 7).tolist()
    assert uniforms.tolist() != event_uniforms(["a:1", "a:2", "b:1"], 8).tolist()
    # The sampling statistic: exact draws give a small chi2, a biased draw a large one.
    rng = np.random.default_rng(5)
    expected = rng.dirichlet(np.ones(4), 3000) * 0.2
    weight = expected.sum(axis=1)
    fair = np.array([rng.choice(4, p=row / row.sum()) for row in expected])
    chi2, dof = sampling_chi2(expected, weight, fair)
    assert dof == 3 and chi2 < 25  # each event's weight lands in some bin: K - 1 degrees of freedom
    assert sampling_chi2(expected, weight, np.zeros(3000, int))[0] > 1000


def test_membership_extension_keeps_every_frozen_assignment():
    base = extend_registry(pd.DataFrame(), pd.DataFrame(dict(event_id=[f"e{i}" for i in range(50)], sample_key="s")))
    grown = [f"e{i}" for i in range(50)] + [f"n{i}" for i in range(500)]
    extended = extend_sample_membership(base, grown[::-1], "s")
    merged = base.merge(extended, on="event_id", suffixes=("", "_new"))
    assert len(merged) == 50 and (merged.split == merged.split_new).all()
    added = extended[~extended.event_id.isin(base.event_id)]
    assert added.split.value_counts().to_dict() == {"train": 400, "val": 50, "test": 50}
    pd.testing.assert_frame_equal(extended, extend_sample_membership(base, grown, "s"))  # order-free
    with pytest.raises(ValueError, match="absent"):
        extend_sample_membership(base, grown[1:], "s")


@pytest.mark.slow
def test_closure_command_on_synthetic_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    truth_root, direct_root, registry = build_world(tmp_path / "world", [
        ("backgrounds", "bkg_bbc", "background", [5, 5, 4, 21], 6000, None, None, 1.0, 1.0),
        ("backgrounds", "bkg_ccj", "background", [5, 4, 21, 1], 12000, None, None, 1.0, 1.0),
        ("rho04", "sig_m300_rho04", "signal", [5, 5, 4, 2], 3000, 300, 0.4, 1.0, 1.0)])

    report, extended = tmp_path / "report", tmp_path / "registry_truth_tag"
    argv = ["truth-tag-closure", "--study", str(STUDY), "--truth-tag-root", str(truth_root),
            "--direct-root", str(direct_root), "--registry-dir", str(registry), "--registry-out", str(extended),
            "--outdir", str(report)]
    main(argv)
    status = json.loads((report / "provenance/status.json").read_text())
    assert status["status"] == "complete"
    assert all(sha256_file(Path(p)) == h for p, h in status["artifacts"].items())
    summary = json.loads((report / "provenance/summary.json").read_text())
    assert summary["adopted"] and all(summary["gates"].values()) and not summary["test_evaluated"]
    consistency = pd.read_csv(report / "tables/consistency.csv")
    assert consistency.passed.all() and (consistency.reproduced > 0).all()
    yields = pd.read_csv(report / "tables/yields.csv")
    assert set(yields["sample"]) == {"bkg_bbc", "bkg_ccj", "sig_m300_rho04"} and (yields.gain > 1).all()
    shapes = pd.read_csv(report / "tables/shapes.csv")
    assert len(shapes[shapes.gated]) == 15 and shapes[shapes.gated].passed.all()
    assignments = pd.read_parquet(extended / "assignments.parquet")
    frozen = pd.read_parquet(registry / "assignments.parquet").merge(assignments, on="event_id", suffixes=("", "_new"))
    assert (frozen.split == frozen.split_new).all()
    assert len(assignments) == sum(json.loads(p.read_text())["cutflow"]["selected"]
                                   for p in truth_root.glob("*/*.export.json"))
    notebook = nbformat.read(report / "report.ipynb", as_version=4)
    assert len([o for c in notebook.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]) == 3
    assert {p.name for p in (report / "figures").glob("*.pdf")} == {"tag_rates.pdf", "yields.pdf", "shapes.pdf"}
    with pytest.raises(FileExistsError):
        main(argv)
