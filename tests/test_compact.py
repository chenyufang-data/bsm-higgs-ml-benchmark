"""Data identity, resumability, physics compatibility, and study isolation."""

import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot
from hepml_compact.config import load_profile
from hepml_compact.export import export_sample
from hepml_compact.parquet_writer import validate_export

from hepml.adapters.dataset_files import benchmark_columns, discover_samples
from hepml.adapters.study_loader import load_study
from hepml.domain.dataset import _clean_df, _stratified_split

STUDY_RECIPE = Path(__file__).resolve().parents[1] / "studies/cg_bbc/extraction.yaml"


def _export(out, root_sample, recipe, **kwargs):
    options = dict(sample="sig_200", kind="signal", mass=200, xs_pb=2.0, step_size=500)
    options.update(kwargs)
    return export_sample([root_sample], out, load_profile(recipe), **options)


def _read(out):
    return pd.concat([pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))], ignore_index=True)


def test_study_selects_known_synthetic_events_and_retains_kinematics(root_sample, jets_recipe):
    study = load_profile(jets_recipe)
    with uproot.open(root_sample) as root:
        arrays = root["Delphes"].arrays(study.required_branches + study.optional_branches)
    result = study.process_chunk(arrays, 1)
    np.testing.assert_array_equal(result.offsets, np.flatnonzero(np.arange(2000) % 5 != 0))
    assert len(result.frame) == 1600
    frame = load_study().plugin.derive_features(result.frame)
    assert (frame["ptb1"] >= frame["ptb2"]).all()
    assert len(load_study().plugin.FEATURES) == 15
    assert set(load_study().plugin.RETAINED_COLUMNS).issubset(frame)
    assert np.isfinite(frame[load_study().plugin.FEATURES].to_numpy()).all()


def test_resume_matches_uninterrupted_and_preserves_files(tmp_path, root_sample, jets_recipe):
    resumed, fresh = tmp_path / "resumed", tmp_path / "fresh"
    partial = _export(resumed, root_sample, jets_recipe, max_chunks=1)
    first = resumed / partial["chunks"][0]["file"]
    mtime = first.stat().st_mtime_ns
    assert partial["status"] == "incomplete"
    assert not (resumed / "signal_sig_200.meta.json").exists()
    with pytest.raises(ValueError, match="Incomplete compact"):
        discover_samples(resumed)
    complete = _export(resumed, root_sample, jets_recipe)
    direct = _export(fresh, root_sample, jets_recipe)
    assert first.stat().st_mtime_ns == mtime
    assert complete["status"] == "complete"
    assert complete["chunks"] == direct["chunks"]
    pd.testing.assert_frame_equal(_read(resumed), _read(fresh), check_exact=True)
    assert complete["processed_entries"] == 2000
    validate_export(resumed, complete)


def test_chunk_size_does_not_change_event_identity_or_values(tmp_path, root_sample, jets_recipe):
    _export(tmp_path / "a", root_sample, jets_recipe, step_size=333)
    _export(tmp_path / "b", root_sample, jets_recipe, step_size=777)
    pd.testing.assert_frame_equal(_read(tmp_path / "a"), _read(tmp_path / "b"), check_exact=True)
    assert _read(tmp_path / "a")["event_id"].is_unique


def test_missing_xs_can_be_added_without_reextracting(tmp_path, monkeypatch, root_sample, jets_recipe):
    _export(tmp_path, root_sample, jets_recipe, xs_pb=None)
    with pytest.raises(ValueError, match="missing merged cross section"):
        discover_samples(tmp_path)
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.parquet")}
    study = load_profile(jets_recipe)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed event data must not be read again")

    monkeypatch.setattr(study, "process_chunk", forbidden)
    export_sample([root_sample], tmp_path, study, sample="sig_200", kind="signal", xs_pb=2.0, step_size=500)
    assert before == {p.name: p.stat().st_mtime_ns for p in tmp_path.glob("*.parquet")}
    signals, _ = discover_samples(tmp_path)
    assert signals["200"]["xs_pb"] == 2.0
    assert signals["200"]["n_events_total"] == 2000


def test_corrupted_and_unexpected_shards_are_rejected(tmp_path, root_sample, jets_recipe):
    result = _export(tmp_path, root_sample, jets_recipe)
    extra = tmp_path / "signal_sig_200.part99999.parquet"
    extra.write_bytes(b"not a shard")
    with pytest.raises(ValueError, match="Unexpected Parquet"):
        discover_samples(tmp_path)
    extra.unlink()
    shard = tmp_path / result["chunks"][0]["file"]
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupted shard"):
        _export(tmp_path, root_sample, jets_recipe)


def test_changed_inputs_settings_and_duplicate_sources_fail(tmp_path, root_sample, jets_recipe):
    _export(tmp_path, root_sample, jets_recipe, max_chunks=1)
    with pytest.raises(ValueError, match="changed"):
        _export(tmp_path, root_sample, jets_recipe, step_size=100)
    with pytest.raises(ValueError, match="Existing mass differs"):
        _export(tmp_path, root_sample, jets_recipe, mass=250)
    with pytest.raises(ValueError, match="Duplicate ROOT source"):
        export_sample(
            [root_sample, root_sample], tmp_path / "duplicate", load_profile(jets_recipe), sample="x", kind="signal"
        )


def test_retained_kinematics_do_not_change_split_membership(tmp_path, root_sample, jets_recipe):
    _export(tmp_path, root_sample, jets_recipe)
    compact = load_study().plugin.derive_features(_read(tmp_path))
    old = _clean_df(compact, load_study().plugin.FEATURES).drop(
        columns=["event_id", "source_id", "source_entry", "sample_key"]
    )
    enriched = _clean_df(compact, load_study().plugin.FEATURES, load_study().plugin.RETAINED_COLUMNS)
    # Only splitting behavior is under test; create both classes identically.
    old["target"] = enriched["target"] = np.arange(len(old)) % 2
    old["sample_weight"] = enriched["sample_weight"] = np.linspace(1, 3, len(old))
    for a, b in zip(_stratified_split(old, 0.1, 0.1, 42), _stratified_split(enriched, 0.1, 0.1, 42)):
        pd.testing.assert_frame_equal(a, b[a.columns], check_exact=True)


def test_empty_chunks_record_offsets_and_preselection_count(tmp_path, root_sample, jets_recipe):
    source = tmp_path / "tiny.root"
    jets = ak.Array([[], [], [60.0, 50.0, 40.0], [], [70.0, 60.0, 50.0]])
    with uproot.recreate(source) as root:
        root.mktree(
            "Delphes",
            {
                "Jet/Jet.PT": jets,
                "Jet/Jet.Eta": ak.zeros_like(jets),
                "Jet/Jet.Phi": ak.zeros_like(jets),
                "Jet/Jet.Mass": ak.ones_like(jets),
                "Jet/Jet.BTag": ak.Array([[], [], [1, 1, 16], [], [1, 1, 16]]),
            },
        )
    result = export_sample(
        [source], tmp_path / "out", load_profile(jets_recipe), sample="tiny", kind="signal", step_size=2
    )
    assert result["chunks"][0]["file"] is None
    assert result["processed_entries"] == 5
    assert _read(tmp_path / "out")["source_entry"].tolist() == [2, 4]


def test_missing_branches_fail_before_export(tmp_path, root_sample, jets_recipe):
    source = tmp_path / "missing.root"
    with uproot.recreate(source) as root:
        root.mktree("Delphes", {"Jet/Jet.PT": ak.Array([[20.0]])})
    with pytest.raises(ValueError, match="missing required branches"):
        export_sample([source], tmp_path / "out", load_profile(jets_recipe), sample="tiny", kind="signal")
    assert not (tmp_path / "out").exists()


def test_other_study_needs_no_exporter_change(tmp_path, root_sample, jets_config):
    import yaml

    directory = tmp_path / "other_study"
    directory.mkdir()
    config = jets_config
    config["name"] = "other"
    config["selection"] = {}
    (directory / "extraction.yaml").write_text(yaml.safe_dump(config))
    result = export_sample([root_sample], tmp_path / "out", load_profile(directory), sample="other", kind="background")
    assert result["selected_events"] == 2000
    assert result["object_schema"]["name"] == "other"
    assert "ptb1" not in _read(tmp_path / "out")
    assert "jets_pt" in _read(tmp_path / "out")


def test_mutated_manifest_cannot_claim_complete(tmp_path, root_sample, jets_recipe):
    result = _export(tmp_path, root_sample, jets_recipe)
    result["chunks"] = result["chunks"][:-1]
    with pytest.raises(ValueError, match="missing entry ranges"):
        validate_export(tmp_path, result)


def test_missing_or_modified_sidecar_is_rejected(tmp_path, root_sample, jets_recipe):
    _export(tmp_path, root_sample, jets_recipe)
    sidecar = tmp_path / "signal_sig_200.meta.json"
    metadata = json.loads(sidecar.read_text())
    metadata["xs_pb"] = 10000
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not match"):
        discover_samples(tmp_path)
    sidecar.unlink()
    with pytest.raises(ValueError, match="Missing compact sidecar"):
        discover_samples(tmp_path)


def test_preparation_persists_disjoint_assignments_and_exact_weights(prepared_dataset):
    dataset = pd.read_parquet(prepared_dataset / "dataset_sig200_vs_bkg.parquet")
    meta = json.loads((prepared_dataset / "splits/split_sig200.meta.json").read_text())
    assignments = pd.read_parquet(prepared_dataset / "splits/assignments_sig200.parquet")
    assert assignments["event_id"].is_unique
    assert set(assignments["event_id"]) == set(dataset["event_id"])
    assert len(assignments) == meta["counts"]["n_all"] == 3200
    for name in ("train", "val", "test"):
        split = pd.read_parquet(prepared_dataset / f"splits/{name}_sig200.parquet")
        expected = assignments.loc[assignments["split"] == name].sort_values("row_in_split")
        assert split["event_id"].tolist() == expected["event_id"].tolist()
        np.testing.assert_array_equal(split["sample_weight"], meta["lumi"] * split["xs_pb"] / 2000)
    assert set(dataset["source_entry"] % 5) == {1, 2, 3, 4}


def test_preparation_rejects_non_compact_sidecars(tmp_path):
    (tmp_path / "old.meta.json").write_text(json.dumps({"schema": 2, "label": 1, "mass": 200}))
    with pytest.raises(ValueError, match="expected a current compact export"):
        discover_samples(tmp_path)


def test_prepared_benchmarks_keep_event_values_and_selected_objects_only(tmp_path):
    """Constituents, particle flow and generator truth stay in the compact export."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    lists = pa.list_(pa.float32())
    table = pa.table({
        "label": pa.array([1], pa.int32()), "met": pa.array([30.0], pa.float32()),
        "jets_pt": pa.array([[50.0, 40.0]], lists), "jets_truth_flavor": pa.array([[5, 4]], pa.list_(pa.int32())),
        "towers_et": pa.array([[1.0]], lists), "eflow_tracks_pt": pa.array([[2.0]], lists),
        "truth_partons_pt": pa.array([[60.0]], lists), "truth_lhe_weights": pa.array([[1.0, 0.9]], lists),
        "role_b1": pa.array([0], pa.int64()), "event_id": pa.array(["e0"]),
    })
    path = tmp_path / "shard.parquet"
    pq.write_table(table, path)
    profile = load_profile(STUDY_RECIPE)
    assert benchmark_columns(path, profile) == ["label", "met", "jets_pt", "role_b1", "event_id"]
