"""Schema-2 recipes: owner links, storage filters, category maps, weights and compatibility."""

import json
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from hepml_compact.config import load_profile, recipes_compatible, selection_fingerprint
from hepml_compact.environment import mismatches, read_constraints
from hepml_compact.export import export_sample
from hepml_compact.extraction import resolve_owners
from hepml_compact.parquet_writer import validate_export

from hepml.adapters.dataset_files import discover_samples, load_sample_frame
from hepml.adapters.study_loader import load_study
from hepml.application.training import make_xyw
from hepml.cli import main

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "studies/cg_bbc"


def delphes_events(n=5):
    """In-memory events shaped like Delphes output, including TRefArray constituents.

    Event 3 fails the selection (its c-tagged jet is untagged). Every jet owns two
    towers; two more towers per event belong to no jet.
    """
    rows = {key: [] for key in (
        "Jet/Jet.PT", "Jet/Jet.Eta", "Jet/Jet.Phi", "Jet/Jet.Mass", "Jet/Jet.BTag", "Jet/Jet.DeltaEta",
        "Jet/Jet.DeltaPhi", "Jet/Jet.EhadOverEem", "Jet/Jet.Flavor", "Jet/Jet.Constituents", "Tower/Tower.fUniqueID",
        "Tower/Tower.ET", "Tower/Tower.Eta", "Tower/Tower.Phi", "Tower/Tower.E", "Tower/Tower.Eem", "Tower/Tower.Ehad",
        "EFlowTrack/EFlowTrack.PT", "EFlowTrack/EFlowTrack.Eta", "EFlowTrack/EFlowTrack.Phi",
        "EFlowTrack/EFlowTrack.Charge", "EFlowTrack/EFlowTrack.PID", "EFlowPhoton/EFlowPhoton.ET",
        "EFlowPhoton/EFlowPhoton.Eta", "EFlowPhoton/EFlowPhoton.Phi", "EFlowPhoton/EFlowPhoton.E",
        "EFlowNeutralHadron/EFlowNeutralHadron.ET", "EFlowNeutralHadron/EFlowNeutralHadron.Eta",
        "EFlowNeutralHadron/EFlowNeutralHadron.Phi", "EFlowNeutralHadron/EFlowNeutralHadron.E",
        "Particle/Particle.PID", "Particle/Particle.Status", "Particle/Particle.PT", "Particle/Particle.Eta",
        "Particle/Particle.Phi", "Particle/Particle.Mass", "Event/Event.Weight", "Event/Event.CrossSection",
        "Event/Event.Number", "MissingET/MissingET.MET", "MissingET/MissingET.Phi", "ScalarHT/ScalarHT.HT",
        "Electron_size", "Muon_size", "Photon_size", "Weight/Weight.Weight")}
    for i in range(n):
        rows["Jet/Jet.PT"].append([60.0, 55.0, 45.0, 20.0])
        rows["Jet/Jet.Eta"].append([0.1, -0.5, 0.8, 3.0])
        rows["Jet/Jet.Phi"].append([0.0, 2.0, -2.0, 1.0])
        rows["Jet/Jet.Mass"].append([5.0, 4.0, 3.0, 2.0])
        rows["Jet/Jet.BTag"].append([1, 1, 0 if i == 3 else 16, 0])
        rows["Jet/Jet.DeltaEta"].append([0.4, 0.3, 0.2, 0.1])
        rows["Jet/Jet.DeltaPhi"].append([0.3, 0.3, 0.2, 0.1])
        rows["Jet/Jet.EhadOverEem"].append([2.0, 1.5, 3.0, 0.5])
        rows["Jet/Jet.Flavor"].append([5, 5, 4, 21])
        # Unique IDs are shuffled per event so resolution cannot rely on order.
        uids = [1000 * (i + 1) + k for k in (7, 3, 9, 1, 5, 2, 8, 4, 6, 10)]
        rows["Tower/Tower.fUniqueID"].append(uids)
        rows["Jet/Jet.Constituents"].append(
            [{"fName": "", "fSize": 2, "refs": [uids[2 * j + 1], uids[2 * j]]} for j in range(4)])
        rows["Tower/Tower.ET"].append([float(k + 1) for k in range(10)])
        rows["Tower/Tower.Eta"].append([0.0] * 10)
        rows["Tower/Tower.Phi"].append([0.1 * k for k in range(10)])
        rows["Tower/Tower.E"].append([float(k + 1) for k in range(10)])
        rows["Tower/Tower.Eem"].append([0.5 * (k + 1) for k in range(10)])
        rows["Tower/Tower.Ehad"].append([0.5 * (k + 1) for k in range(10)])
        rows["EFlowTrack/EFlowTrack.PT"].append([5.0, 4.0, 3.0, 2.0])
        rows["EFlowTrack/EFlowTrack.Eta"].append([0.0, 0.1, 0.2, 0.3])
        rows["EFlowTrack/EFlowTrack.Phi"].append([0.0, 0.1, 0.2, 0.3])
        rows["EFlowTrack/EFlowTrack.Charge"].append([1, 1, -1, -1])
        rows["EFlowTrack/EFlowTrack.PID"].append([211, -11, 13, -321])
        for kind in ("EFlowPhoton/EFlowPhoton", "EFlowNeutralHadron/EFlowNeutralHadron"):
            rows[f"{kind}.ET"].append([3.0, 1.0])
            rows[f"{kind}.Eta"].append([0.5, -0.5])
            rows[f"{kind}.Phi"].append([1.0, -1.0])
            rows[f"{kind}.E"].append([3.5, 1.2])
        rows["Particle/Particle.PID"].append([21, -4, -37, -5, -4, 5, 211, 22, 2212])
        rows["Particle/Particle.Status"].append([21, 21, 22, 23, 23, 23, 1, 1, 4])
        rows["Particle/Particle.PT"].append([0.0, 0.0, 30.0, 70.0, 60.0, 50.0, 1.0, 1.0, 0.0])
        rows["Particle/Particle.Eta"].append([0.0] * 9)
        rows["Particle/Particle.Phi"].append([0.0] * 9)
        rows["Particle/Particle.Mass"].append([0.0, 1.5, 200.0, 4.7, 1.5, 4.7, 0.14, 0.0, 0.94])
        rows["Event/Event.Weight"].append([2.0 + 0.1 * i])
        rows["Event/Event.CrossSection"].append([0.001 * (i + 1)])
        rows["Event/Event.Number"].append([100 + i])
        rows["MissingET/MissingET.MET"].append([10.0 + i])
        rows["MissingET/MissingET.Phi"].append([0.5])
        rows["ScalarHT/ScalarHT.HT"].append([180.0 + i])
        rows["Electron_size"].append(0)
        rows["Muon_size"].append(i % 2)
        rows["Photon_size"].append(0)
        rows["Weight/Weight.Weight"].append([(2.0 + 0.1 * i) * r for r in (1.0, 1.1 + 0.01 * i, 0.9)])
    return ak.Array({key: ak.Array(values) for key, values in rows.items()})


class InMemorySource:
    """Stands in for RootEventSource: same inspect/open/read/assert_unchanged contract."""

    def __init__(self, arrays):
        self.arrays = arrays

    def inspect(self, path, study, step_size):
        return dict(path=str(path), source_id=f"memory-{Path(path).stem}", root_uuid="memory", size=1, mtime_ns=1,
                    entries=len(self.arrays), chunk_entries=int(step_size), branches=list(study.required_branches),
                    missing_optional=[])

    @contextmanager
    def open(self, source, tree):
        yield self

    def read(self, branches, start, stop):
        return self.arrays[start:stop]

    def assert_unchanged(self, source):
        pass


def test_full_recipe_links_towers_filters_partons_and_maps_track_types():
    events = delphes_events()
    profile = load_profile(STUDY)
    chunk = profile.process_chunk(events, 1)
    frame = chunk.frame
    np.testing.assert_array_equal(chunk.offsets, [0, 1, 2, 4])
    for row in frame.itertuples():
        # Only owned towers are stored, each exactly once, with its owner jet index.
        assert sorted(row.towers_source_index.tolist()) == list(range(8))
        for jet in range(4):
            owned = row.towers_source_index[row.towers_owner == jet]
            assert sorted(owned.tolist()) == [2 * jet, 2 * jet + 1]
        assert row.eflow_tracks_kind.tolist() == [0, 1, 2, 0]
        assert row.truth_partons_status.tolist() == [21, 21, 22, 23, 23, 23]
        assert row.jets_truth_flavor.tolist() == [5, 5, 4, 21]
    assert "eflow_tracks_pid" not in frame and "eflow_tracks_mass" not in frame
    assert frame.event_number.tolist() == [100, 101, 102, 104]
    assert frame.n_muons.tolist() == [0, 1, 0, 0]
    np.testing.assert_allclose(frame.truth_lhe_weights[3], np.asarray(events["Weight/Weight.Weight"][4]), rtol=1e-6)
    # Weight-ratio sums cover every event, including the one the selection rejects.
    weights = ak.to_numpy(events["Weight/Weight.Weight"])
    accumulated = chunk.accumulators["truth_lhe_weights"]
    assert accumulated["events"] == 5
    np.testing.assert_allclose(accumulated["sum"], (weights / weights[:, :1]).sum(axis=0))
    assert set(profile.truth_columns) >= {"jets_truth_flavor", "truth_partons_pid", "truth_lhe_weights"}
    assert all("truth" in column for column in profile.truth_columns)


def test_reference_resolution_rejects_ambiguity_and_accepts_plain_nested_lists():
    uid = ak.Array([[7, 3, 9]])
    plain = ak.Array([[[3], [9, 7]]])
    np.testing.assert_array_equal(resolve_owners(plain, uid, ak.Array([2]), "t")[0], [1, 0, 1])
    records = ak.Array([[{"fName": "", "fSize": 1, "refs": [3]}, {"fName": "", "fSize": 1, "refs": [9]}]])
    np.testing.assert_array_equal(resolve_owners(records, uid, ak.Array([2]), "t")[0], [-1, 0, 1])
    with pytest.raises(ValueError, match="exactly one"):
        resolve_owners(ak.Array([[[3], [42]]]), uid, ak.Array([2]), "t")
    with pytest.raises(ValueError, match="more than one owner"):
        resolve_owners(ak.Array([[[3], [3]]]), uid, ak.Array([2]), "t")
    with pytest.raises(ValueError, match="do not align"):
        resolve_owners(plain, uid, ak.Array([3]), "t")
    with pytest.raises(ValueError, match="exactly one"):  # IDs never match across events
        resolve_owners(ak.Array([[[3]], [[5]]]), ak.Array([[3], [3]]), ak.Array([1, 1]), "t")


def _export(out, arrays, **options):
    settings = dict(sample="sig_m200_rho01", kind="signal", mass=200, xs_pb=0.5, step_size=2,
                    source_reader=InMemorySource(arrays))
    settings.update(options)
    return export_sample([Path("memory.root")], out, load_profile(STUDY), **settings)


def test_export_types_provenance_and_accumulators_survive_resume(tmp_path):
    events = delphes_events()
    source = {"method": "pythia_merged_table", "file": "/prod/m200/r01/mergedxs.txt", "sha256": "ab" * 32,
              "row": "first", "merging_scale": 30.0, "mc_uncertainty_pb": 1e-4}
    partial = _export(tmp_path / "resumed", events, max_chunks=1, xs_source=source)
    assert partial["status"] == "incomplete" and "accumulators" not in partial
    resumed = _export(tmp_path / "resumed", events, xs_source=source)
    direct = _export(tmp_path / "direct", events, xs_source=source)
    assert resumed["accumulators"] == direct["accumulators"]
    assert resumed["accumulators"]["truth_lhe_weights"]["events"] == 5
    meta = json.loads((tmp_path / "direct/signal_sig_m200_rho01.meta.json").read_text())
    assert meta["schema"] == 4 and meta["xs_source"] == source and meta["compactor_version"] == direct["compactor"]["version"]
    assert meta["truth_columns"] == load_profile(STUDY).truth_columns == direct["truth_columns"]
    table = pq.read_table(tmp_path / "direct" / direct["chunks"][0]["file"])
    assert table.schema.field("towers_owner").type == pa.list_(pa.int32())
    assert table.schema.field("truth_lhe_weights").type == pa.list_(pa.float32())
    assert table.schema.field("event_number").type == pa.int64()
    assert table.column_names == list(load_profile(STUDY).output_dtypes)
    validate_export(tmp_path / "direct", direct)
    direct["accumulators"]["truth_lhe_weights"]["sum"][1] += 1.0
    with pytest.raises(ValueError, match="accumulators"):
        validate_export(tmp_path / "direct", direct)
    with pytest.raises(ValueError, match="Existing xs_source differs"):
        _export(tmp_path / "direct", events, xs_source={"method": "supplied_in_manifest"})


def test_older_export_schema_remains_readable(tmp_path):
    manifest = _export(tmp_path, delphes_events())
    validate_export(tmp_path, {**manifest, "export_schema": 2})
    with pytest.raises(ValueError, match="Unsupported compact export schema"):
        validate_export(tmp_path, {**manifest, "export_schema": 9})


@pytest.mark.parametrize("mutation, message", [
    (lambda c: c["collections"]["jets"]["fields"]["flavor"].update(truth=True), "must start with 'truth_'"),
    (lambda c: c["event_fields"]["met"].update(truth=True), "must start with 'truth_'"),
    (lambda c: c["collections"].update(partons=c["collections"].pop("truth_partons")), "must start with 'truth_'"),
    (lambda c: c["collections"]["jets"]["fields"]["pt"].update(map_abs={1: 2}, map_default=0), "map_abs"),
    (lambda c: c["collections"]["towers"].pop("owner"), "keep: owned requires an owner"),
    (lambda c: c["collections"]["jets"].update(keep={"pt": {"gt": 10}}), "no keep or owner"),
    (lambda c: c["event_fields"]["met"].update(reduce="list", accumulate="ratio_to_first", required=False,
                                               default=0.0), "must be required"),
    (lambda c: c["event_fields"]["n_muons"].update(accumulate="ratio_to_first"), "ratio_to_first"),
    (lambda c: c.update(schema_version=1), "require schema_version 2"),
])
def test_schema_two_rules_fail_early(tmp_path, mutation, message):
    config = yaml.safe_load((STUDY / "extraction.yaml").read_text())
    config["collections"]["jets"]["fields"]["flavor"] = {"branch": "Jet/Jet.Flavor", "dtype": "int32"}
    mutation(config)
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        load_profile(tmp_path)


def test_selection_fingerprint_allows_richer_recipes_but_not_other_selections(tmp_path, jets_config):
    study = load_profile(STUDY)
    assert selection_fingerprint(jets_config) == selection_fingerprint(study.config)
    assert recipes_compatible(jets_config, "an older fingerprint", study)
    changed = json.loads(json.dumps(jets_config))
    changed["selection"]["c_jets"]["filters"]["btag"]["eq"] = 17
    assert not recipes_compatible(changed, "an older fingerprint", study)
    hooked = dict(jets_config, hook="extraction.py")
    assert not recipes_compatible(hooked, "an older fingerprint", study)


def test_preparation_rejects_an_export_with_a_different_selection(tmp_path, root_sample, jets_config):
    jets_config["selection"]["b_jets"]["filters"]["pt"]["gt"] = 30.0
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(jets_config))
    main(["compact", "--extraction", str(tmp_path / "extraction.yaml"), "--input", str(root_sample),
          "--sample", "sig_200", "--kind", "signal", "--mass", "200", "--xs-pb", "2", "--outdir", str(tmp_path / "out")])
    signals, _ = discover_samples(tmp_path / "out")
    study = load_study(STUDY)
    with pytest.raises(ValueError, match="different event selection"):
        load_sample_frame(tmp_path / "out", signals["200"], "sig200", list(study.plugin.FEATURES), 3000.0, study)


def test_manifest_rate_provenance_reaches_the_export(tmp_path, root_sample, jets_recipe):
    source = {"method": "pythia_merged_table", "file": "/prod/rates.txt", "sha256": "cd" * 32, "row": "first",
              "merging_scale": 30.0, "mc_uncertainty_pb": 1e-5}
    manifest = tmp_path / "samples.ready.yaml"
    manifest.write_text(yaml.safe_dump({"samples": {"sig_200": {
        "kind": "signal", "mass": 200, "xs_pb": 2.0, "xs_source": source, "files": [str(root_sample)]}}}))
    main(["compact", "--extraction", str(jets_recipe), "--config", str(manifest), "--sample", "sig_200",
          "--outdir", str(tmp_path / "manifest")])
    assert json.loads((tmp_path / "manifest/signal_sig_200.meta.json").read_text())["xs_source"] == source
    main(["compact", "--extraction", str(jets_recipe), "--input", str(root_sample), "--sample", "sig_200",
          "--kind", "signal", "--mass", "200", "--xs-pb", "2", "--outdir", str(tmp_path / "typed")])
    assert json.loads((tmp_path / "typed/signal_sig_200.meta.json").read_text())["xs_source"] is None


@pytest.mark.parametrize("feature", ["jets_truth_flavor", "truth_partons_pt", "truth_lhe_weights"])
def test_truth_columns_are_refused_as_features(feature):
    with pytest.raises(ValueError, match="Generator-truth columns"):
        make_xyw(pd.DataFrame({feature: [1.0], "target": [1]}), [feature])


def test_environment_check_reports_every_mismatch(tmp_path):
    exact = tmp_path / "exact.txt"
    exact.write_text(f"# python: 9.9\nnumpy=={version('numpy')}\npyarrow=={version('pyarrow')}\n")
    assert read_constraints(exact) == ("9.9", {"numpy": version("numpy"), "pyarrow": version("pyarrow")})
    problems = mismatches(exact)
    assert len(problems) == 1 and problems[0].startswith("python ")
    wrong = tmp_path / "wrong.txt"
    wrong.write_text("numpy==0.0.1\nnot-a-real-package==1.0\n")
    assert mismatches(wrong) == [f"numpy {version('numpy')} (pinned 0.0.1)", "not-a-real-package missing (pinned 1.0)"]
    (tmp_path / "bad.txt").write_text("numpy>=1\n")
    with pytest.raises(ValueError, match="exact"):
        read_constraints(tmp_path / "bad.txt")
    shipped = read_constraints(REPO / "compactor/constraints-py311.txt")
    assert shipped[0] == "3.11" and set(shipped[1]) == {"awkward", "numpy", "pandas", "pyarrow", "uproot"}


TRUTH_TAG_RECIPE = STUDY / "extraction_truth_tag.yaml"


def test_truth_tag_recipe_keeps_events_with_three_jets_in_acceptance_whatever_their_tags():
    rows = ak.to_list(delphes_events(6))
    # Event 3 fails the direct selection (its c jet is untagged). Event 5's third jet sits
    # at the strict 25 GeV boundary, leaving two jets in acceptance.
    rows[5]["Jet/Jet.PT"] = [60.0, 55.0, 25.0, 20.0]
    events = ak.Array(rows)
    truth_tag, direct = load_profile(TRUTH_TAG_RECIPE), load_profile(STUDY)
    kept, selected = truth_tag.process_chunk(events, 1).frame, direct.process_chunk(events, 1).frame
    assert kept.event_number.tolist() == [100, 101, 102, 103, 104]
    assert selected.event_number.tolist() == [100, 101, 102, 104]
    # Every directly selected event is kept, with identical jets.
    for row in selected.itertuples():
        match = kept[kept.event_number == row.event_number].iloc[0]
        for field in ("pt", "eta", "phi", "mass", "btag", "truth_flavor"):
            np.testing.assert_array_equal(match[f"jets_{field}"], getattr(row, f"jets_{field}"))
    assert not any(column.startswith(("role_", "towers_", "eflow_", "truth_partons_")) for column in kept)
    assert set(truth_tag.truth_columns) == {"jets_truth_flavor", "truth_lhe_weights"}
    # Same branches, dtypes and event fields as the direct recipe; only the selection differs.
    assert truth_tag.config["collections"]["jets"] == direct.config["collections"]["jets"]
    assert truth_tag.config["event_fields"] == direct.config["event_fields"]
    assert selection_fingerprint(truth_tag.config) != selection_fingerprint(direct.config)
    assert not recipes_compatible(truth_tag.config, "a truth-tag export", direct)


def test_truth_tag_export_is_valid_and_named_apart(tmp_path):
    manifest = export_sample([Path("memory.root")], tmp_path, load_profile(TRUTH_TAG_RECIPE), sample="bkg_bbc",
                             kind="background", xs_pb=1.0, step_size=2, source_reader=InMemorySource(delphes_events()))
    assert manifest["status"] == "complete" and manifest["study_name"] == "cg_bbc_truth_tag"
    assert manifest["cutflow"] == {"input": 5, "jets_in_acceptance": 5, "selected": 5}
    validate_export(tmp_path, manifest)


CONSTITUENT_RECIPE = STUDY / "extraction_constituents.yaml"


def test_constituent_recipe_keeps_truth_tag_events_with_detector_level_constituents():
    rows = ak.to_list(delphes_events(6))
    rows[5]["Jet/Jet.PT"] = [60.0, 55.0, 25.0, 20.0]  # two jets in acceptance: dropped
    rows[0]["EFlowTrack/EFlowTrack.PT"] = [5.0, 4.0, 0.3, 2.0]  # one soft track below 0.5 GeV
    rows[0]["EFlowPhoton/EFlowPhoton.ET"] = [3.0, 0.2]
    for i, row in enumerate(rows):
        for kind in ("Electron/Electron", "Muon/Muon"):
            row[f"{kind}.PT"] = [30.0 + i] if i % 2 else []
            row[f"{kind}.Eta"] = [0.5] if i % 2 else []
            row[f"{kind}.Phi"] = [1.0] if i % 2 else []
            row[f"{kind}.Charge"] = [-1] if i % 2 else []
    events = ak.Array(rows)
    constituents = load_profile(CONSTITUENT_RECIPE)
    frame = constituents.process_chunk(events, 1).frame
    truth_tag = load_profile(TRUTH_TAG_RECIPE).process_chunk(events, 1).frame
    assert frame.event_number.tolist() == truth_tag.event_number.tolist() == [100, 101, 102, 103, 104]
    assert selection_fingerprint(constituents.config) == selection_fingerprint(load_profile(TRUTH_TAG_RECIPE).config)
    first = frame.iloc[0]
    assert first.eflow_tracks_pt.tolist() == [5.0, 4.0, 2.0] and first.eflow_photons_et.tolist() == [3.0]
    for row in frame.itertuples():  # every jet's towers, each linked to its jet
        assert sorted(row.towers_owner.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert frame.electrons_pt.map(len).tolist() == [0, 1, 0, 1, 0] and frame.muons_charge[1].tolist() == [-1]
    # No generator-derived detector inputs: no track PDG-derived kind, masses or impact parameters.
    assert not any(column.startswith("eflow_tracks_") and column.split("_")[-1] in {"kind", "mass", "d0", "dz"}
                   for column in frame)
    assert set(constituents.truth_columns) == {"jets_truth_flavor", *(f"truth_partons_{f}" for f in
                                                                       ("pid", "status", "pt", "eta", "phi", "mass"))}
