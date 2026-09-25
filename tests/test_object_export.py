"""Physics equivalence, ragged inputs, configuration and standalone deployment."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import uproot
import yaml
from hepml_compact import __version__
from hepml_compact.config import load_profile
from hepml_compact.export import export_sample

from hepml.adapters.object_inputs import object_batch
from hepml.adapters.study_loader import load_study
from hepml.domain.dataset import _clean_df
from scripts.package_server import package_server

REPO = Path(__file__).parents[1]
RECIPE = REPO / "studies/cg_bbc/extraction.yaml"


def recipe(tmp_path, update, base=None):
    config = base if base is not None else yaml.safe_load(RECIPE.read_text())
    update(config)
    path = tmp_path / "extraction.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_profile(path)


def test_selection_boundaries_roles_and_analytic_observables(jets_recipe):
    # First event has two b jets tied in pT, a back-to-back c, and an extra
    # untagged jet. The additional jet must be kept, but excluded from HT.
    # The study's other observables are checked in test_cg_bbc_features.py.
    rows = [
        ([50.0, 50.0, 40.0, 15.0], [0.0, 0.0, 0.0, 3.0], [1, 1, 16, 0]),
        ([50.0, 50.0, 25.0], [0.0, 0.0, 0.0], [1, 1, 16]),  # strict pT cut
        ([50.0, 50.0, 40.0], [0.0, 0.0, 2.5], [1, 1, 16]),  # strict eta cut
        ([50.0, 50.0, 40.0], [0.0, 0.0, 0.0], [1, 1, 17]),  # equality, not bitmask
        ([50.0, 50.0, 40.0, 30.0], [0.0, 0.0, 0.0, 0.0], [1, 1, 16, 1]),  # third b
        ([30.0, 80.0, 40.0], [0.0, 0.0, 0.0], [1, 1, 16]),  # reordered b roles
    ]
    pt = ak.Array([r[0] for r in rows])
    arrays = ak.Array(
        {
            "Jet/Jet.PT": pt,
            "Jet/Jet.Eta": ak.Array([r[1] for r in rows]),
            "Jet/Jet.Phi": ak.Array([[0.0, 0.0, np.pi] + [0.0] * (len(r[0]) - 3) for r in rows]),
            "Jet/Jet.Mass": ak.zeros_like(pt),
            "Jet/Jet.BTag": ak.Array([r[2] for r in rows]),
        }
    )
    selected = load_profile(jets_recipe).process_chunk(arrays, 1)
    np.testing.assert_array_equal(selected.offsets, [0, 5])
    assert selected.frame["role_b1"].tolist() == [0, 1]
    assert selected.frame["role_b2"].tolist() == [1, 0]
    assert [list(x) for x in selected.frame["jets_source_index"]] == [[0, 1, 2, 3], [0, 1, 2]]
    assert selected.frame["weight"].tolist() == [1.0, 1.0]
    frame = load_study().plugin.derive_features(selected.frame)
    np.testing.assert_array_equal(frame["ht"], [140.0, 150.0])
    np.testing.assert_allclose(frame.loc[0, ["mcb1", "mcb2"]].to_numpy(dtype=float), np.sqrt(8000), rtol=1e-6)


def test_ragged_batch_padding_four_vectors_and_lorentz_mass():
    frame = pd.DataFrame(
        {
            "event_id": ["a", "b"],
            "target": [1, 0],
            "sample_weight": [0.1, 50.0],
            "jets_pt": [[50.0, 30.0], [20.0]],
            "jets_eta": [[0.2, -1.0], [0.0]],
            "jets_phi": [[0.0, 2.0], [1.0]],
            "jets_mass": [[5.0, 3.0], [2.0]],
            "jets_btag": [[1, 16], [0]],
        }
    )
    batch = object_batch(frame)
    assert batch.four_vectors.shape == (2, 2, 4)
    np.testing.assert_array_equal(batch.mask, [[True, True], [True, False]])
    np.testing.assert_array_equal(batch.four_vectors[1, 1], np.zeros(4))
    p4 = batch.four_vectors
    m2 = p4[..., 3] ** 2 - np.sum(p4[..., :3] ** 2, axis=-1)
    np.testing.assert_allclose(m2, batch.kinematics[..., 3] ** 2, atol=1e-10)
    beta = 0.4
    gamma = 1 / np.sqrt(1 - beta**2)
    boosted = p4.copy()
    boosted[..., 2] = gamma * (p4[..., 2] - beta * p4[..., 3])
    boosted[..., 3] = gamma * (p4[..., 3] - beta * p4[..., 2])
    np.testing.assert_allclose(boosted[..., 3] ** 2 - np.sum(boosted[..., :3] ** 2, axis=-1), m2, atol=1e-10)
    np.testing.assert_array_equal(batch.event_id, frame["event_id"])
    np.testing.assert_array_equal(batch.physical_weight, frame["sample_weight"])


def test_neural_inputs_use_identical_prepared_events_and_weights(prepared_dataset):
    frame = pd.read_parquet(prepared_dataset / "splits/test_sig200.parquet")
    batch = object_batch(frame, roles=["b1", "b2", "c1"])
    assert batch.mask.all() and batch.kinematics.shape == (320, 3, 4)
    np.testing.assert_array_equal(batch.event_id, frame["event_id"])
    np.testing.assert_array_equal(batch.physical_weight, frame["sample_weight"])
    np.testing.assert_array_equal(batch.target, frame["target"])
    for index, role in enumerate(("b1", "b2", "c1")):
        np.testing.assert_array_equal(batch.kinematics[:, index, 0], frame[f"pt{role}"])
    meta = json.loads((prepared_dataset / "splits/split_sig200.meta.json").read_text())
    assert "jets" in meta["object_schema"]["collections"]


def test_portable_hook_and_explicit_optional_fields(tmp_path, root_sample, jets_config):
    (tmp_path / "extraction.py").write_text(
        "import numpy as np\ndef select(collections, keep, roles, config):\n"
        "    return keep & (np.arange(len(keep)) % 2 == 0), roles\n"
    )

    def update(config):
        config["hook"] = "extraction.py"
        config["collections"]["jets"]["fields"]["charge"] = {
            "branch": "Jet/Jet.Charge",
            "dtype": "int32",
            "required": False,
            "default": -99,
        }

    profile = recipe(tmp_path, update, jets_config)
    result = export_sample([root_sample], tmp_path / "out", profile, sample="test", kind="background")
    assert result["selected_events"] == 800
    assert "Jet/Jet.Charge" in result["sources"][0]["missing_optional"]
    frame = pd.read_parquet(tmp_path / "out" / result["chunks"][0]["file"])
    assert all(np.all(values == -99) for values in frame["jets_charge"])
    fingerprint = profile.fingerprint
    with (tmp_path / "extraction.py").open("a") as stream:
        stream.write("\n# changed hook\n")
    assert load_profile(tmp_path).fingerprint != fingerprint


def test_empty_additional_collection_has_typed_lists(tmp_path, jets_config):
    source = tmp_path / "input.root"
    with uproot.recreate(source) as root:
        root.mktree("Delphes", {"Jet/Jet.PT": "var * float32", "Muon/Muon.PT": "var * float32"})
        root["Delphes"].extend({"Jet/Jet.PT": ak.Array([[30.0], [50.0]]), "Muon/Muon.PT": ak.Array([[], []])})

    def update(config):
        config["collections"] = {
            "jets": {"fields": {"pt": {"branch": "Jet/Jet.PT", "dtype": "float32"}}},
            "muons": {"fields": {"pt": {"branch": "Muon/Muon.PT", "dtype": "float32"}}},
        }
        config["selection"] = {}

    result = export_sample(
        [source], tmp_path / "out", recipe(tmp_path, update, jets_config), sample="test", kind="background"
    )
    table = pq.read_table(tmp_path / "out" / result["chunks"][0]["file"])
    assert table.schema.field("muons_pt").type == pa.list_(pa.float32())
    assert table["muons_pt"].to_pylist() == [[], []]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda c: c.update(unknown=True), "expected a mapping"),
        (lambda c: c["units"].update(momentum="MeV"), "Supported units"),
        (lambda c: c["selection"]["b_jets"]["filters"]["pt"].update(gtt=30), "expected a mapping"),
        (lambda c: c["selection"]["b_jets"].update(count={"min": 1}), "enough objects"),
        (lambda c: c["event_fields"]["weight"].pop("default"), "explicit default"),
    ],
)
def test_invalid_recipes_fail_early(tmp_path, mutation, message):
    with pytest.raises(ValueError, match=message):
        recipe(tmp_path, mutation)


def test_invalid_collection_alignment_and_values_are_rejected(root_sample, jets_recipe):
    profile = load_profile(jets_recipe)
    with uproot.open(root_sample) as root:
        arrays = root["Delphes"].arrays(profile.required_branches, entry_stop=5)
    fields = {key: arrays[key] for key in arrays.fields}
    fields["Jet/Jet.Eta"] = arrays["Jet/Jet.Eta"][:, :2]
    with pytest.raises(ValueError, match="do not align"):
        profile.process_chunk(ak.Array(fields), 1)
    fields["Jet/Jet.Eta"] = arrays["Jet/Jet.Eta"] * np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        profile.process_chunk(ak.Array(fields), 1)


def test_small_signed_delphes_masses_are_preserved(root_sample, jets_recipe):
    profile = load_profile(jets_recipe)
    with uproot.open(root_sample) as root:
        arrays = root["Delphes"].arrays(profile.required_branches, entry_start=1, entry_stop=3)
    fields = {key: arrays[key] for key in arrays.fields}
    fields["Jet/Jet.Mass"] = ak.full_like(arrays["Jet/Jet.Mass"], -6.743496e-7)
    result = profile.process_chunk(ak.Array(fields), 1)
    assert len(result.frame) == 2
    assert all(np.all(np.asarray(values) == np.float32(-6.743496e-7)) for values in result.frame["jets_mass"])
    derived = load_study().plugin.derive_features(result.frame)
    assert np.isfinite(derived[load_study().plugin.FEATURES].to_numpy()).all()


def test_bad_features_cannot_silently_drop_events():
    frame = pd.DataFrame({"label": [1, 0], "weight": [1.0, 1.0], "xs": [1.0, 1.0], "x": [1.0, np.nan]})
    with pytest.raises(ValueError, match="event membership"):
        _clean_df(frame, ["x"])


def test_server_bundle_runs_without_training_imports(tmp_path, root_sample, jets_config):
    # Synthetic ROOT files lack Delphes references, so bundle a jets-only study.
    study = tmp_path / "study"
    study.mkdir()
    (study / "extraction.yaml").write_text(yaml.safe_dump(jets_config, sort_keys=False))
    shutil.copy(REPO / "studies/cg_bbc/samples.yaml", study / "samples.yaml")
    # Recipe variants ship beside extraction.yaml.
    shutil.copy(REPO / "studies/cg_bbc/extraction_truth_tag.yaml", study / "extraction_truth_tag.yaml")
    archive = package_server(REPO, tmp_path / "packages", study)
    top = f"hepml-compactor-{__version__}"
    sidecar = archive.with_suffix(".zip.sha256").read_bytes()
    assert b"\r" not in sidecar
    assert sidecar.decode().split()[0] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        names = bundle.namelist()
        assert all(name.startswith(top + "/") for name in names)
        for required in ("hepml_compact/cross_sections.py", "hepml_compact/environment.py",
                         "constraints-py311.txt", "run_extraction.sh", "study/extraction.yaml",
                         "study/extraction_truth_tag.yaml", "study/samples.yaml"):
            assert f"{top}/{required}" in names
        assert (bundle.getinfo(f"{top}/run_extraction.sh").external_attr >> 16) & 0o777 == 0o755
        assert not any(name.endswith("LOCAL_SETTINGS.md") for name in names)
        assert all("/src/" not in name and "/studies/" not in name for name in names)
        assert not any(name.endswith(("features.py", "plugin.py", ".root")) for name in names)
        manifest = json.loads(bundle.read(f"{top}/MANIFEST.json"))
        assert manifest["version"] == __version__
        assert {"git_commit", "git_dirty"} <= set(manifest)
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(bundle.read(f"{top}/{name}")).hexdigest() == digest
        bundle.extractall(tmp_path / "unpacked")
    standalone = tmp_path / "unpacked" / top
    # -I excludes cwd/PYTHONPATH; the import blocker makes an installed training
    # package or framework unusable even in the developer's full environment.
    code = """
import importlib.abc
import pathlib
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'hepml', 'studies', 'torch', 'xgboost', 'sklearn', 'shap', 'scipy', 'matplotlib'}:
            raise ImportError('Training import forbidden: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
sys.path.insert(0, sys.argv[1])
import hepml_compact
assert pathlib.Path(hepml_compact.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
from hepml_compact.cli import main
main(['--extraction', str(pathlib.Path(sys.argv[1]) / 'study/extraction.yaml'),
      '--input', sys.argv[2], '--outdir', sys.argv[3], '--sample', 'signal', '--kind', 'signal',
      '--mass', '200', '--step-size', '500', '--xs-pb', '2.0'])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(standalone), str(root_sample), str(tmp_path / "out")],
        cwd=standalone,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    meta = json.loads((tmp_path / "out/signal_signal.meta.json").read_text())
    assert meta["n_selected"] == 1600
    assert meta["schema"] == 4
    export = json.loads((tmp_path / "out/signal_signal.export.json").read_text())
    assert export["compactor"]["bundle"]["version"] == __version__
