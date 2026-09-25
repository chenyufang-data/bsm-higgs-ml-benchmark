"""Production cross-section parsing must never silently pick the wrong row."""

import hashlib

import pytest
import yaml
from hepml_compact.cross_sections import chosen_cross_section, fill_manifest


def test_adjacent_filename_is_explicit_and_sample_path_takes_precedence(tmp_path):
    (tmp_path / "events.root").touch()
    header = "Merging scale Cross-section [pb] MC uncertainty [pb]\n"
    (tmp_path / "custom-rate.dat").write_text(header + "125 2.0 0.01\n")
    (tmp_path / "override.dat").write_text(header + "125 3.0 0.01\n")
    sample = {"files": ["events.root"], "xs_pb": None}
    manifest = tmp_path / "samples.yaml"
    manifest.write_text(yaml.safe_dump({"samples": {"signal": sample}}))
    output = tmp_path / "ready.yaml"
    with pytest.raises(ValueError, match="No rate filename is assumed"):
        fill_manifest(manifest, output, row="first")
    assert not output.exists()
    fill_manifest(manifest, output, row="first", xs_filename="custom-rate.dat")
    assert yaml.safe_load(output.read_text())["samples"]["signal"]["xs_pb"] == 2.0
    sample["xs_file"] = "override.dat"
    manifest.write_text(yaml.safe_dump({"samples": {"signal": sample}}))
    override = tmp_path / "override.yaml"
    fill_manifest(manifest, override, row="first", xs_filename="custom-rate.dat")
    assert yaml.safe_load(override.read_text())["samples"]["signal"]["xs_pb"] == 3.0


def test_adjacent_filename_cannot_infer_multifile_rate(tmp_path):
    for name in ("one.root", "two.root"):
        (tmp_path / name).touch()
    (tmp_path / "rates.dat").write_text("Merging scale Cross-section [pb] MC uncertainty [pb]\n125 2.0 0.01\n")
    manifest = tmp_path / "samples.yaml"
    manifest.write_text(yaml.safe_dump({"samples": {"background": {"files": ["one.root", "two.root"]}}}))
    output = tmp_path / "ready.yaml"
    with pytest.raises(ValueError, match="multiple ROOT files"):
        fill_manifest(manifest, output, row="first", xs_filename="rates.dat")
    assert not output.exists()


@pytest.mark.parametrize("filename", ["", " ", ".", "..", "../rate.dat", "folder/rate.dat", "C:\\rate.dat"])
def test_adjacent_filename_rejects_paths(tmp_path, filename):
    with pytest.raises(ValueError, match="must be a filename"):
        fill_manifest(tmp_path / "input.yaml", tmp_path / "output.yaml", xs_filename=filename)


def test_chosen_row_is_not_assumed_to_be_first(tmp_path):
    path = tmp_path / "rates.dat"
    path.write_text(
        "Merging scale Cross-section [pb] MC uncertainty [pb]\n"
        "187.4 8.157523e-04 1.31e-07\n"
        "125 7.603675e-04 1.26e-07 (choice)\n"
    )
    assert chosen_cross_section(path) == 7.603675e-04
    assert chosen_cross_section(path, row="first") == 8.157523e-04


def test_first_row_without_marker_and_fortran_exponents(tmp_path):
    path = tmp_path / "rates.dat"
    path.write_text(
        "Merging scale Cross-section [pb] MC uncertainty [pb]\n\n"
        "125 7.603675D-04 1.26D-07\n187.4 8.157523e-04 1.31e-07\n"
    )
    assert chosen_cross_section(path, row="first") == 7.603675e-04
    with pytest.raises(ValueError, match="use --row first"):
        chosen_cross_section(path)


@pytest.mark.parametrize("data", ["", "125 1e-3\n", "125 nan 1e-7\n", "125 -1e-3 1e-7\n"])
def test_invalid_first_row_cannot_be_skipped(tmp_path, data):
    path = tmp_path / "rates.dat"
    path.write_text("Merging scale Cross-section [pb] MC uncertainty [pb]\n" + data)
    with pytest.raises(ValueError):
        chosen_cross_section(path, row="first")


def test_mixed_signal_and_explicit_multifile_background(tmp_path):
    signal = tmp_path / "signal"
    signal.mkdir()
    (signal / "events.root").touch()
    (signal / "rates.dat").write_text("Merging scale Cross-section [pb] MC uncertainty [pb]\n125 0.002 1e-7\n")
    for name in ("bkg1.root", "bkg2.root"):
        (tmp_path / name).touch()
    config = {
        "samples": {
            "signal": {"files": ["signal/events.root"], "xs_pb": None, "xs_file": "signal/rates.dat"},
            "background": {"files": ["bkg1.root", "bkg2.root"], "xs_pb": 5.0},
        }
    }
    manifest = tmp_path / "input.yaml"
    manifest.write_text(yaml.safe_dump(config))
    output = tmp_path / "ready.yaml"
    with pytest.raises(ValueError, match="multiple ROOT files"):
        fill_manifest(manifest, output, row="first")
    assert not output.exists()
    assert fill_manifest(manifest, output, row="first", keep_existing=True) == 2
    samples = yaml.safe_load(output.read_text())["samples"]
    assert samples["signal"]["xs_pb"] == 0.002
    assert samples["background"]["xs_pb"] == 5.0
    assert samples["background"]["xs_source"] == {"method": "supplied_in_manifest"}
    assert samples["signal"]["xs_source"]["row"] == "first"
    config["samples"]["background"]["xs_pb"] = None
    manifest.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Supply xs_pb explicitly"):
        fill_manifest(manifest, tmp_path / "bad.yaml", row="first", keep_existing=True)


def test_multifile_sample_uses_explicit_rate_once(tmp_path):
    directory = tmp_path / "bkgbbc"
    directory.mkdir()
    files = [f"bkgbbc/tag_{tag}_delphes_events.root" for tag in (1, 2)]
    for name in files:
        (tmp_path / name).touch()
    (directory / "rates.dat").write_text("Merging scale Cross-section [pb] MC uncertainty [pb]\n125 2.5 0.01\n")
    manifest = tmp_path / "samples.yaml"
    sample = {"kind": "background", "xs_pb": None, "xs_file": "bkgbbc/rates.dat", "files": files}
    manifest.write_text(yaml.safe_dump({"samples": {"bkg_bbc": sample}}))
    output = tmp_path / "ready.yaml"
    assert fill_manifest(manifest, output, row="first") == 1
    result = yaml.safe_load(output.read_text())["samples"]["bkg_bbc"]
    source = result.pop("xs_source")
    assert result == {**sample, "xs_pb": 2.5}  # never multiply the rate by the file count
    assert source["method"] == "pythia_merged_table" and source["merging_scale"] == 125
    assert source["sha256"] == hashlib.sha256((directory / "rates.dat").read_bytes()).hexdigest()
    (directory / "rates.dat").unlink()
    with pytest.raises(FileNotFoundError):
        fill_manifest(manifest, tmp_path / "missing.yaml", row="first")
    assert not (tmp_path / "missing.yaml").exists()


@pytest.mark.parametrize(
    "rows",
    [
        "125 1e-3 1e-6\n",
        "125 1e-3 1e-6 (choice)\n150 2e-3 1e-6 (choice)\n",
        "125 nan 1e-6 (choice)\n",
        "125 -1e-3 1e-6 (choice)\n",
    ],
)
def test_missing_ambiguous_or_invalid_choice_fails(tmp_path, rows):
    path = tmp_path / "rates.dat"
    path.write_text("Merging scale Cross-section [pb] MC uncertainty [pb]\n" + rows)
    with pytest.raises(ValueError):
        chosen_cross_section(path)


def test_fill_preserves_hypothesis_and_automatic_counts(tmp_path):
    root = tmp_path / "production"
    directory = root / "m200/r01"
    directory.mkdir(parents=True)
    (directory / "events.root").touch()  # helper only checks existence, no ROOT reads
    xs_file = directory / "rates.dat"
    xs_file.write_text("Merging scale Cross-section [pb] MC uncertainty [pb]\n125 7.603675e-04 1.26e-07 (choice)\n")
    config = {
        "data_root": "production",
        "samples": {
            "signal": {"kind": "signal", "mass": 200, "rho_tc": 0.1, "xs_pb": None, "files": ["m200/r01/events.root"],
                       "xs_file": "m200/r01/rates.dat"}
        },
    }
    manifest = tmp_path / "samples.yaml"
    manifest.write_text(yaml.safe_dump(config))
    output = tmp_path / "ready.yaml"
    assert fill_manifest(manifest, output) == 1
    result = yaml.safe_load(output.read_text())
    source = result["samples"]["signal"].pop("xs_source")
    assert result["samples"]["signal"] == {**config["samples"]["signal"], "xs_pb": 7.603675e-04}
    assert source["row"] == "choice" and source["mc_uncertainty_pb"] == 1.26e-07
    assert source["file"].endswith("rates.dat")
    assert yaml.safe_load(manifest.read_text()) == config
    with pytest.raises(FileExistsError):
        fill_manifest(manifest, output)
    xs_file.write_text("broken input")
    with pytest.raises(ValueError):
        fill_manifest(manifest, tmp_path / "bad.yaml")
    assert not (tmp_path / "bad.yaml").exists()
