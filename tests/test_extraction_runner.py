"""Exercise the shipped Bash runner with actual synthetic ROOT files."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[1]


@pytest.fixture
def runner(tmp_path, jets_config):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash is required to test the server runner")
    directory = tmp_path / "standalone with spaces"
    directory.mkdir()
    shutil.copy(REPO / "compactor/run_extraction.sh", directory)
    shutil.copytree(REPO / "compactor/hepml_compact", directory / "hepml_compact")
    (directory / "study").mkdir()
    # Synthetic ROOT files lack Delphes references; the runner logic is recipe-agnostic.
    (directory / "study/extraction.yaml").write_text(yaml.safe_dump(jets_config, sort_keys=False))
    output = tmp_path / "exports with spaces"
    env = dict(os.environ, PYTHON_BIN=sys.executable)
    env.pop("HEPML_DATA_ROOT", None)

    manifests = tmp_path / "manifests"  # the default: <script directory>/../manifests
    manifests.mkdir()

    def run(config, mode="all", extra=(), outdir=True):
        (manifests / "samples.ready.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        return subprocess.run(
            [
                str(bash),
                str(directory / "run_extraction.sh"),
                mode,
                *(["--outdir", str(output)] if outdir else []),
                "--step-size",
                "50",
                *extra,
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    run.directory = directory
    return run, output


def configuration(signal, backgrounds):
    return {
        "samples": {
            "sig_m200_rho01": {
                "kind": "signal",
                "mass": 200,
                "rho_tc": 0.1,
                "xs_pb": 0.3483631,
                "files": [str(signal)],
            },
            "bkg_bbc": {"kind": "background", "xs_pb": 4626.276, "files": [str(path) for path in backgrounds]},
        }
    }


@pytest.mark.parametrize("mode,count", [("all", 2), ("signals", 1), ("backgrounds", 1)])
def test_dry_run_filters_without_touching_data(runner, mode, count):
    run, output = runner
    result = run(configuration("missing signal.root", ["missing background.root"]), mode, ["--dry-run"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("--sample") == count
    assert "\\r" not in result.stdout
    if mode == "signals":
        assert "--sample bkg_bbc" not in result.stdout
    if mode == "backgrounds":
        assert "--sample sig_m200" not in result.stdout
    assert not output.exists()


def test_real_batch_background_metadata_and_resume(runner, root_factory):
    run, output = runner
    config = configuration(root_factory(entries=100), [root_factory(entries=100), root_factory(entries=80)])
    result = run(config)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "backgrounds/background_bkg_bbc.export.json").read_text())
    assert manifest["sample_metadata"] == {"xs_pb": 4626.276}
    assert manifest["processed_entries"] == 180
    meta = json.loads((output / "backgrounds/background_bkg_bbc.meta.json").read_text())
    assert meta["mass"] is None and meta["rho_tc"] is None and meta["label"] == 0
    assert meta["n_events_total"] == 180
    signal = json.loads((output / "rho01/signal_sig_m200_rho01.export.json").read_text())
    assert signal["sample_metadata"]["mass"] == 200 and signal["sample_metadata"]["rho_tc"] == 0.1
    previous = {p: p.stat().st_mtime_ns for p in output.rglob("*.parquet")}
    result = run(config)
    assert result.returncode == 0, result.stdout + result.stderr
    assert previous == {p: p.stat().st_mtime_ns for p in output.rglob("*.parquet")}
    statuses = list(output.glob("logs/*/status.tsv"))
    assert len(statuses) == 2
    assert all(path.read_text().count("COMPLETE") == 2 for path in statuses)


def test_failed_sample_stops_batch_and_records_failure(runner, root_factory, tmp_path):
    run, output = runner
    broken = tmp_path / "broken.root"
    broken.write_bytes(b"not a ROOT file")
    config = configuration(root_factory(entries=20), [broken])
    config["samples"]["never_started"] = {**config["samples"]["sig_m200_rho01"], "rho_tc": 0.2}
    result = run(config)
    assert result.returncode != 0
    status = next(output.glob("logs/*/status.tsv")).read_text()
    assert "FAILED" in status and "never_started" not in status
    assert not (output / "rho02").exists()


def test_missing_input_fails_before_any_extraction(runner):
    run, output = runner
    result = run(configuration("missing.root", ["also_missing.root"]))
    assert result.returncode != 0 and "missing ROOT file" in result.stderr
    assert not output.exists()


def test_default_output_directory_carries_the_compactor_version(runner):
    from hepml_compact import __version__

    run, _ = runner
    result = run(configuration("missing signal.root", ["missing background.root"]), "all", ["--dry-run"], outdir=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"exports/cg_bbc-v{__version__}/rho01" in result.stdout
    assert f"hepml-compactor {__version__}" in result.stderr


def test_environment_mismatch_stops_a_batch_unless_allowed(runner, root_factory):
    run, output = runner
    (run.directory / "constraints-py311.txt").write_text("# python: 2.7\nnumpy==0.0.1\n")
    config = configuration(root_factory(entries=20), [root_factory(entries=20)])
    result = run(config)
    assert result.returncode == 2 and "Refusing to start" in result.stderr
    assert "numpy" in result.stdout and not output.exists()
    result = run(config, extra=["--allow-env-mismatch"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "despite the environment mismatch" in result.stderr
