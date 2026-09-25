"""Generate synthetic inputs in pytest's temporary directory, never from production."""

import copy
import os
from pathlib import Path

# Headless figures for every test: whichever test imports matplotlib first fixes the
# backend for the session, and the interactive default (Tk) is unavailable here and in CI.
os.environ["MPLBACKEND"] = "Agg"

import awkward as ak  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import uproot  # noqa: E402
import yaml  # noqa: E402

from hepml.cli import main  # noqa: E402

STUDY_RECIPE = Path(__file__).resolve().parents[1] / "studies" / "cg_bbc" / "extraction.yaml"


def jets_only_config():
    """The cg_bbc recipe reduced to the branches synthetic ROOT files provide.

    uproot cannot write Delphes object references, so file-based tests export
    with this subset. It keeps the study's selection unchanged, which also makes
    every such test exercise reading an export with the richer study recipe.
    """
    config = yaml.safe_load(STUDY_RECIPE.read_text(encoding="utf-8"))
    jets = config["collections"]["jets"]["fields"]
    config["collections"] = {"jets": {"fields": {k: jets[k] for k in ("pt", "eta", "phi", "mass", "btag")}}}
    config["event_fields"] = {k: config["event_fields"][k] for k in ("weight", "xs")}
    return config


@pytest.fixture
def jets_config():
    return copy.deepcopy(jets_only_config())


@pytest.fixture(scope="session")
def jets_recipe(tmp_path_factory):
    path = tmp_path_factory.mktemp("jets-only-recipe") / "extraction.yaml"
    path.write_text(yaml.safe_dump(jets_only_config(), sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def root_factory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic-root")
    counter = 0

    def create(*, seed=42, signal=True, entries=2000):
        nonlocal counter
        counter += 1
        path = directory / f"synthetic_{counter}.root"
        rng = np.random.default_rng(seed)
        pt = rng.uniform(30, 130, (entries, 3)) + (10 if signal else 0)
        pt[::5, 2] = 20  # Exactly one fifth fail the c-jet pT cut.
        branches = {
            "Jet/Jet.PT": ak.Array(pt.tolist()),
            "Jet/Jet.Eta": ak.Array(rng.uniform(-2, 2, (entries, 3)).tolist()),
            "Jet/Jet.Phi": ak.Array(rng.uniform(-np.pi, np.pi, (entries, 3)).tolist()),
            "Jet/Jet.Mass": ak.Array(rng.uniform(1, 15, (entries, 3)).tolist()),
            "Jet/Jet.BTag": ak.Array(np.tile([1, 1, 16], (entries, 1)).tolist()),
            "Event/Event.Weight": ak.Array(rng.uniform(0.8, 1.2, (entries, 1)).tolist()),
            "Event/Event.CrossSection": ak.Array(np.ones((entries, 1)).tolist()),
        }
        with uproot.recreate(path) as root:
            root.mktree("Delphes", branches)
        return path

    return create


@pytest.fixture(scope="session")
def root_sample(root_factory):
    return root_factory()


@pytest.fixture
def prepared_dataset(tmp_path, root_factory, monkeypatch, jets_recipe):
    """Exercise the public compact -> prepare path with two synthetic samples.

    Export uses the jets-only recipe; preparation uses the full study recipe.
    """
    monkeypatch.setenv("HEPML_OUTPUT_ROOT", str(tmp_path / "outputs"))
    study = Path(__file__).resolve().parents[1] / "studies" / "cg_bbc"
    compact = tmp_path / "outputs" / "cg_bbc" / "compact"
    datasets = tmp_path / "outputs" / "cg_bbc" / "datasets"
    for key, kind, seed, xs in (("sig_200", "signal", 42, 2.0), ("synthetic_bkg", "background", 43, 20.0)):
        source = root_factory(seed=seed, signal=kind == "signal")
        args = [
            "compact",
            "--extraction",
            str(jets_recipe),
            "--input",
            str(source),
            "--sample",
            key,
            "--kind",
            kind,
            "--xs-pb",
            str(xs),
            "--outdir",
            str(compact),
            "--step-size",
            "500",
        ]
        if kind == "signal":
            args += ["--mass", "200"]
        main(args)
    main(
        [
            "prepare",
            "--study",
            str(study),
            "--indir",
            str(compact),
            "--outdir",
            str(datasets),
            "--mass",
            "200",
            "--write-splits",
        ]
    )
    return datasets
