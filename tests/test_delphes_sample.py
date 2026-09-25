"""Optional check of the cg_bbc recipe on a real Delphes file (skipped in CI).

Synthetic files cannot carry Delphes object references, so real TRefArray
constituents are only exercised here. Point HEPML_DELPHES_SAMPLE at any Delphes
ROOT file with the standard branches to run it.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from hepml_compact.config import load_profile
from hepml_compact.root_reader import RootEventSource

SAMPLE = os.environ.get("HEPML_DELPHES_SAMPLE")
pytestmark = pytest.mark.skipif(not SAMPLE, reason="set HEPML_DELPHES_SAMPLE to a Delphes ROOT file")


def test_recipe_on_real_delphes_events():
    profile = load_profile(Path(__file__).resolve().parents[1] / "studies/cg_bbc")
    reader = RootEventSource()
    source = reader.inspect(Path(SAMPLE), profile, 3000)
    with reader.open(source, profile.config["tree"]) as tree:
        arrays = tree.read(source["branches"], 0, min(3000, source["entries"]))
    chunk = profile.process_chunk(arrays, 1)
    frame = chunk.frame
    assert len(frame) > 0
    ratios = []
    for row in frame.itertuples():
        assert ((row.towers_owner >= 0) & (row.towers_owner < len(row.jets_pt))).all()
        for jet in np.unique(row.towers_owner):
            owned = row.towers_owner == jet
            px = (row.towers_et[owned] * np.cos(row.towers_phi[owned])).sum()
            py = (row.towers_et[owned] * np.sin(row.towers_phi[owned])).sum()
            ratios.append(np.hypot(px, py) / row.jets_pt[jet])
        assert set(row.eflow_tracks_kind.tolist()) <= {0, 1, 2}
        assert set(row.truth_partons_status.tolist()) <= {21, 22, 23}
    # Jets carry an energy-scale correction; their towers sum to roughly the jet pT.
    assert 0.8 < np.median(ratios) < 1.1
    accumulated = chunk.accumulators["truth_lhe_weights"]
    assert accumulated["events"] == len(arrays)
    assert accumulated["sum"][0] == pytest.approx(len(arrays))
