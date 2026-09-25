"""The example features: four-momentum components as model inputs, plus the baseline columns."""

import numpy as np

from hepml.adapters.study_loader import load_study
from studies.cg_bbc.features import DEFAULT_FEATURES


def _objects():
    def one(value):
        return np.array([value], dtype=np.float32)

    # b1 and b2 coincide (pT 50); the massless c jet (pT 40) is back to back with both.
    return {"b1": (one(50.0), one(0.0), one(0.0), one(0.0)),
            "b2": (one(50.0), one(0.0), one(0.0), one(0.0)),
            "c1": (one(40.0), one(0.0), one(np.pi), one(0.0))}


def test_model_inputs_are_the_four_momentum_components_of_each_jet():
    assert DEFAULT_FEATURES == [f"{component}{role}" for role in ("b1", "b2", "c1")
                                for component in ("pt", "eta", "phi", "mass")]
    frame = load_study().plugin.features_from_objects(_objects())
    assert list(frame.columns) == DEFAULT_FEATURES
    np.testing.assert_allclose(frame.loc[0, ["ptb1", "ptb2", "ptc1", "phic1"]].to_numpy(dtype=float),
                               [50.0, 50.0, 40.0, np.pi], rtol=1e-6)


def test_baseline_columns_for_the_cut_based_selection():
    frame = load_study().plugin.features_from_objects(_objects(), retained=True)
    np.testing.assert_allclose(frame.loc[0, ["mcb1", "mcb2"]].to_numpy(dtype=float), np.sqrt(8000), rtol=1e-6)
    assert frame.loc[0, "ht"] == 140.0
