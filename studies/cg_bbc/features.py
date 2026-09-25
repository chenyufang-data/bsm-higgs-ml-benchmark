"""Model inputs for the example channel. Edit observables here.

Every event has three tagged jets: two bottom-like (b1, b2) and one charm-like (c1). This example
feeds the models their four-momentum components, (pt, eta, phi, mass) per jet, so twelve inputs.
Engineered observables of the same jets, such as invariant masses or angular separations, would
be added in build_features and listed in DEFAULT_FEATURES.

BASELINE_COLUMNS are not model inputs. The cut-based baseline selects a window in the mass of the
charm-like jet paired with either bottom-like jet, together with a minimum scalar sum of the jet pTs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hepml.domain.physics import inv_mass

ROLES = ("b1", "b2", "c1")
COMPONENTS = ("pt", "eta", "phi", "mass")

DEFAULT_FEATURES: list[str] = [f"{component}{role}" for role in ROLES for component in COMPONENTS]
BASELINE_COLUMNS: list[str] = ["mcb1", "mcb2", "ht"]


def build_features(objects: dict, label: int) -> pd.DataFrame:
    """Features from role four-vectors {"b1": (pt, eta, phi, mass), ...}."""
    b1, b2, c1 = (objects[role] for role in ROLES)
    out = {"label": np.full_like(b1[0], label, dtype=np.int32)}
    for role in ROLES:
        for component, values in zip(COMPONENTS, objects[role]):
            out[f"{component}{role}"] = values
    out["mcb1"] = inv_mass(b1, c1).astype(np.float32)
    out["mcb2"] = inv_mass(b2, c1).astype(np.float32)
    out["ht"] = (b1[0] + b2[0] + c1[0]).astype(np.float32)
    return pd.DataFrame(out)
