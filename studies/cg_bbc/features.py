"""Feature definitions for cg_bbc. Edit observables here."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hepml.domain.physics import delta_r, inv_mass

DEFAULT_FEATURES: list[str] = [
    "ptb1",
    "etab1",
    "ptb2",
    "etab2",
    "ptc1",
    "etac1",
    "mcb1",
    "mcb2",
    "dr13",
    "dr23",
    "ht",
    "mcbb",
    "mbb",
    "ratio_ptcb",
    "dr12",
]

JET_COLUMNS = ["phib1", "massb1", "phib2", "massb2", "phic1", "massc1"]


def build_features(objects: dict, label: int) -> pd.DataFrame:
    b1, b2, c1 = (objects[key] for key in ("b1", "b2", "c1"))
    b1_pt, b1_eta, b1_phi, b1_m = b1
    b2_pt, b2_eta, b2_phi, b2_m = b2
    c1_pt, c1_eta, c1_phi, c1_m = c1
    mcb1 = inv_mass(b1, c1).astype(np.float32)
    mcb2 = inv_mass(b2, c1).astype(np.float32)

    dr13 = delta_r(b1_eta, b1_phi, c1_eta, c1_phi).astype(np.float32)
    dr23 = delta_r(b2_eta, b2_phi, c1_eta, c1_phi).astype(np.float32)
    dr12 = delta_r(b1_eta, b1_phi, b2_eta, b2_phi).astype(np.float32)

    ratio_ptcb = (c1_pt / b1_pt).astype(np.float32)
    mbb = inv_mass(b1, b2).astype(np.float32)
    mcbb = inv_mass(c1, b1, b2).astype(np.float32)
    ht = (b1_pt + b2_pt + c1_pt).astype(np.float32)

    out = {
        "label": np.full_like(b1_pt, label, dtype=np.int32),
        "ptb1": b1_pt,
        "etab1": b1_eta,
        "ptb2": b2_pt,
        "etab2": b2_eta,
        "ptc1": c1_pt,
        "etac1": c1_eta,
        "mcb1": mcb1,
        "mcb2": mcb2,
        "dr13": dr13,
        "dr23": dr23,
        "ratio_ptcb": ratio_ptcb,
        "mbb": mbb,
        "mcbb": mcbb,
        "ht": ht,
        "dr12": dr12,
    }

    out.update(phib1=b1_phi, massb1=b1_m, phib2=b2_phi, massb2=b2_m, phic1=c1_phi, massc1=c1_m)
    return pd.DataFrame(out)
