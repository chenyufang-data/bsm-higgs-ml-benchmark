"""Compact objects to model inputs; no Torch dependency or event reselection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hepml.domain.physics import four_vector


def role_kinematics(frame: pd.DataFrame, role: str, collection: str) -> tuple[np.ndarray, ...]:
    """Gather a role in saved event order, returning pt, eta, phi, mass in GeV/rad."""
    indices = frame[f"role_{role}"].to_numpy()
    if indices.dtype.kind not in "iu":
        raise ValueError(f"Role {role} indices must be integers")
    result = []
    for field in ("pt", "eta", "phi", "mass"):
        rows = frame[f"{collection}_{field}"]
        if any(index < 0 or index >= len(row) for row, index in zip(rows, indices)):
            raise ValueError(f"Role {role} refers to a missing object")
        values = np.array([row[index] for row, index in zip(rows, indices)], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Role {role} contains nonfinite kinematics")
        result.append(values)
    return tuple(result)


@dataclass
class ObjectBatch:
    event_id: np.ndarray
    kinematics: np.ndarray  # (events, objects, 4): pt, eta, phi, mass
    four_vectors: np.ndarray  # (events, objects, 4): px, py, pz, E
    mask: np.ndarray  # (events, objects), True for real objects
    attributes: dict[str, np.ndarray]
    target: np.ndarray | None
    physical_weight: np.ndarray | None


def object_batch(
    frame: pd.DataFrame,
    *,
    collection: str = "jets",
    roles: list[str] | None = None,
    attributes: tuple[str, ...] = ("btag",),
) -> ObjectBatch:
    """Build a minibatch from saved splits, with zero padding and no truncation.

    Use roles=['b1', 'b2', 'c1'] for the common three-jet benchmark, or None for
    all objects. Four-vector order is (px, py, pz, E), in GeV. No standardization
    is applied. Model adapters choose their tensor layout and compute dtype.
    """
    fields = ("pt", "eta", "phi", "mass", *attributes)
    n = len(frame)
    counts = np.array([len(row) for row in frame[f"{collection}_pt"]], dtype=np.int64)
    for field in fields:
        if [len(row) for row in frame[f"{collection}_{field}"]] != counts.tolist():
            raise ValueError(f"{collection}: fields are not aligned")
    if roles is not None and (not roles or len(set(roles)) != len(roles)):
        raise ValueError("Provide unique roles or None for all objects")
    width = len(roles) if roles is not None else int(counts.max(initial=0))
    # Float64 avoids unnecessary cancellation before a model chooses its dtype.
    kin = np.zeros((n, width, 4), dtype=np.float64)
    attrs = {name: np.zeros((n, width), dtype=np.float64) for name in attributes}
    mask = np.zeros((n, width), dtype=bool)
    for row_index, (_, row) in enumerate(frame.iterrows()):
        if roles is None:
            indices = np.arange(counts[row_index])
        else:
            indices = np.array([row[f"role_{role}"] for role in roles])
            if indices.dtype.kind not in "iu":
                raise ValueError("Role indices must be integers")
            if np.any(indices < 0) or np.any(indices >= counts[row_index]):
                raise ValueError("Role refers to a missing object")
        size = len(indices)
        for component, field in enumerate(fields[:4]):
            kin[row_index, :size, component] = np.asarray(row[f"{collection}_{field}"])[indices]
        for field in attributes:
            attrs[field][row_index, :size] = np.asarray(row[f"{collection}_{field}"])[indices]
        mask[row_index, :size] = True
    p4 = np.stack(four_vector(*(kin[:, :, i] for i in range(4))), axis=-1)
    if not all(np.isfinite(values).all() for values in (kin, p4, *attrs.values())):
        raise ValueError("Nonfinite object inputs; refusing to change event membership")
    return ObjectBatch(
        event_id=frame["event_id"].to_numpy(copy=True),
        kinematics=kin,
        four_vectors=p4,
        mask=mask,
        attributes=attrs,
        target=frame["target"].to_numpy(copy=True) if "target" in frame else None,
        physical_weight=frame["sample_weight"].to_numpy(copy=True) if "sample_weight" in frame else None,
    )
