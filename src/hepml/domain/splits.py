"""Persistent event assignments shared across hypotheses and model families."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def hash_rank_splits(event_ids, seed):
    """Event IDs ranked by sha256(seed, ID): the first 10% (rounded) test, the next 10% val, the rest train."""
    ids = sorted(event_ids, key=lambda x: (hashlib.sha256(f"{seed}\0{x}".encode()).hexdigest(), x))
    n_test = n_val = round(0.1 * len(ids))
    assignments = np.full(len(ids), "train", dtype=object)
    assignments[:n_test] = "test"
    assignments[n_test:n_test + n_val] = "val"
    return ids, assignments


def extend_sample_membership(previous: pd.DataFrame, event_ids, sample_key: str, *, seed=42) -> pd.DataFrame:
    """One physical sample's assignments after its membership grows, e.g. by a looser extraction.

    Registered events keep their split. Only the new events are ranked by the same
    hash and split 80/10/10 among themselves, so no existing assignment can move.
    Every registered event must still be present.
    """
    columns = ["event_id", "sample_key", "split", "assignment_method"]
    ids = pd.Index(event_ids)
    if ids.has_duplicates:
        raise ValueError(f"Duplicate physical event IDs: {sample_key}")
    if (previous.sample_key != sample_key).any():
        raise ValueError("Previous assignments belong to another sample")
    missing = pd.Index(previous.event_id).difference(ids)
    if len(missing):
        raise ValueError(f"{sample_key}: {len(missing)} registered events are absent from the new membership")
    new_ids, assignments = hash_rank_splits(ids.difference(pd.Index(previous.event_id)), seed)
    added = pd.DataFrame(dict(event_id=new_ids, sample_key=sample_key, split=assignments,
                              assignment_method=f"sha256_rank_80_10_10_v1_seed{seed}_membership_extension"))
    return pd.concat([previous[columns], added], ignore_index=True).sort_values("event_id").reset_index(drop=True)


def extend_registry(registry: pd.DataFrame, events: pd.DataFrame, *, seed=42) -> pd.DataFrame:
    """Preserve anchored samples; assign new samples by stable event-ID hash rank.

    Ranking within a physical sample gives rounded 80/10/10 counts. Adding or
    reordering other samples cannot move its events. Changing membership of an
    already registered sample is an error, requiring a new versioned registry.
    """
    columns = ["event_id", "sample_key", "split", "assignment_method"]
    if registry.empty:
        registry = pd.DataFrame(columns=columns)
    if registry.event_id.duplicated().any() or events.event_id.duplicated().any():
        raise ValueError("Duplicate physical event IDs in registry or sample")
    if not set(registry.split).issubset({"train", "val", "test"}):
        raise ValueError("Invalid registry split")
    additions = []
    for sample, group in events.groupby("sample_key", sort=True):
        previous = registry.loc[registry.sample_key == sample]
        if len(previous):
            if set(previous.event_id) != set(group.event_id):
                raise ValueError(f"Registered sample membership changed: {sample}")
            continue
        if set(group.event_id) & set(registry.event_id):
            raise ValueError("A physical event occurs in different samples")
        ids, assignments = hash_rank_splits(group.event_id, seed)
        if min((assignments == name).sum() for name in ("train", "val", "test")) < 1:
            raise ValueError(f"Too few events for three splits: {sample}")
        additions.append(pd.DataFrame(dict(event_id=ids, sample_key=sample, split=assignments,
                                          assignment_method=f"sha256_rank_80_10_10_v1_seed{seed}")))
    result = pd.concat([registry[columns], *additions], ignore_index=True)
    if result.event_id.duplicated().any():
        raise ValueError("A physical event occurs in different samples")
    return result.sort_values(["sample_key", "event_id"]).reset_index(drop=True)
