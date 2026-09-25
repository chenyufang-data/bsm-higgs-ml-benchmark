"""Expand physical events only after splitting; conditioning weights are not physics weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def condition_frame(frame, mass, conditioning):
    """Condition every event at one explicit, supported hypothesis for inference."""
    if mass is None or not np.isfinite(mass) or float(mass) not in conditioning["support"]:
        raise ValueError("An explicit supported hypothesis mass is required; interpolation is not validated")
    feature = conditioning["feature"]
    if feature in frame and not np.all(frame[feature].to_numpy() == mass):
        raise ValueError("Existing conditioning column conflicts with requested hypothesis")
    return frame.assign(**{feature: float(mass)})


def anchored_class_weights(target, reference_signal_count) -> np.ndarray:
    """Loss weights giving one hypothesis the reference mass's class totals.

    Each signal event weighs N_signal(reference)/N_signal(this hypothesis) and each
    background event 1, so the signal/background loss ratio is the reference's.
    """
    target = np.asarray(target)
    signal = int((target == 1).sum())
    if not signal or reference_signal_count <= 0:
        raise ValueError("Anchored weights need signal events and a positive reference count")
    return np.where(target == 1, reference_signal_count / signal, 1.0)


def expand_hypotheses(frames, *, physics_features, conditioning, reference_mass, split):
    """Preserve the reference's unweighted class totals and raw within-class mixture.

    N_s(ref)*q(h)/N_s(h) for each signal event; q(h) for each replicated background.
    Thus background copies sum to one and both class-conditional hypothesis priors
    equal q(h). Class totals and regularization scale match the reference split.
    """
    masses = list(conditioning["support"])
    if set(frames) != set(masses) or reference_mass not in frames or split not in {"train", "val"}:
        raise ValueError("Conditioning requires complete train/validation hypothesis support")
    feature = conditioning["feature"]
    if feature in physics_features or any(feature in f for f in frames.values()):
        raise ValueError("Conditioning feature conflicts with physical inputs")
    prior = {float(h): float(q) for h, q in conditioning["prior"].items()}
    if (
        set(prior) != set(masses)
        or any(not np.isfinite(q) or q <= 0 for q in prior.values())
        or not np.isclose(sum(prior.values()), 1)
    ):
        raise ValueError("Invalid hypothesis prior")
    columns = ["event_id", "sample_key", "sample", "target", "sample_weight", *physics_features]
    reference = frames[reference_mass]
    background = reference.loc[reference.target == 0, columns].sort_values("event_id").reset_index(drop=True)
    reference_signal_count = int((reference.target == 1).sum())
    if not len(background) or not reference_signal_count:
        raise ValueError("Both classes need reference support")
    all_signal_ids, expanded = set(), []
    for mass in masses:
        frame = frames[mass]
        if frame.event_id.duplicated().any() or set(frame.target) != {0, 1}:
            raise ValueError("Require unique physical events and both classes")
        other = frame.loc[frame.target == 0, columns].sort_values("event_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(background, other, check_exact=True)
        signal = frame.loc[frame.target == 1, columns].copy()
        if set(signal.event_id) & (all_signal_ids | set(background.event_id)):
            raise ValueError("Physical identity reused across signal hypotheses or classes")
        all_signal_ids.update(signal.event_id)
        part = pd.concat([signal, background], ignore_index=True)
        part[feature] = float(mass)
        part["hypothesis_id"] = str(int(mass))
        part["generated_mass"] = np.where(part.target == 1, float(mass), np.nan)
        part["split"] = split
        part["fit_weight"] = anchored_class_weights(part.target, reference_signal_count) * prior[mass]
        expanded.append(part)
    pooled = pd.concat(expanded, ignore_index=True)
    if pooled.duplicated(["event_id", "hypothesis_id"]).any():
        raise ValueError("Duplicate physical-event/hypothesis pair")
    totals = pooled.groupby(["target", feature]).fit_weight.sum()
    for target, count in [(0, len(background)), (1, reference_signal_count)]:
        for mass in masses:
            if not np.isclose(totals.loc[target, mass], count * prior[mass], rtol=1e-12):
                raise ValueError("Class-conditional hypothesis prior does not close")
    audit = (
        pooled.groupby([feature, "target", "sample"])
        .agg(
            rows=("event_id", "size"),
            unique_events=("event_id", "nunique"),
            fit_weight_sum=("fit_weight", "sum"),
            stored_physics_weight_sum=("sample_weight", "sum"),
        )
        .reset_index()
        .assign(split=split)
    )
    return pooled, audit


def assert_split_isolation(train, validation, test_ids):
    train_ids, val_ids, test_ids = set(train.event_id), set(validation.event_id), set(test_ids)
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Physical events cross train/validation/test boundaries")
