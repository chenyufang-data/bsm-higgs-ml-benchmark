"""Paired comparison of models on identical physical events (Phase 5).

Every model is scored on the same validation events. Each bootstrap replica draws
one set of event multiplicities and applies it to every model, and to every
hypothesis that shares an event, so differences between models keep the
correlation of their shared events. Filesystem-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hepml.domain.metrics import safe_auc

IDENTITY = ["event_id", "sample", "target", "sample_weight"]
OUTCOMES = ("non-inferior", "inferior", "inconclusive")


def align_predictions(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One frame with each model's score as a column, matched by physical event.

    Every model must cover exactly the same events with the same process, class
    and stored physical weight. Rows are matched by event_id, never by position.
    """
    if len(predictions) < 2:
        raise ValueError("A paired comparison needs at least two models")
    merged = None
    for name, frame in predictions.items():
        if name in IDENTITY:
            raise ValueError(f"Model name {name!r} collides with an identity column")
        if frame.event_id.duplicated().any():
            raise ValueError(f"{name}: duplicate physical events")
        part = frame[IDENTITY].assign(**{name: frame.bdt_score.to_numpy(dtype=float)})
        if merged is None:
            merged = part
            continue
        if set(part.event_id) != set(merged.event_id):
            raise ValueError(f"{name}: scored a different set of physical events")
        joined = merged.merge(part, on="event_id", suffixes=("", "_other"), validate="one_to_one")
        for column in ("sample", "target", "sample_weight"):
            if not (joined[column].to_numpy() == joined[f"{column}_other"].to_numpy()).all():
                raise ValueError(f"{name}: {column} differs for the same physical events")
        merged = joined.drop(columns=[f"{column}_other" for column in ("sample", "target", "sample_weight")])
    return merged.sort_values("event_id", kind="stable").reset_index(drop=True)


def roc_path(target, scores, weights, counts=None):
    """Weighted ROC points in threshold order, from all-rejected to all-accepted.

    Acceptance is score >= threshold; one point per distinct score. Returns
    (background efficiency, signal efficiency, threshold) arrays, all ordered
    from the tightest threshold to the loosest.
    """
    y = np.asarray(target)
    w = np.asarray(weights, dtype=float) * (1.0 if counts is None else np.asarray(counts, dtype=float))
    unique, inverse = np.unique(np.asarray(scores, dtype=float), return_inverse=True)
    signal = np.bincount(inverse, weights=w * (y == 1), minlength=len(unique))
    background = np.bincount(inverse, weights=w * (y == 0), minlength=len(unique))
    if signal.sum() <= 0 or background.sum() <= 0:
        raise ValueError("Both classes need positive weight")
    eps_s = np.r_[0.0, np.cumsum(signal[::-1]) / signal.sum()]
    eps_b = np.r_[0.0, np.cumsum(background[::-1]) / background.sum()]
    return eps_b, eps_s, np.r_[np.inf, unique[::-1]]


def _interpolate(path, background_efficiency):
    if not 0 < background_efficiency < 1:
        raise ValueError("Background efficiency must lie strictly between 0 and 1")
    eps_b, eps_s, thresholds = path
    last = int(np.flatnonzero(eps_b <= background_efficiency)[-1])
    efficiency, upper = eps_s[last], eps_b[last]
    if eps_b[last] < background_efficiency:
        upper = eps_b[last + 1]
        efficiency += (background_efficiency - eps_b[last]) / (upper - eps_b[last]) * (eps_s[last + 1] - eps_s[last])
    return float(efficiency), float(thresholds[last]), float(eps_b[last]), float(upper)


def working_point(target, scores, weights, *, background_efficiency, counts=None) -> dict:
    """Signal efficiency at a fixed weighted background efficiency.

    Linear interpolation along the ROC path between the last point at or below
    the target and the next point above it (the same convention as interpolating
    sklearn's roc_curve). The reported threshold is the loosest one whose
    background efficiency does not exceed the target; it is descriptive only.
    Tied scores can make the bracketing points far apart; both ends are returned.
    """
    efficiency, threshold, achieved, upper = _interpolate(roc_path(target, scores, weights, counts),
                                                          background_efficiency)
    return dict(background_efficiency=float(background_efficiency), signal_efficiency=efficiency,
                threshold=threshold, achieved_background_efficiency=achieved, next_background_efficiency=upper)


def selection_support(target, scores, weights, threshold) -> dict:
    """MC events and background effective count kept by score >= threshold."""
    y, kept = np.asarray(target), np.asarray(scores, dtype=float) >= threshold
    w = np.asarray(weights, dtype=float)[kept & (y == 0)]
    return dict(signal_mc=int((kept & (y == 1)).sum()), background_mc=int(len(w)),
                background_neff=float(w.sum() ** 2 / (w**2).sum()) if len(w) else 0.0)


def resample_counts(samples, rng) -> np.ndarray:
    """Within-process event multiplicities: every process keeps its event count."""
    samples = np.asarray(samples)
    counts = np.zeros(len(samples), dtype=np.int64)
    for sample in np.unique(samples):
        group = np.flatnonzero(samples == sample)
        counts[group] = np.bincount(rng.integers(len(group), size=len(group)), minlength=len(group))
    return counts


def paired_metrics(frame, models, weights, *, background_efficiencies, counts=None) -> pd.DataFrame:
    """Weighted AUC and signal efficiency at each background efficiency, per model."""
    y = frame.target.to_numpy()
    w = np.asarray(weights, dtype=float)
    effective = w if counts is None else w * counts
    rows = []
    for model in models:
        scores = frame[model].to_numpy(dtype=float)
        row = dict(model=model, auc_weighted=safe_auc(y, scores, effective))
        path = roc_path(y, scores, w, counts)
        for target in background_efficiencies:
            row[f"signal_efficiency_at_{target:g}"] = _interpolate(path, target)[0]
        rows.append(row)
    return pd.DataFrame(rows)


def joint_paired_bootstrap(groups, *, background_efficiencies, replicates, seed) -> pd.DataFrame:
    """Replica metrics for every group and model from one shared resampling of physical events.

    groups maps a name (e.g. a hypothesis mass) to (aligned frame, models, weights).
    An event_id present in several groups, such as a common background event, gets
    the same multiplicity in each of them within a replica.
    """
    if replicates < 2:
        raise ValueError("Use at least two bootstrap replicates")
    union = pd.concat([frame[["event_id", "sample"]] for frame, _, _ in groups.values()], ignore_index=True)
    union = union.drop_duplicates()
    if union.event_id.duplicated().any():
        raise ValueError("One physical event belongs to different processes in different groups")
    positions = pd.Index(union.event_id)
    rows = {name: positions.get_indexer(frame.event_id) for name, (frame, _, _) in groups.items()}
    samples = union["sample"].to_numpy()
    rng = np.random.default_rng(seed)
    parts = []
    for replicate in range(replicates):
        counts = resample_counts(samples, rng)
        for name, (frame, models, weights) in groups.items():
            metrics = paired_metrics(frame, models, weights, background_efficiencies=background_efficiencies,
                                     counts=counts[rows[name]])
            # A list, so tuple names such as (mass, seed) stay one label per row.
            parts.append(metrics.assign(group=[name] * len(metrics), replicate=replicate))
    return pd.concat(parts, ignore_index=True)


def paired_bootstrap(frame, models, weights, *, background_efficiencies, replicates, seed) -> pd.DataFrame:
    """Replica metrics for every model; replica r uses the same multiplicities for all models."""
    replicas = joint_paired_bootstrap({0: (frame, models, weights)}, background_efficiencies=background_efficiencies,
                                      replicates=replicates, seed=seed)
    return replicas.drop(columns="group")


def noninferiority(reference, candidate, reference_replicas, candidate_replicas, *, margin, relative,
                   confidence) -> dict:
    """Loss of the candidate against the reference, with one-sided bootstrap bounds.

    loss = 1 - candidate/reference if relative, else reference - candidate; a
    negative loss means the candidate is better. Non-inferior when the upper
    bound is within the margin, inferior when the lower bound exceeds it, and
    inconclusive otherwise.
    """
    if not 0.5 < confidence < 1 or margin < 0:
        raise ValueError("Invalid confidence level or margin")
    reference_replicas = np.asarray(reference_replicas, dtype=float)
    candidate_replicas = np.asarray(candidate_replicas, dtype=float)
    if reference_replicas.shape != candidate_replicas.shape:
        raise ValueError("Paired replicas must align")

    def loss(ref, cand):
        return 1.0 - cand / ref if relative else ref - cand

    estimate = float(loss(reference, candidate))
    replicas = loss(reference_replicas, candidate_replicas)
    lower, upper = (float(x) for x in np.quantile(replicas, [1.0 - confidence, confidence]))
    outcome = "non-inferior" if upper <= margin else "inferior" if lower > margin else "inconclusive"
    return dict(loss=estimate, lower=lower, upper=upper, margin=float(margin), relative=bool(relative),
                confidence=float(confidence), outcome=outcome)


def combine_outcomes(outcomes) -> str:
    """All criteria must be non-inferior; any inferior criterion makes the result inferior."""
    outcomes = list(outcomes)
    if not outcomes or any(outcome not in OUTCOMES for outcome in outcomes):
        raise ValueError("Unknown or missing outcome")
    if "inferior" in outcomes:
        return "inferior"
    return "non-inferior" if all(outcome == "non-inferior" for outcome in outcomes) else "inconclusive"
