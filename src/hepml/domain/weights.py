"""The physics event-weight formula, defined once.

sample_weight = lumi * xs_pb / n_events_total          (uniform per event)

where
  xs_pb          = the sample's MERGED cross-section (from an external rate table or
                   an inline manifest value; for the signal rho_tc grid this
                   is the reference rho's xs)
  n_events_total = total number of GENERATED events in the sample's ROOT
                   file(s), a raw count - generator per-event weights are
                   deliberately NOT used in this convention
  lumi           = target integrated luminosity, in the inverse unit of
                   xs_pb (xs in pb -> lumi in pb^-1)

Summed over selected events this gives the expected yield
  N = xs * lumi * (n_passed / n_total).

This is convention v2 (external merged xs). The v1 convention based on the
Delphes Event.CrossSection branch (lumi * 2 * gen_weight * evt_xs / sum_w_all)
was retired in this commit and remains available in git history.
"""

from __future__ import annotations

import numpy as np


def cross_section_factors(targets, *, signal_k_factor=1.0, background_k_factor=1.0):
    """Evaluation-only correction to raw manifest rates; never modify stored weights."""
    targets = np.asarray(targets)
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Cross-section corrections require binary targets")
    if any(not np.isfinite(k) or k <= 0 for k in (signal_k_factor, background_k_factor)):
        raise ValueError("Cross-section k-factors must be finite and positive")
    return np.where(targets == 1, signal_k_factor, background_k_factor)


def balanced_hypothesis_weights(y, physical_weights, hypotheses, prior, *, event_ids):
    """Ephemeral loss weights: equal class totals and the same q(h) in each class.

    Physical weights must already describe the intended process mixture. Within
    each class/hypothesis their relative proportions are preserved. Counting unique
    events fixes the total loss weight when background hypotheses are replicated.
    This function never changes or returns a replacement physics-weight column.
    """
    y = np.asarray(y)
    weights = np.asarray(physical_weights, dtype=float)
    hypotheses, event_ids = np.asarray(hypotheses), np.asarray(event_ids)
    if (not (len(y) == len(weights) == len(hypotheses) == len(event_ids))
            or set(y) != {0, 1} or not np.isfinite(weights).all() or (weights < 0).any()
            or set(hypotheses) != set(prior)
            or any(not np.isfinite(p) or p <= 0 for p in prior.values())
            or not np.isclose(sum(prior.values()), 1)):
        raise ValueError("Invalid physical weights or hypothesis prior")
    target_total = len(np.unique(event_ids)) / 2
    result = np.empty_like(weights)
    for target in (0, 1):
        for hypothesis, probability in prior.items():
            mask = (y == target) & (hypotheses == hypothesis)
            total = weights[mask].sum()
            if total <= 0:
                raise ValueError("Every class must have support at every training hypothesis")
            result[mask] = weights[mask] * (target_total * probability / total)
    return result


def compute_sample_weight(xs_pb, n_events_total, *, lumi: float):
    """Uniform per-event physics weight; broadcasts over arrays/Series."""
    return lumi * xs_pb / n_events_total
