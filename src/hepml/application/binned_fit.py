"""Binned shape-fit sensitivity: jet calibration, c-b mass, bin merging and significance.

Filesystem-free. The significance is the Gaussian limit of the binned profile
likelihood for a small signal (Asimov data, discovery test):

    Z^2 = s^T C^-1 s,  C = diag(B + MC variance + (shape B)^2) + normalization^2 B B^T,

where the normalization uncertainty is shared by every bin and the shape
uncertainty is independent per bin. For one bin without MC or shape terms it
reduces to S / sqrt(B + (normalization B)^2), the S << B limit of Cowan's Eq. 20.
"""

from __future__ import annotations

import numpy as np

from hepml.domain.physics import delta_r, four_vector


def asimov_significance(signal, background, *, normalization, shape=0.0, mc_variance=None) -> float:
    s, b = np.asarray(signal, dtype=float), np.asarray(background, dtype=float)
    if s.shape != b.shape or s.ndim != 1 or not len(s):
        raise ValueError("Signal and background need one value per bin")
    if not (np.isfinite(s).all() and np.isfinite(b).all()) or (s < 0).any() or (b <= 0).any():
        raise ValueError("Every bin needs finite, non-negative signal and positive background")
    if min(normalization, shape) < 0:
        raise ValueError("Uncertainties must be non-negative")
    diagonal = b + (shape * b) ** 2
    if mc_variance is not None:
        variance = np.asarray(mc_variance, dtype=float)
        if variance.shape != b.shape or (variance < 0).any() or not np.isfinite(variance).all():
            raise ValueError("MC variance needs one finite, non-negative value per bin")
        diagonal = diagonal + variance
    covariance = np.diag(diagonal) + normalization**2 * np.outer(b, b)
    return float(np.sqrt(s @ np.linalg.solve(covariance, s)))


def log_bins(x_min, x_max, step) -> np.ndarray:
    """Edges in x: an underflow bin, log-spaced bins from x_min reaching x_max, an overflow bin."""
    if not 0 < x_min < x_max or step <= 0:
        raise ValueError("Invalid binning")
    count = int(np.ceil(np.log(x_max / x_min) / step - 1e-9))
    return np.r_[0.0, x_min * np.exp(step * np.arange(count + 1)), np.inf]


def merge_bins(sum_w, sum_w2, min_neff) -> list[list[int]]:
    """Group adjacent fine bins from low x upwards until each group's N_eff reaches min_neff.

    A trailing group below the threshold joins the previous one. Only background
    enters, so signal never shapes the binning.
    """
    sum_w, sum_w2 = np.asarray(sum_w, dtype=float), np.asarray(sum_w2, dtype=float)
    groups, current, w, w2 = [], [], 0.0, 0.0
    for index in range(len(sum_w)):
        current.append(index)
        w, w2 = w + sum_w[index], w2 + sum_w2[index]
        if w2 > 0 and w * w / w2 >= min_neff:
            groups.append(current)
            current, w, w2 = [], 0.0, 0.0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def score_boundaries(score, weight, efficiencies) -> np.ndarray:
    """Increasing score thresholds keeping at most each weighted fraction of the events.

    Acceptance is score >= threshold. efficiencies must decrease strictly; each
    threshold is the lowest score whose accepted fraction does not exceed it.
    Pass background only, so signal never shapes the categories.
    """
    score, weight = np.asarray(score, dtype=float), np.asarray(weight, dtype=float)
    efficiencies = np.asarray(efficiencies, dtype=float)
    if score.shape != weight.shape or not len(score) or (weight <= 0).any() or not np.isfinite(score).all():
        raise ValueError("Need finite scores with positive weights")
    if not ((efficiencies > 0) & (efficiencies < 1)).all() or (np.diff(efficiencies) >= 0).any():
        raise ValueError("Efficiencies must lie in (0, 1) and decrease strictly")
    order = np.argsort(-score, kind="stable")
    ranked, accepted = score[order], np.cumsum(weight[order]) / weight.sum()
    last_of_tie = np.r_[ranked[1:] != ranked[:-1], True]  # a threshold accepts every tied score
    ranked, accepted = ranked[last_of_tie], accepted[last_of_tie]
    index = np.searchsorted(accepted, efficiencies, side="right") - 1
    if (index < 0).any():
        raise ValueError("The highest-scoring event alone exceeds a requested efficiency")
    thresholds = ranked[index]
    if (np.diff(thresholds) <= 0).any():
        raise ValueError("Too few events to separate the requested efficiencies")
    return thresholds


def boundaries_follow_rule(score, weight, thresholds, efficiencies) -> bool:
    """score_boundaries' contract, exactly: each threshold accepts at most its fraction, and the
    next lower score would accept more. A single heavy event may leave a threshold far below its
    fraction; that is the rule, not an error."""
    score, weight = np.asarray(score, dtype=float), np.asarray(weight, dtype=float)
    total, tolerance = weight.sum(), 1e-9
    for threshold, efficiency in zip(thresholds, efficiencies):
        if weight[score >= threshold].sum() / total > efficiency + tolerance:
            return False
        lower = score[score < threshold]
        if len(lower) and weight[score >= lower.max()].sum() / total <= efficiency - tolerance:
            return False
    return True


def response_curve(reco_pt, parton_pt, edges):
    """Median parton/reco pT per reco-pT bin; nodes are each bin's median reco pT.

    edges are the lower bin edges in GeV; the last bin is open. Jets below the
    first edge are ignored.
    """
    reco_pt, parton_pt = np.asarray(reco_pt, dtype=float), np.asarray(parton_pt, dtype=float)
    keep = np.isfinite(reco_pt) & np.isfinite(parton_pt) & (reco_pt > 0) & (parton_pt > 0)
    reco_pt, parton_pt = reco_pt[keep], parton_pt[keep]
    index = np.digitize(reco_pt, np.asarray(edges, dtype=float)) - 1
    nodes, values, counts = [], [], []
    for k in range(len(edges)):
        inside = index == k
        if not inside.any():
            raise ValueError(f"No calibration jets in reco-pT bin starting at {edges[k]} GeV")
        nodes.append(float(np.median(reco_pt[inside])))
        values.append(float(np.median(parton_pt[inside] / reco_pt[inside])))
        counts.append(int(inside.sum()))
    return np.array(nodes), np.array(values), np.array(counts)


def response_scale(pt, nodes, values):
    """Calibration factor: linear in log(pT) between nodes, constant beyond the outer nodes."""
    return np.interp(np.log(np.asarray(pt, dtype=float)), np.log(nodes), values)


def pair_mass(first, second, first_scale=1.0, second_scale=1.0):
    """Invariant mass of two jets given as (pt, eta, phi, mass) arrays, each scaled as a four-vector."""
    px1, py1, pz1, e1 = (first_scale * c for c in four_vector(*first))
    px2, py2, pz2, e2 = (second_scale * c for c in four_vector(*second))
    return np.sqrt(np.maximum((e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2, 0.0))


def closest_cb_mass(b1, b2, c1, tested_mass, *, scale_b1=1.0, scale_b2=1.0, scale_c1=1.0):
    """c-b mass of the pairing closest to the tested mass, and which b-jet was chosen (1 or 2)."""
    first = pair_mass(b1, c1, scale_b1, scale_c1)
    second = pair_mass(b2, c1, scale_b2, scale_c1)
    pick_first = np.abs(first - tested_mass) <= np.abs(second - tested_mass)
    return np.where(pick_first, first, second), np.where(pick_first, 1, 2)


def match_to_partons(jet_eta, jet_phi, parton_pid, parton_status, parton_eta, parton_phi, *, flavour, max_delta_r):
    """Index of the nearest status-23 parton of |pid| == flavour within max_delta_r, or -1."""
    eligible = (np.asarray(parton_status) == 23) & (np.abs(np.asarray(parton_pid)) == flavour)
    if not eligible.any():
        return -1
    distances = np.where(eligible, delta_r(jet_eta, jet_phi, np.asarray(parton_eta), np.asarray(parton_phi)), np.inf)
    nearest = int(np.argmin(distances))
    return nearest if distances[nearest] < max_delta_r else -1
