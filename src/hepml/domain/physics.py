"""Kinematics helpers shared by the dataset-building stages.

All functions are vectorized over numpy arrays.
"""

from __future__ import annotations

import numpy as np

# A jet is passed around as a (pt, eta, phi, mass) tuple of arrays.
Jet = tuple


def four_vector(pt, eta, phi, mass):
    """Return (px, py, pz, E) for pt/eta/phi/mass arrays."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p2 = px * px + py * py + pz * pz
    e = np.sqrt(np.maximum(p2 + mass * mass, 0.0))
    return px, py, pz, e


def inv_mass(*jets: Jet):
    """Invariant mass of any number of (pt, eta, phi, mass) jets.

    Negative mass-squared from floating-point round-off is clamped to 0.
    """
    if not jets:
        raise ValueError("inv_mass needs at least one (pt, eta, phi, mass) jet")
    px_t, py_t, pz_t, e_t = 0.0, 0.0, 0.0, 0.0
    for pt, eta, phi, mass in jets:
        px, py, pz, e = four_vector(pt, eta, phi, mass)
        px_t = px_t + px
        py_t = py_t + py
        pz_t = pz_t + pz
        e_t = e_t + e
    m2 = e_t * e_t - (px_t * px_t + py_t * py_t + pz_t * pz_t)
    return np.sqrt(np.maximum(m2, 0.0))


def delta_r(eta1, phi1, eta2, phi2):
    """Angular distance sqrt(deta^2 + dphi^2) with dphi wrapped to [-pi, pi]."""
    dphi = phi1 - phi2
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    deta = eta1 - eta2
    return np.sqrt(deta * deta + dphi * dphi)
