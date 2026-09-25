"""Hand-computed checks for the kinematics helpers."""

import numpy as np
import pytest

from hepml.domain.physics import delta_r, four_vector, inv_mass


def jet(pt, eta, phi, m):
    return (np.array([pt]), np.array([eta]), np.array([phi]), np.array([m]))


def test_four_vector_components():
    px, py, pz, e = four_vector(*jet(10.0, 0.0, 0.0, 0.0))
    assert px[0] == pytest.approx(10.0)
    assert py[0] == pytest.approx(0.0)
    assert pz[0] == pytest.approx(0.0)
    assert e[0] == pytest.approx(10.0)


def test_inv_mass_back_to_back_massless():
    # two massless 50 GeV jets, back to back: m = 100 GeV
    m = inv_mass(jet(50.0, 0.0, 0.0, 0.0), jet(50.0, 0.0, np.pi, 0.0))
    assert m[0] == pytest.approx(100.0)


def test_inv_mass_single_particle_at_rest():
    m = inv_mass(jet(0.0, 0.0, 0.0, 125.0))
    assert m[0] == pytest.approx(125.0)


def test_inv_mass_three_body_matches_manual_four_vectors():
    jets = [jet(40.0, 0.5, 0.1, 5.0), jet(35.0, -1.2, 2.0, 4.0), jet(25.0, 0.8, -2.5, 3.0)]
    px = py = pz = e = 0.0
    for j in jets:
        a, b, c, d = four_vector(*j)
        px, py, pz, e = px + a, py + b, pz + c, e + d
    expected = np.sqrt(np.maximum(e * e - px * px - py * py - pz * pz, 0.0))
    assert inv_mass(*jets)[0] == pytest.approx(expected[0])


def test_inv_mass_clamps_roundoff_to_zero():
    # identical massless jets: m should be exactly 0, never NaN
    m = inv_mass(jet(30.0, 1.0, 1.0, 0.0), jet(30.0, 1.0, 1.0, 0.0))
    assert m[0] == pytest.approx(0.0)
    assert np.isfinite(m).all()


def test_inv_mass_requires_a_jet():
    with pytest.raises(ValueError):
        inv_mass()


def test_delta_r_simple():
    dr = delta_r(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
    assert dr[0] == pytest.approx(1.0)


def test_delta_r_phi_wraparound():
    # phi = 3.1 vs -3.1 are only ~0.083 apart across the -pi/pi boundary
    dr = delta_r(np.array([0.0]), np.array([3.1]), np.array([0.0]), np.array([-3.1]))
    assert dr[0] == pytest.approx(2 * np.pi - 6.2, abs=1e-6)
