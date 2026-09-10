import numpy as np
import pytest

from aeropinn.geometry.naca import naca4_surface, naca_camber, naca_thickness


def test_symmetric_airfoil_zero_camber():
    x = np.linspace(0, 1, 50)
    yc = naca_camber(x, m=0.0, p=0.0)
    assert np.allclose(yc, 0.0)


def test_symmetric_airfoil_mirror_surfaces():
    # NACA 0012: m=0 -> upper/lower surfaces should be exact mirror images about y=0
    upper, lower = naca4_surface(m=0.0, p=0.0, t=0.12, n_points=100)
    assert np.allclose(upper[:, 0], lower[:, 0])  # same x stations
    assert np.allclose(upper[:, 1], -lower[:, 1])  # mirrored y


def test_naca0012_max_thickness_matches_spec():
    # NACA 0012 -> max half-thickness should be t/2 = 0.06, occurring near x/c ~ 0.3
    x = np.linspace(1e-6, 1.0, 2000)
    yt = naca_thickness(x, t=0.12)
    i_max = np.argmax(yt)
    assert yt[i_max] == pytest.approx(0.06, rel=0.05)
    assert 0.2 < x[i_max] < 0.4


def test_surface_starts_at_leading_edge_and_ends_at_trailing_edge():
    upper, lower = naca4_surface(m=0.04, p=0.4, t=0.12, n_points=100)
    # Leading edge: thickness is exactly zero at x=0, so both surfaces meet exactly at (0,0)
    # regardless of camber slope there.
    assert upper[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert lower[0, 0] == pytest.approx(0.0, abs=1e-9)
    # Trailing edge: the standard open-TE NACA4 thickness formula leaves a small residual
    # thickness at x=1 (~0.002*t), which combined with nonzero camber slope there rotates
    # the surface x-coordinate slightly off 1.0 — a known, expected property of this
    # parameterization, not a bug. Tolerance reflects that small, real offset.
    assert upper[-1, 0] == pytest.approx(1.0, abs=2e-3)
    assert lower[-1, 0] == pytest.approx(1.0, abs=2e-3)


def test_cambered_airfoil_camber_line_peaks_near_p():
    m, p = 0.04, 0.4
    x = np.linspace(0, 1, 2000)
    yc = naca_camber(x, m=m, p=p)
    i_max = np.argmax(yc)
    assert x[i_max] == pytest.approx(p, abs=0.02)
    assert yc[i_max] == pytest.approx(m, rel=0.05)
