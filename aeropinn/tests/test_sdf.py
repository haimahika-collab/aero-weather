import numpy as np
import pytest
import torch

from aeropinn.geometry.naca import naca4_surface
from aeropinn.geometry.sdf import signed_distance_numpy, signed_distance_torch


def test_far_point_is_positive_and_large():
    d = signed_distance_numpy(np.array([[5.0, 5.0]]), m=0.02, p=0.4, t=0.12)
    assert d[0] > 5.0  # must be at least roughly the straight-line distance to the origin region


def test_point_on_camber_line_of_thick_airfoil_is_inside():
    # A thick symmetric airfoil (t=0.2) at x=0.5, y=0 should be well inside the surface.
    d = signed_distance_numpy(np.array([[0.5, 0.0]]), m=0.0, p=0.0, t=0.2)
    assert d[0] < 0.0


def test_point_on_surface_is_approximately_zero():
    upper, _ = naca4_surface(m=0.02, p=0.4, t=0.12, n_points=200)
    probe = upper[100:101]  # a genuine surface point
    d = signed_distance_numpy(probe, m=0.02, p=0.4, t=0.12, n_samples=400)
    assert abs(d[0]) < 5e-3


def test_numpy_and_torch_agree():
    m, p, t = 0.03, 0.35, 0.15
    query = np.array([[0.5, 0.05], [0.5, -0.05], [2.0, 1.0], [-0.5, 0.0]])
    d_np = signed_distance_numpy(query, m, p, t, n_samples=400)

    query_t = torch.tensor(query, dtype=torch.float64)
    m_t = torch.full((4,), m, dtype=torch.float64)
    p_t = torch.full((4,), p, dtype=torch.float64)
    t_t = torch.full((4,), t, dtype=torch.float64)
    d_torch = signed_distance_torch(query_t, m_t, p_t, t_t, n_samples=400).numpy()

    assert np.allclose(d_np, d_torch, atol=1e-6)


def test_torch_sdf_is_differentiable_wrt_query_point():
    query = torch.tensor([[1.5, 0.5]], dtype=torch.float64, requires_grad=True)
    m_t = torch.tensor([0.02], dtype=torch.float64)
    p_t = torch.tensor([0.4], dtype=torch.float64)
    t_t = torch.tensor([0.12], dtype=torch.float64)
    d = signed_distance_torch(query, m_t, p_t, t_t, n_samples=200)
    d.sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
