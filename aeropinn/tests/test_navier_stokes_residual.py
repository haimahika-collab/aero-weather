import torch

from aeropinn.physics.navier_stokes import pde_residual, velocity_pressure_derivatives

torch.set_default_dtype(torch.float64)


def _constant_freestream_forward(x, y, cond):
    # Multiply by x/y (times zero) rather than using torch.full_like/zeros_like so that
    # u, v, p stay connected to x, y in the autograd graph — otherwise autograd.grad
    # raises "does not require grad" instead of correctly returning zero derivatives.
    zero = 0.0 * x + 0.0 * y
    u = zero + 0.99
    v = zero + 0.14
    p = zero
    return u, v, p


def _manufactured_forward(x, y, cond):
    # Arbitrary smooth field, deliberately NOT a solution of the NS equations.
    u = torch.sin(x) * torch.cos(y)
    v = torch.cos(x) * torch.sin(y)
    p = x**2 * y
    return u, v, p


def test_constant_field_gives_zero_residual():
    n = 16
    x = torch.rand(n, 1) * 2 - 1
    y = torch.rand(n, 1) * 2 - 1
    cond = torch.zeros(n, 1)
    Re = torch.full((n, 1), 5000.0)

    res = pde_residual(_constant_freestream_forward, x, y, cond, Re)
    assert torch.allclose(res["continuity"], torch.zeros_like(res["continuity"]), atol=1e-10)
    assert torch.allclose(res["momentum_x"], torch.zeros_like(res["momentum_x"]), atol=1e-10)
    assert torch.allclose(res["momentum_y"], torch.zeros_like(res["momentum_y"]), atol=1e-10)


def test_nontrivial_field_gives_nonzero_residual():
    n = 16
    x = torch.rand(n, 1) * 2 - 1
    y = torch.rand(n, 1) * 2 - 1
    cond = torch.zeros(n, 1)
    Re = torch.full((n, 1), 5000.0)

    res = pde_residual(_manufactured_forward, x, y, cond, Re)
    # A field with no physical reason to satisfy NS should show real residual magnitude.
    assert res["momentum_x"].abs().mean().item() > 1e-3


def test_autograd_derivatives_match_finite_difference():
    torch.manual_seed(0)
    n = 12
    x0 = torch.rand(n, 1) * 2 - 1
    y0 = torch.rand(n, 1) * 2 - 1
    cond = torch.zeros(n, 1)

    x = x0.clone().requires_grad_(True)
    y = y0.clone().requires_grad_(True)
    d = velocity_pressure_derivatives(_manufactured_forward, x, y, cond)

    def u_val(xx, yy):
        return torch.sin(xx) * torch.cos(yy)

    def v_val(xx, yy):
        return torch.cos(xx) * torch.sin(yy)

    def p_val(xx, yy):
        return xx**2 * yy

    h = 1e-3
    u_x_fd = (u_val(x0 + h, y0) - u_val(x0 - h, y0)) / (2 * h)
    u_y_fd = (u_val(x0, y0 + h) - u_val(x0, y0 - h)) / (2 * h)
    v_x_fd = (v_val(x0 + h, y0) - v_val(x0 - h, y0)) / (2 * h)
    v_y_fd = (v_val(x0, y0 + h) - v_val(x0, y0 - h)) / (2 * h)
    p_x_fd = (p_val(x0 + h, y0) - p_val(x0 - h, y0)) / (2 * h)
    p_y_fd = (p_val(x0, y0 + h) - p_val(x0, y0 - h)) / (2 * h)

    u_xx_fd = (u_val(x0 + h, y0) - 2 * u_val(x0, y0) + u_val(x0 - h, y0)) / h**2
    u_yy_fd = (u_val(x0, y0 + h) - 2 * u_val(x0, y0) + u_val(x0, y0 - h)) / h**2
    v_xx_fd = (v_val(x0 + h, y0) - 2 * v_val(x0, y0) + v_val(x0 - h, y0)) / h**2
    v_yy_fd = (v_val(x0, y0 + h) - 2 * v_val(x0, y0) + v_val(x0, y0 - h)) / h**2

    assert torch.allclose(d["u_x"], u_x_fd, atol=1e-4)
    assert torch.allclose(d["u_y"], u_y_fd, atol=1e-4)
    assert torch.allclose(d["v_x"], v_x_fd, atol=1e-4)
    assert torch.allclose(d["v_y"], v_y_fd, atol=1e-4)
    assert torch.allclose(d["p_x"], p_x_fd, atol=1e-4)
    assert torch.allclose(d["p_y"], p_y_fd, atol=1e-4)
    assert torch.allclose(d["u_xx"], u_xx_fd, atol=1e-3)
    assert torch.allclose(d["u_yy"], u_yy_fd, atol=1e-3)
    assert torch.allclose(d["v_xx"], v_xx_fd, atol=1e-3)
    assert torch.allclose(d["v_yy"], v_yy_fd, atol=1e-3)
