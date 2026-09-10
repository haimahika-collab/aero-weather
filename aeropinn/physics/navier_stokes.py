"""Steady, 2D, incompressible, nondimensional Navier-Stokes residuals (Plan Section 7/11).

Continuity:      u_x + v_y = 0
x-momentum:      u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy) = 0
y-momentum:      u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy) = 0

`forward_fn` is deliberately decoupled from any specific network architecture: it is
any callable (x, y, cond) -> (u, v, p) where x, y are (N,1) leaf tensors with
requires_grad=True and cond is an (N,k) tensor of auxiliary conditioning inputs
(SDF, m, p, t, alpha, Re, ...) that does not need gradients. This keeps the residual
math testable against manufactured solutions independent of model.py (Section 24).
"""
from typing import Callable, Dict

import torch

ForwardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], "tuple[torch.Tensor, torch.Tensor, torch.Tensor]"]


def _grad(outputs: torch.Tensor, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
    if not outputs.requires_grad:
        # A derivative whose VALUE is a true constant (e.g. the second derivative of a
        # term that is exactly linear in `inputs`) legitimately has no further graph
        # connection to `inputs` at all — not merely "unused", but genuinely
        # non-differentiable further. autograd.grad requires outputs.requires_grad, so
        # short-circuit to an explicit zero rather than letting it raise.
        return torch.zeros_like(inputs, requires_grad=True)
    ones = torch.ones_like(outputs)
    grad = torch.autograd.grad(
        outputs, inputs, grad_outputs=ones, create_graph=True, retain_graph=True, allow_unused=True, **kwargs
    )[0]
    if grad is None:
        # A first derivative that is identically constant (e.g. a locally linear or
        # constant field) is legitimately disconnected from the input in the graph, so
        # its own derivative (needed for the second-order viscous terms) has no grad_fn.
        # That is a real, valid zero, not a bug — substitute a requires_grad leaf (not a
        # plain zeros tensor) so a further _grad() call on this value also degrades
        # gracefully via the same allow_unused path, instead of hitting "does not
        # require grad" on a tensor with no grad tracking at all.
        grad = torch.zeros_like(inputs, requires_grad=True)
    return grad


def velocity_pressure_derivatives(forward_fn: ForwardFn, x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
    """First and second spatial derivatives of (u, v, p) needed for the NS residual."""
    if not x.requires_grad:
        x = x.requires_grad_(True)
    if not y.requires_grad:
        y = y.requires_grad_(True)

    u, v, p = forward_fn(x, y, cond)

    u_x = _grad(u, x)
    u_y = _grad(u, y)
    v_x = _grad(v, x)
    v_y = _grad(v, y)
    p_x = _grad(p, x)
    p_y = _grad(p, y)

    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)
    v_xx = _grad(v_x, x)
    v_yy = _grad(v_y, y)

    return {
        "u": u, "v": v, "p": p,
        "u_x": u_x, "u_y": u_y, "v_x": v_x, "v_y": v_y, "p_x": p_x, "p_y": p_y,
        "u_xx": u_xx, "u_yy": u_yy, "v_xx": v_xx, "v_yy": v_yy,
    }


def pde_residual(forward_fn: ForwardFn, x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor, Re: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Continuity + x/y-momentum residuals at each (x, y) collocation point.

    Re: (N, 1) tensor, one Reynolds number per point (varies across a mixed-geometry
    training batch, per Plan Section 10's batching strategy).
    """
    d = velocity_pressure_derivatives(forward_fn, x, y, cond)

    continuity = d["u_x"] + d["v_y"]
    momentum_x = d["u"] * d["u_x"] + d["v"] * d["u_y"] + d["p_x"] - (1.0 / Re) * (d["u_xx"] + d["u_yy"])
    momentum_y = d["u"] * d["v_x"] + d["v"] * d["v_y"] + d["p_y"] - (1.0 / Re) * (d["v_xx"] + d["v_yy"])

    return {
        "continuity": continuity,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "u": d["u"], "v": d["v"], "p": d["p"],
    }


def pde_loss(residuals: Dict[str, torch.Tensor]) -> torch.Tensor:
    """L_PDE = mean squared residual, pooling continuity + both momentum components (Section 11)."""
    return (
        residuals["continuity"].pow(2).mean()
        + residuals["momentum_x"].pow(2).mean()
        + residuals["momentum_y"].pow(2).mean()
    )
