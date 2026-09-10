"""Nondimensionalization utilities (Plan Section 7).

Characteristic length L = chord c. Characteristic velocity U = U_inf.
x* = x/c, y* = y/c, u* = u/U_inf, v* = v/U_inf, p* = p/(rho * U_inf**2), Re = rho*U_inf*c/mu.

Pressure coefficient convention used throughout this project: since p* is normalized
by the FULL rho*U_inf**2 (not the usual 1/2 rho U_inf**2), Cp = 2 * p_star. This must
stay consistent everywhere p is reported — see Plan Section 7's explicit warning about
normalization-convention bugs.
"""
import math

import torch


def reynolds_number(rho: float, u_inf: float, chord: float, mu: float) -> float:
    return rho * u_inf * chord / mu


def cp_from_pressure_star(p_star: torch.Tensor) -> torch.Tensor:
    """Cp = (p - p_inf) / (0.5 rho U_inf^2) = 2 * p_star, given p_star's rho*U_inf^2 scaling."""
    return 2.0 * p_star


def freestream_components(alpha_deg: torch.Tensor):
    """(u_inf_x, u_inf_y) unit-freestream components for angle of attack alpha (degrees)."""
    alpha_rad = alpha_deg * (math.pi / 180.0)
    return torch.cos(alpha_rad), torch.sin(alpha_rad)
