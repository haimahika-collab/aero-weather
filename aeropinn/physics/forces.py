"""Cp-panel integration: CL/CD from a surface pressure distribution (Plan Section
19's inverse-design objective needs this — "maximize CL/CD" requires a differentiable
CL/CD, not just SU2's own force-solver numbers on the 6 held-out cases).

Standard 2D panel-integration formula (e.g. Anderson, Fundamentals of Aerodynamics):
given surface points traversed once around the closed body and the Cp at each,

    CN =  (1/c) * sum_j( Cp_avg_j * dx_j )
    CA = -(1/c) * sum_j( Cp_avg_j * dy_j )
    CL = CN*cos(alpha) - CA*sin(alpha)
    CD = CN*sin(alpha) + CA*cos(alpha)

where (dx_j, dy_j) is the panel-j vector along the traverse direction (loop order
below). Sign was verified empirically, not assumed: integrating actual SU2 reference
Cp (surf_cp_true) on 5 held-out cases and comparing to SU2's own force-solver
cl_true/cd_true showed the naive formula's CN/CA sign backwards for this traverse
direction; the signs above are the corrected ones. With them, CL matches SU2 to
~5-10% relative error across those cases -- a credible pressure-only CL.

CD from this function is PRESSURE (FORM) DRAG ONLY. It does NOT include viscous
skin-friction drag, which is a real, non-negligible fraction of total drag at this
project's laminar Reynolds range (Plan Section 6, Re ~ 1e4-1e5). Empirically it is
noticeably smaller than SU2's total CD, consistent with that missing term -- this is
an expected physical limitation, not a bug, and must be labeled as such (e.g.
"CD (pressure only)") everywhere it's shown, never presented as equivalent to SU2's
total CD.

surf_xy must be ordered around the closed loop matching
aeropinn.geometry.naca.naca4_boundary_loop / sdf._naca4_boundary_loop_torch's
convention: lower surface LE->TE, then upper surface TE->LE.
"""
import math

import numpy as np
import torch


def integrate_cl_cd_torch(surf_xy: torch.Tensor, cp: torch.Tensor, alpha_deg) -> "tuple[torch.Tensor, torch.Tensor]":
    """Differentiable. surf_xy: (N, 2) closed-loop-ordered. cp: (N,). alpha_deg: scalar or 0-d tensor.
    Chord assumed = 1 (nondimensional convention used throughout this project).
    """
    x, y = surf_xy[:, 0], surf_xy[:, 1]
    dx = torch.roll(x, -1) - x
    dy = torch.roll(y, -1) - y
    cp_avg = 0.5 * (cp + torch.roll(cp, -1))

    cn = (cp_avg * dx).sum()
    ca = -(cp_avg * dy).sum()

    if not torch.is_tensor(alpha_deg):
        alpha_deg = torch.as_tensor(alpha_deg, dtype=surf_xy.dtype, device=surf_xy.device)
    alpha_rad = alpha_deg * (math.pi / 180.0)

    cl = cn * torch.cos(alpha_rad) - ca * torch.sin(alpha_rad)
    cd = cn * torch.sin(alpha_rad) + ca * torch.cos(alpha_rad)
    return cl, cd


def integrate_cl_cd_numpy(surf_xy: np.ndarray, cp: np.ndarray, alpha_deg: float):
    """Non-differentiable numpy mirror, for quick offline sanity checks against SU2."""
    x, y = surf_xy[:, 0], surf_xy[:, 1]
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    cp_avg = 0.5 * (cp + np.roll(cp, -1))

    cn = (cp_avg * dx).sum()
    ca = -(cp_avg * dy).sum()

    alpha_rad = math.radians(alpha_deg)
    cl = cn * math.cos(alpha_rad) - ca * math.sin(alpha_rad)
    cd = cn * math.sin(alpha_rad) + ca * math.cos(alpha_rad)
    return float(cl), float(cd)
