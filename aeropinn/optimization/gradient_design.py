"""Gradient-based inverse airfoil design (Plan Section 19): maximize CL/CD by
backpropagating through the frozen, already-trained GeometryConditionedPINN and the
differentiable Cp-panel integration in physics/forces.py.

Box constraints (m, p, t, optionally alpha) are enforced via a sigmoid
reparameterization — z (unconstrained) -> lo + (hi-lo)*sigmoid(z) — per PLAN.md
Section 19's explicit "clipping/sigmoid reparam, not penalty terms" requirement, so
every iterate the optimizer ever produces is guaranteed inside the validated design
envelope, not just penalized for leaving it.

HARD RULE (Plan Section 20): nothing in this module, or any caller of it, may present
the result as a validated improvement. That requires a real, independent solver run
via aeropinn/evaluation/independent_validation.py -- this module only ever reports
the surrogate's OWN prediction for the candidate it found.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch

from aeropinn.geometry.sdf import naca4_boundary_loop_torch
from aeropinn.physics.forces import integrate_cl_cd_torch

DEFAULT_BOUNDS = {
    "m": (0.0, 0.06),
    "p": (0.10, 0.60),
    "t": (0.08, 0.20),
    "alpha_deg": (-4.0, 8.0),
}


@dataclass
class OptimizationTrace:
    history: list                # [{iter, m, p, t, alpha_deg, cl, cd, cl_over_cd}, ...]
    start_design: dict
    final_design: dict
    converged: bool
    method: str = "gradient"


def _sigmoid_param(value: float, lo: float, hi: float) -> torch.Tensor:
    """Inverse-sigmoid init so the unconstrained z starts exactly at `value`."""
    frac = max(min((value - lo) / (hi - lo), 1 - 1e-5), 1e-5)
    z0 = torch.log(torch.tensor(frac / (1 - frac)))
    return z0.clone().requires_grad_(True)


def _from_z(z: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(z)


def optimize_gradient(
    model, device, start_geom: dict, reynolds: float,
    objective: str = "cl_over_cd", optimize_alpha: bool = False,
    bounds: dict = None, t_min: float = 0.09,
    n_iters: int = 200, lr: float = 0.02,
) -> OptimizationTrace:
    if objective != "cl_over_cd":
        raise ValueError("Only 'cl_over_cd' is supported in this phase.")
    bounds = dict(DEFAULT_BOUNDS if bounds is None else bounds)
    bounds["t"] = (max(bounds["t"][0], t_min), bounds["t"][1])

    z_m = _sigmoid_param(start_geom["m"], *bounds["m"])
    z_p = _sigmoid_param(start_geom["p"], *bounds["p"])
    z_t = _sigmoid_param(start_geom["t"], *bounds["t"])
    params = [z_m, z_p, z_t]
    if optimize_alpha:
        z_alpha = _sigmoid_param(start_geom["alpha_deg"], *bounds["alpha_deg"])
        params.append(z_alpha)
    else:
        z_alpha = None

    fixed_alpha = torch.tensor(float(start_geom["alpha_deg"]))
    reynolds_t = torch.tensor(float(reynolds))

    optimizer = torch.optim.Adam(params, lr=lr)
    history = []

    for i in range(n_iters):
        optimizer.zero_grad()
        m = _from_z(z_m, *bounds["m"])
        p = _from_z(z_p, *bounds["p"])
        t = _from_z(z_t, *bounds["t"])
        alpha = _from_z(z_alpha, *bounds["alpha_deg"]) if optimize_alpha else fixed_alpha

        surf_xy = naca4_boundary_loop_torch(m, p, t, n_points=80).to(device)
        n = surf_xy.shape[0]
        sdf0 = torch.zeros(n, 1, device=device)
        cond = torch.cat([
            sdf0,
            m.to(device).expand(n).unsqueeze(-1),
            p.to(device).expand(n).unsqueeze(-1),
            t.to(device).expand(n).unsqueeze(-1),
            alpha.to(device).expand(n).unsqueeze(-1),
            reynolds_t.to(device).expand(n).unsqueeze(-1),
        ], dim=-1)
        _, _, p_star = model(surf_xy[:, 0:1], surf_xy[:, 1:2], cond)
        cp = 2.0 * p_star.squeeze(-1)

        cl, cd = integrate_cl_cd_torch(surf_xy, cp, alpha.to(device))
        # Maximize CL/CD == minimize -CL/CD; guard near-zero CD (early iterates can
        # start close to a symmetric, near-zero-lift design).
        loss = -cl / (cd.abs() + 1e-3)
        loss.backward()
        optimizer.step()

        cl_over_cd = (cl / (cd.abs() + 1e-9)).item()
        history.append({
            "iter": i, "m": m.item(), "p": p.item(), "t": t.item(), "alpha_deg": alpha.item(),
            "cl": cl.item(), "cd": cd.item(), "cl_over_cd": cl_over_cd,
        })

    final = history[-1]
    final_design = {"m": final["m"], "p": final["p"], "t": final["t"], "alpha_deg": final["alpha_deg"]}
    converged = abs(history[-1]["cl_over_cd"] - history[max(0, n_iters - 20)]["cl_over_cd"]) < 0.05

    return OptimizationTrace(
        history=history,
        start_design={"m": start_geom["m"], "p": start_geom["p"], "t": start_geom["t"],
                      "alpha_deg": start_geom["alpha_deg"]},
        final_design=final_design,
        converged=converged,
    )
