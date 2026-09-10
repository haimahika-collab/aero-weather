"""Brute-force grid search over (m, p, t) — a cheap, easily-independently-checkable
comparison for the gradient-based optimizer (Plan Section 19). Legitimate here
specifically because the design space is low-dimensional (<=3 free dims at fixed
alpha/Re): even a few thousand candidates evaluate in well under a second, batched
through the model, no gradients needed.
"""
from dataclasses import dataclass

import numpy as np
import torch

from aeropinn.geometry.sdf import _naca4_boundary_loop_torch
from aeropinn.physics.forces import integrate_cl_cd_numpy
from aeropinn.optimization.gradient_design import DEFAULT_BOUNDS


@dataclass
class GridResult:
    best_design: dict
    best_cl_over_cd: float
    all_results: list  # [{m, p, t, cl, cd, cl_over_cd}, ...]


def optimize_grid(model, device, start_geom: dict, reynolds: float,
                   resolution: int = 12, bounds: dict = None, t_min: float = 0.09) -> GridResult:
    bounds = dict(DEFAULT_BOUNDS if bounds is None else bounds)
    bounds["t"] = (max(bounds["t"][0], t_min), bounds["t"][1])
    alpha = float(start_geom["alpha_deg"])

    m_vals = np.linspace(*bounds["m"], resolution)
    p_vals = np.linspace(*bounds["p"], resolution)
    t_vals = np.linspace(*bounds["t"], resolution)
    Mg, Pg, Tg = np.meshgrid(m_vals, p_vals, t_vals, indexing="ij")
    candidates = np.stack([Mg.ravel(), Pg.ravel(), Tg.ravel()], axis=-1)  # (N, 3)
    n_cand = candidates.shape[0]

    n_points = 60
    with torch.no_grad():
        m_t = torch.tensor(candidates[:, 0], dtype=torch.float32, device=device)
        p_t = torch.tensor(candidates[:, 1], dtype=torch.float32, device=device)
        t_t = torch.tensor(candidates[:, 2], dtype=torch.float32, device=device)
        loops = _naca4_boundary_loop_torch(m_t, p_t, t_t, n_points=n_points)  # (N, M, 2)

        flat_xy = loops.reshape(-1, 2)
        n_per = loops.shape[1]
        m_rep = m_t.repeat_interleave(n_per).unsqueeze(-1)
        p_rep = p_t.repeat_interleave(n_per).unsqueeze(-1)
        t_rep = t_t.repeat_interleave(n_per).unsqueeze(-1)
        sdf0 = torch.zeros(flat_xy.shape[0], 1, device=device)
        alpha_col = torch.full((flat_xy.shape[0], 1), alpha, device=device)
        re_col = torch.full((flat_xy.shape[0], 1), float(reynolds), device=device)
        cond = torch.cat([sdf0, m_rep, p_rep, t_rep, alpha_col, re_col], dim=-1)

        _, _, p_star = model(flat_xy[:, 0:1], flat_xy[:, 1:2], cond)
        cp_flat = (2.0 * p_star).squeeze(-1).cpu().numpy().reshape(n_cand, n_per)

    loops_np = loops.cpu().numpy()
    results = []
    for i in range(n_cand):
        cl, cd = integrate_cl_cd_numpy(loops_np[i], cp_flat[i], alpha)
        cl_over_cd = cl / abs(cd) if abs(cd) > 1e-6 else float("nan")
        results.append({
            "m": float(candidates[i, 0]), "p": float(candidates[i, 1]), "t": float(candidates[i, 2]),
            "cl": cl, "cd": cd, "cl_over_cd": cl_over_cd,
        })

    valid = [r for r in results if np.isfinite(r["cl_over_cd"])]
    best = max(valid, key=lambda r: r["cl_over_cd"])
    return GridResult(
        best_design={"m": best["m"], "p": best["p"], "t": best["t"], "alpha_deg": alpha},
        best_cl_over_cd=best["cl_over_cd"],
        all_results=results,
    )
