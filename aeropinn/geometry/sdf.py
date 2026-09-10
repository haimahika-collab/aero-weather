"""Signed-distance-function (SDF) features for NACA 4-digit airfoils.

Exact closed-form SDFs don't exist for this curve family, so we use the standard
practical approximation: densely sample the airfoil boundary as a closed polygon
and take the (signed) nearest-point distance. This is differentiable w.r.t. the
query point and the geometry parameters through the boundary-point coordinates
(not through the argmin/point-in-polygon index selection, which is standard and
accepted practice for using SDF as a network *input feature*).

Convention: negative inside the airfoil, positive outside, zero on the surface.

Provides a numpy reference implementation (used for meshing/plotting/tests) and a
batched PyTorch implementation (used at training time, where every collocation
point in a batch may belong to a different training geometry).
"""
import numpy as np
import torch

from aeropinn.geometry.naca import naca4_boundary_loop


def _point_in_polygon_numpy(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized crossing-number point-in-polygon test.

    points: (N, 2). polygon: (M, 2), implicitly closed (edge from last to first vertex).
    Returns a boolean (N,) array, True where the point is inside the polygon.
    """
    x, y = points[:, 0], points[:, 1]
    px, py = polygon[:, 0], polygon[:, 1]
    n = polygon.shape[0]
    inside = np.zeros(points.shape[0], dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        denom = (yj - yi) if (yj - yi) != 0 else 1e-300
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / denom + xi)
        inside ^= intersect
        j = i
    return inside


def signed_distance_numpy(query_xy: np.ndarray, m: float, p: float, t: float, n_samples: int = 400) -> np.ndarray:
    """SDF for a single geometry (scalar m, p, t) evaluated at many query points.

    query_xy: (N, 2). Returns (N,) signed distances.
    """
    query_xy = np.asarray(query_xy, dtype=np.float64)
    polygon = naca4_boundary_loop(m, p, t, n_points=n_samples)
    # unsigned nearest-point distance
    diff = query_xy[:, None, :] - polygon[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1)).min(axis=1)
    inside = _point_in_polygon_numpy(query_xy, polygon)
    sign = np.where(inside, -1.0, 1.0)
    return sign * dist


def _naca4_boundary_loop_torch(m: torch.Tensor, p: torch.Tensor, t: torch.Tensor, n_points: int) -> torch.Tensor:
    """Batched torch boundary polygon: one closed loop per (m, p, t) triple.

    m, p, t: (N,) tensors. Returns (N, 2*n_points - 1, 2).
    """
    device, dtype = m.device, m.dtype
    beta = torch.linspace(0.0, torch.pi, n_points, device=device, dtype=dtype)
    x = 0.5 * (1.0 - torch.cos(beta))  # (n_points,)

    a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1015
    t_ = t.unsqueeze(-1)  # (N, 1)
    yt = 5.0 * t_ * (a0 * torch.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)  # (N, n_points)

    m_ = m.unsqueeze(-1)
    p_safe = torch.where(p.abs() < 1e-8, torch.ones_like(p), p).unsqueeze(-1)
    front = (x.unsqueeze(0) < p_safe)  # (N, n_points)

    yc_front = (m_ / p_safe**2) * (2 * p_safe * x - x**2)
    yc_back = (m_ / (1 - p_safe) ** 2) * ((1 - 2 * p_safe) + 2 * p_safe * x - x**2)
    yc = torch.where(front, yc_front, yc_back)
    yc = torch.where((m.unsqueeze(-1) == 0), torch.zeros_like(yc), yc)

    dyc_front = (2 * m_ / p_safe**2) * (p_safe - x)
    dyc_back = (2 * m_ / (1 - p_safe) ** 2) * (p_safe - x)
    dyc = torch.where(front, dyc_front, dyc_back)
    dyc = torch.where((m.unsqueeze(-1) == 0), torch.zeros_like(dyc), dyc)

    theta = torch.atan(dyc)
    x_b = x.unsqueeze(0).expand_as(yt)
    xu = x_b - yt * torch.sin(theta)
    yu = yc + yt * torch.cos(theta)
    xl = x_b + yt * torch.sin(theta)
    yl = yc - yt * torch.cos(theta)

    upper = torch.stack([xu, yu], dim=-1)  # (N, n_points, 2)
    lower = torch.stack([xl, yl], dim=-1)
    upper_rev = torch.flip(upper, dims=[1])[:, 1:, :]
    return torch.cat([lower, upper_rev], dim=1)  # (N, 2*n_points - 1, 2)


def naca4_boundary_loop_torch(m, p, t, n_points: int = 150) -> torch.Tensor:
    """Public single-geometry wrapper around _naca4_boundary_loop_torch: differentiable
    w.r.t. scalar (0-d or 1-element) m, p, t. Returns (2*n_points-1, 2), ordered lower
    surface LE->TE then upper surface TE->LE (matches physics/forces.py's convention).
    Used by aeropinn/optimization/{gradient_design,grid_search}.py, where the design
    variables m/p/t need gradients to flow through the geometry itself, not just the
    network's output.
    """
    m1 = torch.as_tensor(m).reshape(1).float()
    p1 = torch.as_tensor(p).reshape(1).float()
    t1 = torch.as_tensor(t).reshape(1).float()
    loop = _naca4_boundary_loop_torch(m1, p1, t1, n_points=n_points)  # (1, M, 2)
    return loop.squeeze(0)


def _point_in_polygon_torch(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Batched crossing-number test: one polygon per point.

    points: (N, 2). polygon: (N, M, 2). Returns bool (N,).
    """
    x, y = points[:, 0], points[:, 1]
    px, py = polygon[..., 0], polygon[..., 1]  # (N, M)
    m = polygon.shape[1]
    inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    j = m - 1
    for i in range(m):
        xi, yi = px[:, i], py[:, i]
        xj, yj = px[:, j], py[:, j]
        denom = yj - yi
        denom = torch.where(denom.abs() < 1e-12, denom + 1e-12, denom)
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / denom + xi)
        inside = inside ^ intersect
        j = i
    return inside


def signed_distance_torch(
    query_xy: torch.Tensor,
    m: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    n_samples: int = 200,
) -> torch.Tensor:
    """Batched SDF: query_xy[i] evaluated against its own geometry (m[i], p[i], t[i]).

    query_xy: (N, 2). m, p, t: (N,) — broadcastable scalars are accepted and expanded.
    Returns (N,) signed distances (negative inside, positive outside).

    Note: this computes one boundary polygon per query point, which is simple and
    correct but not the most compute-efficient option when many points share the
    same geometry (the common case for a training batch). Callers that group
    collocation points by geometry can call this once per distinct geometry instead.
    """
    n = query_xy.shape[0]
    m = torch.as_tensor(m, device=query_xy.device, dtype=query_xy.dtype).expand(n)
    p = torch.as_tensor(p, device=query_xy.device, dtype=query_xy.dtype).expand(n)
    t = torch.as_tensor(t, device=query_xy.device, dtype=query_xy.dtype).expand(n)

    polygon = _naca4_boundary_loop_torch(m, p, t, n_points=n_samples)  # (N, M, 2)
    diff = query_xy.unsqueeze(1) - polygon  # (N, M, 2)
    dist = torch.linalg.norm(diff, dim=-1).min(dim=1).values  # (N,)
    inside = _point_in_polygon_torch(query_xy, polygon)
    sign = torch.where(inside, -torch.ones_like(dist), torch.ones_like(dist))
    return sign * dist
