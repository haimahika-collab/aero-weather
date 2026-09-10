"""Pure-compute logic for the AeroPINN tab — no matplotlib, no Gradio. Every view in
build_tab.py calls through here so there is exactly one code path per computation
(prediction, domain-validity check, held-out case lookup), not one per view.

Deploy-path module (Plan Section 24): only imports models/geometry/physics/evaluation
code, never solvers/training (dev-only, SU2/mikfoil/DeepXDE).
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = REPO_ROOT / "aeropinn" / "experiments" / "checkpoints"

VERSION = "pinn_v1"  # tied to the checkpoint filename actually loaded, not a marketing string

_cache = {}


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_state() -> dict:
    """Lazily load the trained model + OOD calibration + held-out demo cases. Cached
    after first call so this tab never slows down app.py's startup or the other tabs.
    """
    if _cache:
        return _cache
    from aeropinn.models.pinn import GeometryConditionedPINN

    dev = device()
    model = GeometryConditionedPINN().to(dev)
    ckpt_path = CHECKPOINT_DIR / "pinn_v1.pt"
    if not ckpt_path.exists():
        _cache["ready"] = False
        return _cache
    ckpt = torch.load(ckpt_path, map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _cache["model"] = model
    _cache["device"] = dev
    _cache["ready"] = True
    _cache["version"] = VERSION

    ood_path = CHECKPOINT_DIR / "ood_mahalanobis.npz"
    if ood_path.exists():
        from aeropinn.evaluation.ood import MahalanobisOOD
        _cache["ood"] = MahalanobisOOD.load(ood_path)

    demo_path = CHECKPOINT_DIR / "held_out_demo_cases.npz"
    if demo_path.exists():
        d = np.load(demo_path, allow_pickle=True)
        _cache["demo_cases"] = list(d["cases"])

    # Physics Ablation (B2 data-only) model, if it's been trained -- optional, tab
    # degrades gracefully (Physics Ablation view shows "not yet trained") if absent.
    b2_path = CHECKPOINT_DIR / "pinn_v1_b2_dataonly.pt"
    if b2_path.exists():
        model_b2 = GeometryConditionedPINN().to(dev)
        ckpt_b2 = torch.load(b2_path, map_location=dev)
        model_b2.load_state_dict(ckpt_b2["model_state"])
        model_b2.eval()
        _cache["model_b2"] = model_b2

    b2_eval_path = CHECKPOINT_DIR / "held_out_demo_cases_b2.npz"
    if b2_eval_path.exists():
        d = np.load(b2_eval_path, allow_pickle=True)
        _cache["demo_cases_b2"] = list(d["cases"])

    geoms_path = REPO_ROOT / "aeropinn" / "data" / "processed" / "sweep_v1" / "geometries.json"
    if geoms_path.exists():
        import json
        with open(geoms_path) as f:
            _cache["geometries"] = json.load(f)

    return _cache


@dataclass
class FieldResult:
    X: np.ndarray
    Y: np.ndarray
    speed: np.ndarray
    cp_field: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    surf_xy: np.ndarray       # (Ns, 2) surface points used for CL/CD integration
    surf_cp: np.ndarray       # (Ns,) predicted Cp at those surface points
    latency_ms: float
    m: float = 0.0
    p: float = 0.0
    t: float = 0.0
    alpha_deg: float = 0.0
    reynolds: float = 0.0


@dataclass
class DomainStatus:
    available: bool
    is_ood: bool = False
    distance: float = 0.0
    threshold: float = 0.0

    @property
    def text(self) -> str:
        if not self.available:
            return "OOD check unavailable (not calibrated)"
        if self.is_ood:
            return f"OUTSIDE VALIDATED DESIGN ENVELOPE (distance {self.distance:.2f} > {self.threshold:.2f})"
        return f"WITHIN VALIDATED DOMAIN (distance {self.distance:.2f} ≤ {self.threshold:.2f})"


def predict_field(m: float, p: float, t: float, alpha: float, reynolds: float,
                   model_key: str = "model") -> FieldResult:
    """Run the model over a field grid AND a dedicated surface point set (used both
    for the Cp-curve view and for CL/CD integration in physics/forces.py).
    """
    from aeropinn.geometry.naca import naca4_surface
    from aeropinn.geometry.sdf import signed_distance_torch

    state = load_state()
    model, dev = state[model_key], state["device"]

    xs = np.linspace(-0.8, 1.8, 90)
    ys = np.linspace(-0.9, 0.9, 60)
    X, Y = np.meshgrid(xs, ys)
    xy_flat = np.stack([X.ravel(), Y.ravel()], axis=-1)

    upper, lower = naca4_surface(m, p, t, n_points=150)
    # Surface points nudged a hair outside the body (SDF slightly > 0) so the model
    # is queried just off the (possibly non-differentiable) surface itself.
    surf_xy = np.concatenate([lower, upper[::-1]], axis=0)

    all_xy = np.concatenate([xy_flat, surf_xy], axis=0)
    xy_t = torch.tensor(all_xy, dtype=torch.float32, device=dev)
    n = xy_t.shape[0]
    m_t = torch.full((n,), m, dtype=torch.float32, device=dev)
    p_t = torch.full((n,), p, dtype=torch.float32, device=dev)
    t_t = torch.full((n,), t, dtype=torch.float32, device=dev)

    t0 = time.time()
    with torch.no_grad():
        sdf = signed_distance_torch(xy_t, m_t, p_t, t_t, n_samples=150).unsqueeze(-1)
        cond = torch.cat([
            sdf, m_t.unsqueeze(-1), p_t.unsqueeze(-1), t_t.unsqueeze(-1),
            torch.full((n, 1), alpha, device=dev), torch.full((n, 1), reynolds, device=dev),
        ], dim=-1)
        u, v, p_star = model(xy_t[:, 0:1], xy_t[:, 1:2], cond)
    latency_ms = (time.time() - t0) * 1000.0

    n_grid = xy_flat.shape[0]
    speed_grid = torch.sqrt(u[:n_grid] ** 2 + v[:n_grid] ** 2).cpu().numpy().reshape(X.shape)
    cp_grid = (2.0 * p_star[:n_grid]).cpu().numpy().reshape(X.shape)
    inside = sdf[:n_grid].cpu().numpy().reshape(X.shape) < 0
    speed_grid[inside] = np.nan
    cp_grid[inside] = np.nan

    surf_cp = (2.0 * p_star[n_grid:]).squeeze(-1).cpu().numpy()

    return FieldResult(
        X=X, Y=Y, speed=speed_grid, cp_field=cp_grid, upper=upper, lower=lower,
        surf_xy=surf_xy, surf_cp=surf_cp, latency_ms=latency_ms,
        m=m, p=p, t=t, alpha_deg=alpha, reynolds=reynolds,
    )


def field_forces(result: FieldResult) -> dict:
    """CL and pressure-only CD from the model's own surface Cp (physics/forces.py,
    validated to ~5% mean CL relative error against SU2 on the 36 held-out cases).
    CD here is pressure/form drag only -- it does not include viscous skin-friction
    drag, so it will read lower than a true total CD; callers must label it as such.
    """
    from aeropinn.physics.forces import integrate_cl_cd_numpy
    cl, cd_pressure = integrate_cl_cd_numpy(result.surf_xy, result.surf_cp, result.alpha_deg)
    cl_over_cd = cl / cd_pressure if abs(cd_pressure) > 1e-9 else float("nan")
    return {"cl": cl, "cd_pressure": cd_pressure, "cl_over_cd": cl_over_cd}


def domain_status(m: float, p: float, t: float, alpha: float, reynolds: float) -> DomainStatus:
    state = load_state()
    if "ood" not in state:
        return DomainStatus(available=False)
    query = np.array([m, p, t, alpha, reynolds])
    dist = state["ood"].distance(query)
    is_ood = state["ood"].is_out_of_distribution(query)
    return DomainStatus(available=True, is_ood=is_ood, distance=dist, threshold=state["ood"].threshold)


def case_label(c) -> str:
    return f"m={c['m']:.3f}, p={c['p']:.2f}, t={c['t']:.3f}, AoA={c['alpha_deg']:.1f}, Re={c['reynolds']:.0e}"


def case_labels(model_key: str = "demo_cases") -> list:
    state = load_state()
    return [case_label(c) for c in state.get(model_key, [])]


def held_out_case(label: str, model_key: str = "demo_cases") -> Optional[dict]:
    state = load_state()
    cases = state.get(model_key, [])
    labels = [case_label(c) for c in cases]
    if label not in labels:
        return None
    return cases[labels.index(label)]


def load_held_out_eval(suffix: str = "") -> Optional[dict]:
    """suffix='' -> held_out_eval.json (physics-guided). suffix='_b2' -> the
    data-only ablation model's eval, if it has been trained."""
    import json
    path = CHECKPOINT_DIR / f"held_out_eval{suffix}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def naca_designation(m: float, p: float, t: float) -> str:
    d1 = round(m * 100)
    d2 = round(p * 10)
    d34 = round(t * 100)
    return f"NACA {d1:01d}{d2:01d}{d34:02d}"
