"""Section 20 hard rule: an optimizer-proposed geometry is never reported as an
improvement until it's re-evaluated by a solver OUTSIDE the optimization loop.

This module NEVER imports aeropinn.models.pinn or loads any checkpoint — only the
SU2 wrapper. That separation is enforced by omission, not by a runtime check: if you
find yourself importing the model here, stop, that defeats the point of the module.

Dev/offline-only in the sense that it needs a real SU2 install (only this dev
machine has one, not the Railway deploy target) -- but importing this module itself
is safe from any context; su2_is_available() lets callers degrade gracefully.
"""
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aeropinn.solvers.su2_runner import GeometryCondition, ensure_su2_on_path, run_case, su2_is_available


@dataclass
class ValidationResult:
    candidate: dict
    surrogate_predicted: dict
    su2_reference: Optional[dict]
    agreement_pct: Optional[float]
    within_tolerance: Optional[bool]
    status: str  # "validated" | "validated_disagreement" | "unavailable"
    elapsed_s: Optional[float] = None


def _write_candidate_dat(m: float, p: float, t: float, out_dir: Path) -> Path:
    from mikfoil.geometry.geometry_types import AirfoilGeometry
    from mikfoil.geometry.selig_format import export_selig_dat
    from aeropinn.geometry.naca import naca4_surface

    upper, lower = naca4_surface(m, p, t, n_points=150)
    geom = AirfoilGeometry(upper_surface=upper, lower_surface=lower, chord=1.0)
    dat_path = out_dir / "candidate.dat"
    export_selig_dat(geom, "candidate", dat_path)
    return dat_path


def validate_candidate(
    candidate: dict, surrogate_predicted: dict, reynolds: float,
    tolerance_pct: float = 15.0, mesh_fineness: float = 1.0,
) -> ValidationResult:
    """candidate: {m, p, t, alpha_deg}. surrogate_predicted: {cl, cd_pressure, cl_over_cd}
    from the optimizer's own (never-independent) forward pass -- passed in, not
    recomputed here, since this module must never touch the model.
    """
    import time

    ensure_su2_on_path()
    if not su2_is_available():
        return ValidationResult(
            candidate=candidate, surrogate_predicted=surrogate_predicted,
            su2_reference=None, agreement_pct=None, within_tolerance=None, status="unavailable",
        )

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        dat_path = _write_candidate_dat(candidate["m"], candidate["p"], candidate["t"], Path(tmp))
        condition = GeometryCondition(
            airfoil_path=dat_path, alpha_deg=candidate["alpha_deg"], reynolds=reynolds, mach=0.1,
        )
        case = run_case(condition, project_dir=tmp, mesh_fineness=mesh_fineness)
    elapsed = time.time() - t0

    if not case.results.success:
        return ValidationResult(
            candidate=candidate, surrogate_predicted=surrogate_predicted,
            su2_reference=None, agreement_pct=None, within_tolerance=None, status="unavailable",
            elapsed_s=elapsed,
        )

    su2_cl = case.results.cl
    su2_cd = case.results.cd
    su2_cl_over_cd = su2_cl / su2_cd if abs(su2_cd) > 1e-9 else float("nan")
    su2_ref = {"cl": su2_cl, "cd": su2_cd, "cl_over_cd": su2_cl_over_cd}

    pred_ratio = surrogate_predicted["cl_over_cd"]
    agreement_pct = 100.0 * (1.0 - abs(pred_ratio - su2_cl_over_cd) / (abs(su2_cl_over_cd) + 1e-9))
    within = abs(pred_ratio - su2_cl_over_cd) <= (tolerance_pct / 100.0) * abs(su2_cl_over_cd)
    status = "validated" if within else "validated_disagreement"

    return ValidationResult(
        candidate=candidate, surrogate_predicted=surrogate_predicted, su2_reference=su2_ref,
        agreement_pct=agreement_pct, within_tolerance=within, status=status, elapsed_s=elapsed,
    )
