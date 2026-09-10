"""SU2 case automation for the laminar NACA 4-digit sweep (Plan Section 12).

Wraps `mikfoil` (https://pypi.org/project/mikfoil/) for NACA-to-SU2 C-grid mesh
generation, combined with our OWN laminar incompressible SU2 config template
(configs/su2_templates/laminar_incompressible.cfg) rather than mikfoil's bundled
default, which is turbulent RANS/SST — appropriate for AirfRANS-regime Re but not
for this project's laminar training regime (Plan Section 6).

Dev/offline-only: never imported by the deployed app.py (Plan Section 24).

RESOLVED — two real config bugs found and fixed during Week 1 smoke testing (see
test_su2_runner.py::test_naca0012_lift_sign_and_symmetry, a permanent regression
test for both):
  1. The config used FREESTREAM_VELOCITY, which SU2's INC_* solver family silently
     ignores (it's a compressible-solver keyword) — the actual freestream for
     incompressible solvers must be set via INC_VELOCITY_INIT. Without it, AoA had
     zero effect on the flow at all, giving CL ~= 0 for a symmetric airfoil at
     nonzero AoA regardless of mesh resolution.
  2. Once that was fixed, lift came out with the wrong sign — root cause: AoA is
     applied by rotating the inflow (not the mesh/geometry), and positive AoA in
     that convention means the apparent inflow has a NEGATIVE y-component in the
     airfoil-fixed frame, not positive. Fixed in the template's vel_y formula.
Verified at default mesh resolution: NACA0012 gives CL(+2deg)=+0.131, CL(-2deg)=
-0.112, CL(0deg)~=0.00001 — correct sign, correct antisymmetry, correct zero.

STILL OPEN: this is sign/direction correctness, not full physical-accuracy
validation. The Plan Section 12 step 5 mesh-independence check (CL/CD stable
across >=3 mesh densities) has NOT been run yet — no sweep-scale data from this
module should be treated as final ground truth until that check passes.
"""
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from mikfoil import Case, MeshConfig

_LAMINAR_CONFIG_TEMPLATE = str(
    Path(__file__).resolve().parent.parent / "configs" / "su2_templates" / "laminar_incompressible.cfg"
)

# Fallback install location used this session (Plan/README: `~/SU2`). ambient shell
# PATH only carries this when a login shell sources ~/.zprofile — NOT guaranteed for
# every subprocess (this bit us repeatedly during development: manual `export
# PATH=...` was needed per command). su2_is_available()/ensure_su2_on_path() below
# make discovery robust regardless of how the calling process was started.
_SU2_FALLBACK_BIN = Path.home() / "SU2" / "bin"


def su2_is_available() -> bool:
    return shutil.which("SU2_CFD") is not None or (_SU2_FALLBACK_BIN / "SU2_CFD").exists()


def ensure_su2_on_path() -> None:
    """Idempotent: prepend the known SU2 install dir to PATH/SU2_RUN/PYTHONPATH for
    this process if SU2_CFD isn't already discoverable via ambient PATH."""
    if shutil.which("SU2_CFD") is not None:
        return
    if not (_SU2_FALLBACK_BIN / "SU2_CFD").exists():
        return
    su2_bin = str(_SU2_FALLBACK_BIN)
    os.environ["PATH"] = su2_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ["SU2_RUN"] = su2_bin
    os.environ["PYTHONPATH"] = su2_bin + os.pathsep + os.environ.get("PYTHONPATH", "")


@dataclass
class GeometryCondition:
    """One (geometry, operating-condition) combination in the sweep (Plan Section 12 step 3).

    Either naca_digits (an integer NACA 4-digit code, e.g. "0012") or airfoil_path
    (a Selig .dat coordinate file, for arbitrary continuous (m,p,t) that doesn't land
    on an integer code — e.g. an inverse-design optimizer's output) must be set.
    """

    alpha_deg: float
    reynolds: float
    naca_digits: Optional[str] = None
    airfoil_path: Optional[Union[str, Path]] = None
    mach: float = 0.1  # low-Mach proxy for "incompressible", per Plan Section 6's M_inf <= 0.3 bound

    def __post_init__(self):
        if not self.naca_digits and not self.airfoil_path:
            raise ValueError("GeometryCondition requires either naca_digits or airfoil_path")


def run_case(
    condition: GeometryCondition,
    project_dir: str,
    mesh_fineness: float = 1.0,
    skip_xfoil_validation: bool = True,
) -> "Case":
    """Generate mesh + run one laminar incompressible SU2 case. Returns the mikfoil Case
    (case.results holds cl/cd/cm; case.su2_dir holds history.csv/volume.vtu/surface.vtu).

    skip_xfoil_validation: mikfoil's execute_case() calls XFOIL for a cross-check by
    default (Plan Section 12 step 7's intended role for XFOIL), which requires XFOIL
    to be installed separately. Defaults to True so this runs standalone; set False
    once XFOIL is available and wired in.
    """
    ensure_su2_on_path()
    import mikfoil.main as mikfoil_main

    airfoil = Path(condition.airfoil_path) if condition.airfoil_path else condition.naca_digits
    case = Case(
        airfoil=airfoil,
        aoa=condition.alpha_deg,
        mach=condition.mach,
        reynolds=condition.reynolds,
        mesh_config=MeshConfig(fineness=mesh_fineness),
        su2_config=_LAMINAR_CONFIG_TEMPLATE,
        project_dir=project_dir,
    )

    if skip_xfoil_validation:
        original_validate = mikfoil_main.perform_validation
        mikfoil_main.perform_validation = lambda _case: None
        try:
            mikfoil_main.execute_case(case)
        finally:
            mikfoil_main.perform_validation = original_validate
    else:
        mikfoil_main.execute_case(case)

    return case
