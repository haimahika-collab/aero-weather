"""SU2 case automation for the laminar NACA 4-digit sweep (Plan Section 12).

Wraps `mikfoil` (https://pypi.org/project/mikfoil/) for NACA-to-SU2 C-grid mesh
generation, combined with our OWN laminar incompressible SU2 config template
(configs/su2_templates/laminar_incompressible.cfg) rather than mikfoil's bundled
default, which is turbulent RANS/SST — appropriate for AirfRANS-regime Re but not
for this project's laminar training regime (Plan Section 6).

Dev/offline-only: never imported by the deployed app.py (Plan Section 24).

IMPORTANT — not yet trustworthy ground truth: an initial coarse-mesh smoke test
(NACA0012, AoA=2 deg, Re=5e4, fineness=0.3) ran to SU2's own convergence criteria
in ~29s but produced CL ~= 0, which is physically suspicious for a symmetric
airfoil at nonzero AoA (thin-airfoil theory suggests CL ~ 0.2 in this regime).
This has NOT been root-caused yet (candidates: mesh too coarse at this fineness,
genuine low-Re separation, or a config/reference-value issue) and no case run
through this module should be treated as valid data until the Plan Section 12
step 5 mesh-independence check is actually performed and passes.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mikfoil import Case, MeshConfig

_LAMINAR_CONFIG_TEMPLATE = str(
    Path(__file__).resolve().parent.parent / "configs" / "su2_templates" / "laminar_incompressible.cfg"
)


@dataclass
class GeometryCondition:
    """One (geometry, operating-condition) combination in the sweep (Plan Section 12 step 3)."""

    naca_digits: str  # e.g. "0012", "4412" — matches mikfoil's create_airfoil string format
    alpha_deg: float
    reynolds: float
    mach: float = 0.1  # low-Mach proxy for "incompressible", per Plan Section 6's M_inf <= 0.3 bound


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
    import mikfoil.main as mikfoil_main

    case = Case(
        airfoil=condition.naca_digits,
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
