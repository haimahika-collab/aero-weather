"""Integration smoke test for the SU2 wrapper — requires a real SU2 install.

Skipped automatically if SU2_CFD isn't on PATH (e.g. in CI, or before Week 2's CFD
setup step), so it never blocks the fast unit-test suite. When it does run, it
exercises the real toolchain (mikfoil meshing -> our laminar config -> SU2_CFD),
not a mock, per the plan's "verify against the real solver" requirement.
"""
import shutil

import pytest

su2_available = shutil.which("SU2_CFD") is not None

pytestmark = pytest.mark.skipif(not su2_available, reason="SU2_CFD not found on PATH")


def test_laminar_naca0012_smoke_run(tmp_path):
    from aeropinn.solvers.su2_runner import GeometryCondition, run_case

    condition = GeometryCondition(naca_digits="0012", alpha_deg=2.0, reynolds=5.0e4, mach=0.1)
    # Deliberately coarse/fast mesh — this is a "does the toolchain run" check, not a
    # trustworthy physics result (mesh-independence per Plan Section 12 step 5 is still
    # a prerequisite before any sweep-scale case is treated as ground truth).
    case = run_case(condition, project_dir=str(tmp_path), mesh_fineness=0.3)

    assert case.results.success is True
    assert case.su2_dir.joinpath("history.csv").exists()
    assert case.su2_dir.joinpath("volume.vtu").exists()


def test_naca0012_lift_sign_and_symmetry(tmp_path):
    """Regression test for a real bug found during Week 1 smoke testing: the config
    silently used FREESTREAM_VELOCITY (compressible-solver-only, ignored by INC_
    solvers) instead of INC_VELOCITY_INIT, and separately had the AoA-rotation sign
    backwards — together these made a symmetric airfoil show ~zero or wrong-signed
    lift at nonzero AoA. Guards against reintroducing either bug.

    Uses the default (finer) mesh resolution deliberately: at the very coarse
    fineness=0.3 used by the smoke-run test above, force coefficients are dominated
    by mesh-resolution noise (verified empirically — that mesh gives |CL| ~ 0.005
    regardless of AoA sign, too small and unreliable to check sign against), so a
    coarse mesh cannot actually demonstrate this fix either way.
    """
    from aeropinn.solvers.su2_runner import GeometryCondition, run_case

    case_pos = run_case(
        GeometryCondition(naca_digits="0012", alpha_deg=2.0, reynolds=5.0e4, mach=0.1),
        project_dir=str(tmp_path), mesh_fineness=1.0,
    )
    case_neg = run_case(
        GeometryCondition(naca_digits="0012", alpha_deg=-2.0, reynolds=5.0e4, mach=0.1),
        project_dir=str(tmp_path), mesh_fineness=1.0,
    )

    # Correct sign convention: positive AoA on a symmetric airfoil gives positive CL.
    assert case_pos.results.cl > 0.05
    assert case_neg.results.cl < -0.05
    # Symmetric airfoil: CL(-alpha) ~= -CL(alpha), not required to be exact on a
    # finite mesh, but should be the same order of magnitude and opposite sign.
    assert abs(case_pos.results.cl + case_neg.results.cl) < 0.5 * abs(case_pos.results.cl)
