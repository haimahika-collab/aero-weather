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
    # trustworthy physics result (see su2_runner.py's module docstring: CL from a run
    # this coarse should NOT be treated as valid data without a mesh-independence pass).
    case = run_case(condition, project_dir=str(tmp_path), mesh_fineness=0.3)

    assert case.results.success is True
    assert case.su2_dir.joinpath("history.csv").exists()
    assert case.su2_dir.joinpath("volume.vtu").exists()
