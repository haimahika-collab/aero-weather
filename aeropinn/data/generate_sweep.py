"""Fast-path CFD sweep generation (Plan Section 12, scoped down per the days-not-weeks
timeline): sample a modest NACA 4-digit geometry family, run each (geometry, AoA, Re)
combination through SU2 (laminar incompressible, Plan Section 6), and save processed
training/eval data — never the raw meshes/solver working directories (those stay under
data/raw/, gitignored).

Run as: python -m aeropinn.data.generate_sweep [--n-train N] [--n-held-out N] [--workers N]

Dev/offline-only: never imported by the deployed app.py.
"""
import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from aeropinn.geometry.sampling import sample_geometries
from aeropinn.geometry.sdf import signed_distance_numpy

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "aeropinn" / "data" / "raw" / "su2_sweep"
PROCESSED_DIR = REPO_ROOT / "aeropinn" / "data" / "processed" / "sweep_v1"

ALPHAS_DEG = [-2.0, 2.0, 6.0]
REYNOLDS = [3.0e4, 7.0e4]
MACH = 0.1
MESH_FINENESS = 1.0  # verified (su2_runner.py) to give physically correct, sign-consistent results
N_INTERIOR_POINTS = 300
DOMAIN_CLIP_RADIUS = 3.0  # chords; keep training focus near the airfoil, not the far outer domain


def _run_one_case(args):
    geom, alpha, reynolds, case_index, total_cases = args
    from aeropinn.solvers.su2_runner import GeometryCondition, run_case  # re-import in worker process

    case_dir = RAW_DIR / geom.geom_id
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{geom.geom_id}_aoa{alpha:+.1f}_re{reynolds:.0e}.npz"

    t0 = time.time()
    try:
        condition = GeometryCondition(naca_digits=geom.naca_digits, alpha_deg=alpha, reynolds=reynolds, mach=MACH)
        case = run_case(condition, project_dir=str(case_dir), mesh_fineness=MESH_FINENESS)
        if not case.results.success:
            return {"status": "su2_failed", "geom_id": geom.geom_id, "alpha": alpha, "reynolds": reynolds}

        import pyvista as pv

        vol = pv.read(str(case.su2_dir / "volume.vtu"))
        pts = vol.points[:, :2]
        u_inf = MACH * 340.29
        vel = vol.point_data["Velocity"][:, :2]
        cp = vol.point_data["Pressure_Coefficient"]

        near = np.linalg.norm(pts, axis=1) < DOMAIN_CLIP_RADIUS
        idx_pool = np.where(near)[0]
        rng = np.random.default_rng(hash(geom.geom_id + str(alpha) + str(reynolds)) % (2**32))
        n_pick = min(N_INTERIOR_POINTS, len(idx_pool))
        idx = rng.choice(idx_pool, size=n_pick, replace=False)

        xy = pts[idx]
        sdf = signed_distance_numpy(xy, geom.m, geom.p, geom.t, n_samples=300)
        u_star = vel[idx, 0] / u_inf
        v_star = vel[idx, 1] / u_inf
        p_star = cp[idx] / 2.0  # Plan Section 7 convention: Cp = 2*p_star

        surf = pd.read_csv(case.dir / "surface_data.csv")
        surf_xy = surf[["x", "y"]].to_numpy()
        surf_cp = surf["cp"].to_numpy()

        np.savez(
            out_path,
            geom_id=geom.geom_id, split=geom.split, m=geom.m, p=geom.p, t=geom.t,
            alpha_deg=alpha, reynolds=reynolds, mach=MACH,
            xy=xy.astype(np.float32), sdf=sdf.astype(np.float32),
            u_star=u_star.astype(np.float32), v_star=v_star.astype(np.float32), p_star=p_star.astype(np.float32),
            surf_xy=surf_xy.astype(np.float32), surf_cp=surf_cp.astype(np.float32),
            cl=case.results.cl, cd=case.results.cd, cm=case.results.cm,
        )
        dt = time.time() - t0
        print(f"[{case_index}/{total_cases}] OK  {geom.geom_id} aoa={alpha:+.1f} Re={reynolds:.0e}  ({dt:.0f}s)", flush=True)
        return {"status": "ok", "geom_id": geom.geom_id, "alpha": alpha, "reynolds": reynolds, "out": str(out_path)}
    except Exception as e:
        print(f"[{case_index}/{total_cases}] FAIL {geom.geom_id} aoa={alpha:+.1f} Re={reynolds:.0e}: {e}", flush=True)
        traceback.print_exc()
        return {"status": "error", "geom_id": geom.geom_id, "alpha": alpha, "reynolds": reynolds, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-held-out", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    geometries = sample_geometries(args.n_train, args.n_held_out, seed=args.seed)
    with open(PROCESSED_DIR / "geometries.json", "w") as f:
        json.dump([g.__dict__ for g in geometries], f, indent=2)

    tasks = []
    for geom in geometries:
        for alpha in ALPHAS_DEG:
            for re in REYNOLDS:
                tasks.append((geom, alpha, re))
    total = len(tasks)
    tasks = [(g, a, r, i + 1, total) for i, (g, a, r) in enumerate(tasks)]

    print(f"Sweep: {len(geometries)} geometries x {len(ALPHAS_DEG)} AoA x {len(REYNOLDS)} Re = {total} cases, "
          f"{args.workers} parallel workers", flush=True)

    results = []
    with mp.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(_run_one_case, tasks):
            results.append(r)
            with open(PROCESSED_DIR / "manifest.json", "w") as f:
                json.dump(results, f, indent=2)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"DONE: {n_ok}/{total} cases succeeded. Manifest: {PROCESSED_DIR / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
