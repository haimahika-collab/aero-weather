"""Export real project data as clean static JSON for the Next.js research website
(web/). One-directional: reads aeropinn/ artifacts, writes web/data/*.json. Never
runs from the website itself -- run manually before each site deploy.

Run as: python -m aeropinn.export_site_data
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "aeropinn" / "experiments" / "checkpoints"
PROCESSED_DIR = REPO_ROOT / "aeropinn" / "data" / "processed" / "sweep_v1"
WEB_DATA_DIR = REPO_ROOT / "web" / "data"

SOURCE_FILES = {
    "held_out_eval": CHECKPOINT_DIR / "held_out_eval.json",
    "held_out_eval_b2": CHECKPOINT_DIR / "held_out_eval_b2.json",
    "inverse_design_cache": CHECKPOINT_DIR / "inverse_design_cache.json",
    "geometries": PROCESSED_DIR / "geometries.json",
    "held_out_demo_cases": CHECKPOINT_DIR / "held_out_demo_cases.npz",
    "held_out_demo_cases_b2": CHECKPOINT_DIR / "held_out_demo_cases_b2.npz",
    "theme": REPO_ROOT / "aeropinn" / "app" / "theme.py",
}


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def export_palette():
    from aeropinn.app.theme import AERO_PALETTE
    (WEB_DATA_DIR / "palette.json").write_text(json.dumps(AERO_PALETTE, indent=2))
    print("wrote palette.json", flush=True)


def _latency_cold_warm_split(per_case):
    """Split latency into first-call-per-geometry (cold, includes model/OOD load
    path) vs. subsequent calls (warm) -- the blended top-level mean is misleading
    on its own (Plan integrity checklist item)."""
    seen_geoms = set()
    cold, warm = [], []
    for c in per_case:
        gid = c["geom_id"]
        if gid not in seen_geoms:
            cold.append(c["inference_latency_ms"])
            seen_geoms.add(gid)
        else:
            warm.append(c["inference_latency_ms"])
    return {
        "cold_mean_ms": float(np.mean(cold)) if cold else None,
        "cold_n": len(cold),
        "warm_mean_ms": float(np.mean(warm)) if warm else None,
        "warm_median_ms": float(np.median(warm)) if warm else None,
        "warm_n": len(warm),
    }


def export_eval(name: str, out_name: str):
    path = SOURCE_FILES[name]
    if not path.exists():
        print(f"SKIP {name}: {path} not found", flush=True)
        return None
    with open(path) as f:
        data = json.load(f)
    data["latency_breakdown"] = _latency_cold_warm_split(data["per_case"])
    (WEB_DATA_DIR / out_name).write_text(json.dumps(data, indent=2))
    print(f"wrote {out_name}", flush=True)
    return data


def export_geometries():
    path = SOURCE_FILES["geometries"]
    with open(path) as f:
        data = json.load(f)
    (WEB_DATA_DIR / "geometries.json").write_text(json.dumps(data, indent=2))
    print("wrote geometries.json", flush=True)
    return data


def _npz_to_json(path: Path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    cases = list(d["cases"])
    out = []
    for c in cases:
        out.append({
            "geom_id": str(c["geom_id"]), "m": float(c["m"]), "p": float(c["p"]), "t": float(c["t"]),
            "alpha_deg": float(c["alpha_deg"]), "reynolds": float(c["reynolds"]),
            "surf_x": c["surf_x"].tolist(), "surf_y": c["surf_y"].tolist(),
            "surf_cp_true": c["surf_cp_true"].tolist(), "surf_cp_pred": c["surf_cp_pred"].tolist(),
            "cl_true": float(c["cl_true"]), "cd_true": float(c["cd_true"]),
            "u_rel_l2": float(c["u_rel_l2"]), "v_rel_l2": float(c["v_rel_l2"]), "p_rel_l2": float(c["p_rel_l2"]),
        })
    return out


def export_cp_examples():
    physics = _npz_to_json(SOURCE_FILES["held_out_demo_cases"])
    data_only = _npz_to_json(SOURCE_FILES["held_out_demo_cases_b2"])
    out = {"physics_guided": physics or [], "data_only": data_only or []}
    (WEB_DATA_DIR / "cp_examples.json").write_text(json.dumps(out, indent=2))
    print(f"wrote cp_examples.json ({len(out['physics_guided'])} physics, "
          f"{len(out['data_only'])} data-only cases)", flush=True)


def export_inverse_design():
    path = SOURCE_FILES["inverse_design_cache"]
    if not path.exists():
        print("SKIP inverse_design: not found", flush=True)
        return
    with open(path) as f:
        data = json.load(f)
    (WEB_DATA_DIR / "inverse_design.json").write_text(json.dumps(data, indent=2))
    print(f"wrote inverse_design.json ({len(data)} runs)", flush=True)


def export_design_space_error(geometries, held_out_eval):
    """Figure 06 join: geometries.json (m,p,t,split) + held_out_eval.json per_case
    error, matched by geom_id (averaged across that geometry's alpha/Re cases)."""
    if geometries is None or held_out_eval is None:
        print("SKIP design_space_error: missing inputs", flush=True)
        return
    by_geom = {}
    for c in held_out_eval["per_case"]:
        by_geom.setdefault(c["geom_id"], []).append(c)

    points = []
    for g in geometries:
        entry = {"geom_id": g["geom_id"], "m": g["m"], "p": g["p"], "t": g["t"], "split": g["split"]}
        cases = by_geom.get(g["geom_id"])
        if cases:
            entry["mean_u_rel_l2"] = float(np.mean([c["u_rel_l2"] for c in cases]))
            entry["mean_v_rel_l2"] = float(np.mean([c["v_rel_l2"] for c in cases]))
            entry["mean_p_rel_l2"] = float(np.mean([c["p_rel_l2"] for c in cases]))
        points.append(entry)

    (WEB_DATA_DIR / "design_space_error.json").write_text(json.dumps(points, indent=2))
    print(f"wrote design_space_error.json ({len(points)} geometries, "
          f"{sum(1 for p in points if 'mean_p_rel_l2' in p)} with error)", flush=True)


LIMITATIONS = [
    {"topic": "Training data", "text": "16 training geometries and 6 held-out test geometries, "
     "a single Latin-Hypercube sample, a single training seed. Not a multi-seed study.",
     "source": "aeropinn/PLAN.md Section 13/17"},
    {"topic": "Reference method", "text": "SU2 steady laminar incompressible solver. The mesh-independence "
     "check (Plan Section 12 step 5) has not yet been run -- sweep-scale CFD data should not be treated "
     "as fully certified ground truth until it is.", "source": "aeropinn/PLAN.md Section 12"},
    {"topic": "Geometry family", "text": "NACA 4-digit airfoils only, bounded to m in [0, 0.06], "
     "p in [0.10, 0.60], t in [0.08, 0.20].", "source": "aeropinn/PLAN.md Section 6"},
    {"topic": "Operating envelope", "text": "Reynolds number 3x10^4-1x10^5 (laminar regime), angle of attack "
     "-4 deg to 8 deg (pre-stall), low-Mach incompressible proxy (M=0.1).",
     "source": "aeropinn/PLAN.md Section 6"},
    {"topic": "Governing physics", "text": "Steady, 2D, incompressible, laminar Navier-Stokes. This is a "
     "physics-informed neural network in the literal sense (PDE-residual loss term), not a general-purpose "
     "or turbulent-flow solver -- it does not claim to solve the Navier-Stokes existence/smoothness problem "
     "or generalize beyond this laminar 2D regime.", "source": "aeropinn/PLAN.md Section 6/7"},
    {"topic": "Model uncertainty", "text": "Out-of-domain detection is a binary Mahalanobis-distance check "
     "in (m,p,t,alpha,Re) space, calibrated on the training distribution. It is not a graded confidence "
     "score, and PDE residual magnitude is a diagnostic only -- never treated as proof of correctness.",
     "source": "aeropinn/PLAN.md Section 21, aeropinn/evaluation/ood.py"},
    {"topic": "Inverse design", "text": "All 3 tested optimization runs disagreed with independent SU2 "
     "validation ('validated_disagreement') -- the optimizer found designs the surrogate over- or "
     "under-rated near the edges of the validated envelope. No inverse-design candidate from this project "
     "has yet been confirmed as a genuine aerodynamic improvement.",
     "source": "aeropinn/experiments/checkpoints/inverse_design_cache.json"},
]


def export_limitations():
    (WEB_DATA_DIR / "limitations.json").write_text(json.dumps(LIMITATIONS, indent=2))
    print("wrote limitations.json", flush=True)


def export_manifest():
    hashes = {name: _sha256(path) for name, path in SOURCE_FILES.items()}
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_file_hashes": hashes,
    }
    (WEB_DATA_DIR / "site_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote site_manifest.json", flush=True)


def main():
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_palette()
    held_out_eval = export_eval("held_out_eval", "held_out_eval.json")
    export_eval("held_out_eval_b2", "held_out_eval_b2.json")
    geometries = export_geometries()
    export_cp_examples()
    export_inverse_design()
    export_design_space_error(geometries, held_out_eval)
    export_limitations()
    export_manifest()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
