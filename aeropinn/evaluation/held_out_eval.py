"""Held-out geometry evaluation (Plan Section 13/16, fast-path scope: one pooled
held-out set rather than the full plan's separate interpolation/extrapolation splits).

Run as: python -m aeropinn.evaluation.held_out_eval
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from aeropinn.geometry.sdf import signed_distance_torch
from aeropinn.models.pinn import GeometryConditionedPINN
from aeropinn.training.train_pinn import CHECKPOINT_DIR, PROCESSED_DIR, load_sweep_data, _device


def evaluate():
    device = _device()
    ckpt = torch.load(CHECKPOINT_DIR / "pinn_v1.pt", map_location=device)
    model = GeometryConditionedPINN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, held_out_data, _ = load_sweep_data()
    print(f"Evaluating on {len(held_out_data)} held-out cases", flush=True)

    results = []
    demo_cases = []
    for e in held_out_data:
        xy = torch.tensor(e["xy"], dtype=torch.float32, device=device)
        sdf = torch.tensor(e["sdf"], dtype=torch.float32, device=device).unsqueeze(-1)
        n = xy.shape[0]
        m = torch.full((n, 1), float(e["m"]), device=device)
        p = torch.full((n, 1), float(e["p"]), device=device)
        t = torch.full((n, 1), float(e["t"]), device=device)
        alpha = torch.full((n, 1), float(e["alpha_deg"]), device=device)
        re = torch.full((n, 1), float(e["reynolds"]), device=device)
        cond = torch.cat([sdf, m, p, t, alpha, re], dim=-1)

        x = xy[:, 0:1]
        y = xy[:, 1:2]

        t0 = time.time()
        with torch.no_grad():
            u_pred, v_pred, p_pred = model(x, y, cond)
        latency_ms = (time.time() - t0) * 1000.0

        u_true = torch.tensor(e["u_star"], dtype=torch.float32, device=device).unsqueeze(-1)
        v_true = torch.tensor(e["v_star"], dtype=torch.float32, device=device).unsqueeze(-1)
        p_true = torch.tensor(e["p_star"], dtype=torch.float32, device=device).unsqueeze(-1)

        def rel_l2(pred, true):
            return (torch.norm(pred - true) / (torch.norm(true) + 1e-8)).item()

        results.append({
            "geom_id": str(e["geom_id"]), "alpha_deg": float(e["alpha_deg"]), "reynolds": float(e["reynolds"]),
            "u_rel_l2": rel_l2(u_pred, u_true), "v_rel_l2": rel_l2(v_pred, v_true), "p_rel_l2": rel_l2(p_pred, p_true),
            "n_points": n, "inference_latency_ms": latency_ms,
        })

        # Demo asset: predicted vs. reference surface Cp at the actual CFD surface
        # points, for the live-site "held-out validation" panel (Plan Section 27
        # step 4/5's "compare against hidden reference" requirement).
        surf_xy = torch.tensor(e["surf_xy"], dtype=torch.float32, device=device)
        n_surf = surf_xy.shape[0]
        surf_sdf = torch.zeros(n_surf, 1, device=device)  # on the surface by construction
        surf_cond = torch.cat([
            surf_sdf,
            torch.full((n_surf, 1), float(e["m"]), device=device),
            torch.full((n_surf, 1), float(e["p"]), device=device),
            torch.full((n_surf, 1), float(e["t"]), device=device),
            torch.full((n_surf, 1), float(e["alpha_deg"]), device=device),
            torch.full((n_surf, 1), float(e["reynolds"]), device=device),
        ], dim=-1)
        with torch.no_grad():
            _, _, p_surf_pred = model(surf_xy[:, 0:1], surf_xy[:, 1:2], surf_cond)
        cp_pred = (2.0 * p_surf_pred).squeeze(-1).cpu().numpy()

        demo_cases.append({
            "geom_id": str(e["geom_id"]), "m": float(e["m"]), "p": float(e["p"]), "t": float(e["t"]),
            "alpha_deg": float(e["alpha_deg"]), "reynolds": float(e["reynolds"]),
            "surf_x": e["surf_xy"][:, 0], "surf_y": e["surf_xy"][:, 1],
            "surf_cp_true": e["surf_cp"], "surf_cp_pred": cp_pred,
            "cl_true": float(e["cl"]), "cd_true": float(e["cd"]),
            "u_rel_l2": results[-1]["u_rel_l2"], "v_rel_l2": results[-1]["v_rel_l2"], "p_rel_l2": results[-1]["p_rel_l2"],
        })

    u_errs = [r["u_rel_l2"] for r in results]
    v_errs = [r["v_rel_l2"] for r in results]
    p_errs = [r["p_rel_l2"] for r in results]
    latencies = [r["inference_latency_ms"] for r in results]

    summary = {
        "n_cases": len(results),
        "mean_u_rel_l2": float(np.mean(u_errs)), "mean_v_rel_l2": float(np.mean(v_errs)),
        "mean_p_rel_l2": float(np.mean(p_errs)),
        "mean_inference_latency_ms": float(np.mean(latencies)),
        "device": str(device),
        "per_case": results,
    }
    out_path = CHECKPOINT_DIR / "held_out_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Mean held-out relative L2 error: u={summary['mean_u_rel_l2']:.4f} "
          f"v={summary['mean_v_rel_l2']:.4f} p={summary['mean_p_rel_l2']:.4f}", flush=True)
    print(f"Mean inference latency: {summary['mean_inference_latency_ms']:.2f} ms "
          f"(on {device}, batch of ~{results[0]['n_points']} points)", flush=True)
    print(f"Saved: {out_path}", flush=True)

    demo_path = CHECKPOINT_DIR / "held_out_demo_cases.npz"
    np.savez(demo_path, cases=np.array(demo_cases, dtype=object), allow_pickle=True)
    print(f"Saved demo assets ({len(demo_cases)} cases): {demo_path}", flush=True)

    return summary


if __name__ == "__main__":
    evaluate()
