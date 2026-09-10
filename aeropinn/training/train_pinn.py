"""Train the geometry-conditioned PINN on the sweep data (Plan Section 10/11, fast-path
scope: fixed loss weights rather than full SA-PINN/NTK adaptive reweighting, single
seed rather than the full plan's >=3-seed requirement — both explicitly deferred, not
silently dropped, given the days-not-weeks timeline).

Loss = lambda_pde * L_PDE + lambda_bc * L_BC + lambda_data * L_data  (Plan Section 11).
PDE collocation points are sampled per training geometry each epoch (not reused from
the CFD data) — L_data supervises the network on actual CFD points; L_PDE enforces the
governing equations everywhere in the domain, which is the point of a PINN vs. a plain
regressor (this distinction is what ablation B2 in the full plan would isolate).

Run as: python -m aeropinn.training.train_pinn [--epochs N]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aeropinn.geometry.sdf import signed_distance_torch
from aeropinn.models.pinn import GeometryConditionedPINN
from aeropinn.physics.navier_stokes import pde_residual

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "aeropinn" / "data" / "processed" / "sweep_v1"
CHECKPOINT_DIR = REPO_ROOT / "aeropinn" / "experiments" / "checkpoints"

MACH = 0.1
U_INF = MACH * 340.29


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sweep_data():
    with open(PROCESSED_DIR / "geometries.json") as f:
        geometries = {g["geom_id"]: g for g in json.load(f)}

    train_data, held_out_data = [], []
    for npz_path in sorted(PROCESSED_DIR.glob("*.npz")):
        d = np.load(npz_path)
        geom_id = str(d["geom_id"])
        entry = {k: d[k] for k in d.files}
        if geometries[geom_id]["split"] == "train":
            train_data.append(entry)
        else:
            held_out_data.append(entry)
    return train_data, held_out_data, geometries


def build_data_tensors(entries, device):
    """Flatten a list of per-case npz dicts into (N, ...) tensors for L_data."""
    xs, ys, sdfs, ms, ps, ts, alphas, res, us, vs, pps = [], [], [], [], [], [], [], [], [], [], []
    for e in entries:
        n = e["xy"].shape[0]
        xs.append(e["xy"][:, 0]); ys.append(e["xy"][:, 1]); sdfs.append(e["sdf"])
        ms.append(np.full(n, e["m"])); ps.append(np.full(n, e["p"])); ts.append(np.full(n, e["t"]))
        alphas.append(np.full(n, e["alpha_deg"])); res.append(np.full(n, e["reynolds"]))
        us.append(e["u_star"]); vs.append(e["v_star"]); pps.append(e["p_star"])

    def cat(arrs):
        return torch.tensor(np.concatenate(arrs), dtype=torch.float32, device=device).unsqueeze(-1)

    x, y = cat(xs), cat(ys)
    cond = torch.cat([cat(sdfs), cat(ms), cat(ps), cat(ts), cat(alphas), cat(res)], dim=-1)
    target = torch.cat([cat(us), cat(vs), cat(pps)], dim=-1)
    return x, y, cond, target


def sample_pde_collocation(train_geometries, n_points, device):
    """Sample interior collocation points across randomly-drawn training geometries
    (Plan Section 10's mixed-geometry batching), each with its own (alpha, Re) drawn
    from the same ranges used to generate the CFD sweep, and its own analytic SDF.
    """
    rng = np.random.default_rng()
    geoms = list(train_geometries.values())
    picks = rng.choice(len(geoms), size=n_points)
    m = np.array([geoms[i]["m"] for i in picks])
    p = np.array([geoms[i]["p"] for i in picks])
    t = np.array([geoms[i]["t"] for i in picks])
    alpha = rng.uniform(-4.0, 8.0, size=n_points)
    re = rng.uniform(1.0e4, 1.0e5, size=n_points)
    x = rng.uniform(-3.0, 4.0, size=n_points)
    y = rng.uniform(-3.0, 3.0, size=n_points)

    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(-1).requires_grad_(True)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1).requires_grad_(True)
    m_t = torch.tensor(m, dtype=torch.float32, device=device)
    p_t = torch.tensor(p, dtype=torch.float32, device=device)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
    re_t = torch.tensor(re, dtype=torch.float32, device=device)

    xy = torch.cat([x_t, y_t], dim=-1)
    sdf = signed_distance_torch(xy, m_t, p_t, t_t, n_samples=150).unsqueeze(-1)
    cond = torch.cat([sdf, m_t.unsqueeze(-1), p_t.unsqueeze(-1), t_t.unsqueeze(-1),
                       alpha_t.unsqueeze(-1), re_t.unsqueeze(-1)], dim=-1)
    re_for_residual = re_t.unsqueeze(-1)
    return x_t, y_t, cond, re_for_residual


def sample_bc_points(train_geometries, n_points, device):
    """Freestream far-field BC points (Plan Section 8) — simple box far from any airfoil."""
    rng = np.random.default_rng()
    geoms = list(train_geometries.values())
    picks = rng.choice(len(geoms), size=n_points)
    m = np.array([geoms[i]["m"] for i in picks])
    p = np.array([geoms[i]["p"] for i in picks])
    t = np.array([geoms[i]["t"] for i in picks])
    alpha = rng.uniform(-4.0, 8.0, size=n_points)
    re = rng.uniform(1.0e4, 1.0e5, size=n_points)

    # Points on a far-field ring, |r| ~ domain radius, far enough that SDF is unambiguous.
    theta = rng.uniform(0, 2 * np.pi, size=n_points)
    radius = rng.uniform(3.5, 4.0, size=n_points)
    x = 0.5 + radius * np.cos(theta)
    y = radius * np.sin(theta)

    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
    m_t = torch.tensor(m, dtype=torch.float32, device=device)
    p_t = torch.tensor(p, dtype=torch.float32, device=device)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
    re_t = torch.tensor(re, dtype=torch.float32, device=device)

    xy = torch.cat([x_t, y_t], dim=-1)
    sdf = signed_distance_torch(xy, m_t, p_t, t_t, n_samples=150).unsqueeze(-1)
    cond = torch.cat([sdf, m_t.unsqueeze(-1), p_t.unsqueeze(-1), t_t.unsqueeze(-1),
                       alpha_t.unsqueeze(-1), re_t.unsqueeze(-1)], dim=-1)

    alpha_rad = torch.deg2rad(alpha_t.unsqueeze(-1))
    u_target = torch.cos(alpha_rad)
    v_target = torch.sin(alpha_rad)
    return x_t, y_t, cond, u_target, v_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--n-collocation", type=int, default=2000)
    parser.add_argument("--n-bc", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-pde", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=5.0)
    parser.add_argument("--lambda-data", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", type=str, default="pinn_v1",
                         help="Checkpoint/history filename stem, e.g. 'pinn_v1_b2_dataonly' for "
                              "the Physics Ablation baseline (--lambda-pde 0) so it doesn't "
                              "overwrite the production pinn_v1.pt.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device()
    print(f"Device: {device}", flush=True)

    train_data, held_out_data, geometries = load_sweep_data()
    train_geometries = {gid: g for gid, g in geometries.items() if g["split"] == "train"}
    print(f"Loaded {len(train_data)} train cases, {len(held_out_data)} held-out cases, "
          f"{len(train_geometries)} distinct training geometries", flush=True)

    x_data, y_data, cond_data, target_data = build_data_tensors(train_data, device)
    print(f"L_data supervision points: {x_data.shape[0]}", flush=True)

    model = GeometryConditionedPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def forward_fn(x, y, cond):
        return model(x, y, cond)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        x_f, y_f, cond_f, re_f = sample_pde_collocation(train_geometries, args.n_collocation, device)
        residuals = pde_residual(forward_fn, x_f, y_f, cond_f, re_f)
        l_pde = residuals["continuity"].pow(2).mean() + residuals["momentum_x"].pow(2).mean() + residuals["momentum_y"].pow(2).mean()

        x_bc, y_bc, cond_bc, u_target, v_target = sample_bc_points(train_geometries, args.n_bc, device)
        u_bc, v_bc, _ = model(x_bc, y_bc, cond_bc)
        l_bc = (u_bc - u_target).pow(2).mean() + (v_bc - v_target).pow(2).mean()

        u_pred, v_pred, p_pred = model(x_data, y_data, cond_data)
        l_data = (u_pred - target_data[:, 0:1]).pow(2).mean() + \
                 (v_pred - target_data[:, 1:2]).pow(2).mean() + \
                 (p_pred - target_data[:, 2:3]).pow(2).mean()

        loss = args.lambda_pde * l_pde + args.lambda_bc * l_bc + args.lambda_data * l_data
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f"epoch {epoch:5d}  loss={loss.item():.5f}  pde={l_pde.item():.5f}  "
                  f"bc={l_bc.item():.5f}  data={l_data.item():.5f}  ({elapsed:.0f}s)", flush=True)
            history.append({"epoch": epoch, "loss": loss.item(), "l_pde": l_pde.item(),
                             "l_bc": l_bc.item(), "l_data": l_data.item(), "elapsed_s": elapsed})

    ckpt_path = CHECKPOINT_DIR / f"{args.out_name}.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args), "history": history}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)

    history_name = "train_history.json" if args.out_name == "pinn_v1" else f"train_history_{args.out_name}.json"
    with open(CHECKPOINT_DIR / history_name, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
