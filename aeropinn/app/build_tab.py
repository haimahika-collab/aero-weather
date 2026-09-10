"""AeroPINN Gradio tab (Plan Section 27, fast-path scope). Imported by the top-level
app.py and called once inside its `with gr.Blocks(...) as demo:` context to add a new
tab alongside the existing flight-delay tabs.

Only this module + aeropinn/models/pinn.py + aeropinn/geometry/{naca,sdf}.py +
aeropinn/physics/nondim.py are on the deploy path — everything else in aeropinn/ is
dev/offline-only (Plan Section 24). Model/OOD/demo-case artifacts are loaded lazily on
first use, not at import time, so this tab never slows down app.py's startup or the
existing tabs (Plan Section 23 failure mode #10).

HONESTY NOTE shown in the UI itself, not just here: this is a fast-path preliminary
build (Plan conversation, "a few days" scope cut) — ONE trained geometry-conditioned
PINN on a modest (~22 geometry) NACA family, no neural-operator baseline comparison,
no ablations, no multi-seed reporting, and inverse design is not included. It is not
the full validated research study described in aeropinn/PLAN.md.
"""
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = REPO_ROOT / "aeropinn" / "experiments" / "checkpoints"

_cache = {}


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load():
    if _cache:
        return _cache
    from aeropinn.models.pinn import GeometryConditionedPINN

    device = _device()
    model = GeometryConditionedPINN().to(device)
    ckpt_path = CHECKPOINT_DIR / "pinn_v1.pt"
    if not ckpt_path.exists():
        _cache["ready"] = False
        return _cache
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _cache["model"] = model
    _cache["device"] = device
    _cache["ready"] = True

    ood_path = CHECKPOINT_DIR / "ood_mahalanobis.npz"
    if ood_path.exists():
        from aeropinn.evaluation.ood import MahalanobisOOD
        _cache["ood"] = MahalanobisOOD.load(ood_path)

    demo_path = CHECKPOINT_DIR / "held_out_demo_cases.npz"
    if demo_path.exists():
        d = np.load(demo_path, allow_pickle=True)
        _cache["demo_cases"] = list(d["cases"])

    return _cache


def _predict_field(m, p, t, alpha, reynolds):
    from aeropinn.geometry.naca import naca4_surface
    from aeropinn.geometry.sdf import signed_distance_torch

    state = _load()
    model, device = state["model"], state["device"]

    xs = np.linspace(-0.8, 1.8, 90)
    ys = np.linspace(-0.9, 0.9, 60)
    X, Y = np.meshgrid(xs, ys)
    xy_flat = np.stack([X.ravel(), Y.ravel()], axis=-1)

    xy_t = torch.tensor(xy_flat, dtype=torch.float32, device=device)
    n = xy_t.shape[0]
    m_t = torch.full((n,), m, dtype=torch.float32, device=device)
    p_t = torch.full((n,), p, dtype=torch.float32, device=device)
    t_t = torch.full((n,), t, dtype=torch.float32, device=device)

    import time
    t0 = time.time()
    with torch.no_grad():
        sdf = signed_distance_torch(xy_t, m_t, p_t, t_t, n_samples=150).unsqueeze(-1)
        cond = torch.cat([
            sdf, m_t.unsqueeze(-1), p_t.unsqueeze(-1), t_t.unsqueeze(-1),
            torch.full((n, 1), alpha, device=device), torch.full((n, 1), reynolds, device=device),
        ], dim=-1)
        u, v, p_star = model(xy_t[:, 0:1], xy_t[:, 1:2], cond)
    latency_ms = (time.time() - t0) * 1000.0

    speed = torch.sqrt(u**2 + v**2).cpu().numpy().reshape(X.shape)
    cp_field = (2.0 * p_star).cpu().numpy().reshape(X.shape)
    inside = sdf.cpu().numpy().reshape(X.shape) < 0
    speed[inside] = np.nan
    cp_field[inside] = np.nan

    upper, lower = naca4_surface(m, p, t, n_points=150)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    c0 = axes[0].contourf(X, Y, speed, levels=30, cmap="viridis")
    axes[0].plot(upper[:, 0], upper[:, 1], "k-", lw=1.5)
    axes[0].plot(lower[:, 0], lower[:, 1], "k-", lw=1.5)
    axes[0].fill(np.concatenate([upper[:, 0], lower[::-1, 0]]), np.concatenate([upper[:, 1], lower[::-1, 1]]), "white")
    axes[0].set_aspect("equal"); axes[0].set_title("Predicted |velocity| / U_inf")
    plt.colorbar(c0, ax=axes[0], fraction=0.04)

    c1 = axes[1].contourf(X, Y, cp_field, levels=30, cmap="RdBu_r")
    axes[1].plot(upper[:, 0], upper[:, 1], "k-", lw=1.5)
    axes[1].plot(lower[:, 0], lower[:, 1], "k-", lw=1.5)
    axes[1].fill(np.concatenate([upper[:, 0], lower[::-1, 0]]), np.concatenate([upper[:, 1], lower[::-1, 1]]), "white")
    axes[1].set_aspect("equal"); axes[1].set_title("Predicted Cp")
    plt.colorbar(c1, ax=axes[1], fraction=0.04)
    plt.tight_layout()

    ood_text = "OOD check unavailable (not calibrated yet)."
    if "ood" in state:
        query = np.array([m, p, t, alpha, reynolds])
        dist = state["ood"].distance(query)
        is_ood = state["ood"].is_out_of_distribution(query)
        if is_ood:
            ood_text = f"⚠️ OUTSIDE the validated training envelope (Mahalanobis distance {dist:.2f} > threshold {state['ood'].threshold:.2f}). Treat this prediction as untrustworthy, not just less accurate."
        else:
            ood_text = f"✓ Inside the validated training envelope (Mahalanobis distance {dist:.2f} ≤ threshold {state['ood'].threshold:.2f})."

    status = (
        f"**Measured inference latency: {latency_ms:.2f} ms** (device: {device}, no retraining — "
        f"this exact model was trained once on ~22 airfoils and never saw this specific geometry/condition combination).\n\n{ood_text}"
    )
    return fig, status


def _validate_held_out(case_label):
    from aeropinn.geometry.naca import naca_camber

    state = _load()
    if "demo_cases" not in state:
        return None, "No held-out validation cases available yet."
    cases = state["demo_cases"]
    labels = [_case_label(c) for c in cases]
    idx = labels.index(case_label)
    c = cases[idx]

    # A NACA airfoil has two surface points (upper, lower) at most x stations; sorting
    # purely by x interleaves them into a jagged zigzag. Split by the camber line first.
    yc = naca_camber(np.clip(c["surf_x"], 0, 1), c["m"], c["p"])
    is_upper = c["surf_y"] >= yc

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for mask, style_true, style_pred, name in [
        (is_upper, "k-", "r--", "upper"), (~is_upper, "k-", "r--", "lower"),
    ]:
        order = np.argsort(c["surf_x"][mask])
        x_s = c["surf_x"][mask][order]
        ax.plot(x_s, c["surf_cp_true"][mask][order], style_true, lw=2,
                 label="SU2 (hidden reference)" if name == "upper" else None)
        ax.plot(x_s, c["surf_cp_pred"][mask][order], style_pred, lw=2,
                 label="AeroPINN prediction" if name == "upper" else None)
    ax.invert_yaxis()
    ax.set_xlabel("x / chord"); ax.set_ylabel("Cp"); ax.legend()
    ax.set_title(f"Held-out geometry: m={c['m']:.3f} p={c['p']:.2f} t={c['t']:.3f}, "
                 f"AoA={c['alpha_deg']:.1f} deg, Re={c['reynolds']:.0e}")
    plt.tight_layout()

    text = (
        f"**This exact airfoil/condition combination was never in the training set.**\n\n"
        f"Reference (SU2 CFD): CL={c['cl_true']:.4f}, CD={c['cd_true']:.4f}\n\n"
        f"AeroPINN field relative L2 error vs. SU2 — u: {c['u_rel_l2']:.3f}, v: {c['v_rel_l2']:.3f}, p: {c['p_rel_l2']:.3f}"
    )
    return fig, text


def _case_label(c):
    return f"m={c['m']:.3f}, p={c['p']:.2f}, t={c['t']:.3f}, AoA={c['alpha_deg']:.1f}, Re={c['reynolds']:.0e}"


def build_tab():
    with gr.Tab("AeroPINN — Airfoil Aerodynamics"):
        state = _load()
        if not state.get("ready"):
            gr.Markdown(
                "### AeroPINN\n\nNo trained checkpoint found yet — this tab activates once "
                "`aeropinn/experiments/checkpoints/pinn_v1.pt` exists. See `aeropinn/PLAN.md` "
                "for the full engineering plan."
            )
            return

        gr.Markdown(
            "### AeroPINN — Geometry-Conditioned Physics-Informed Airfoil Surrogate\n\n"
            "**Preliminary fast-path build** — one trained PINN on ~22 NACA 4-digit airfoils, "
            "not yet the full validated study (no baseline comparison, no ablations, single "
            "training seed). Full plan and honest scope caveats: `aeropinn/PLAN.md` in this repo.\n\n"
            "Predicts flow around an airfoil from its geometry (camber, camber location, thickness) "
            "and operating condition (angle of attack, Reynolds number) — **one trained model, "
            "no retraining per shape.**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                m_in = gr.Slider(0.0, 0.06, value=0.02, step=0.005, label="m — max camber")
                p_in = gr.Slider(0.10, 0.60, value=0.40, step=0.01, label="p — camber location")
                t_in = gr.Slider(0.08, 0.20, value=0.12, step=0.005, label="t — max thickness")
                alpha_in = gr.Slider(-4.0, 8.0, value=2.0, step=0.5, label="Angle of attack (deg)")
                re_in = gr.Slider(1.0e4, 1.0e5, value=5.0e4, step=1.0e3, label="Reynolds number")
                predict_btn = gr.Button("Predict Flow", variant="primary")
            with gr.Column(scale=2):
                field_plot = gr.Plot(label="Predicted flow field")
                status_md = gr.Markdown()

        predict_btn.click(_predict_field, inputs=[m_in, p_in, t_in, alpha_in, re_in], outputs=[field_plot, status_md])

        if "demo_cases" in state:
            gr.Markdown("---\n#### Held-out geometry validation\n"
                        "These airfoils were *never seen during training* — compare AeroPINN's "
                        "prediction directly against the hidden SU2 CFD reference.")
            labels = [_case_label(c) for c in state["demo_cases"]]
            with gr.Row():
                case_dropdown = gr.Dropdown(choices=labels, value=labels[0], label="Held-out case")
                validate_btn = gr.Button("Compare vs. hidden CFD reference")
            with gr.Row():
                cp_plot = gr.Plot(label="Surface Cp: predicted vs. reference")
                validate_text = gr.Markdown()
            validate_btn.click(_validate_held_out, inputs=[case_dropdown], outputs=[cp_plot, validate_text])
