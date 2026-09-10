"""All matplotlib rendering for the AeroPINN tab. Consumes aeropinn/app/core.py's
dataclasses — no model loading, no torch calls here, just plotting. Colors come from
aeropinn/app/theme.py::AERO_PALETTE so plots and UI chrome never disagree.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from aeropinn.app.theme import AERO_PALETTE as P


def _dark_style(fig, *axes):
    fig.patch.set_facecolor(P["panel"])
    for ax in axes:
        ax.set_facecolor(P["panel_alt"])
        ax.tick_params(colors=P["text_muted"], labelsize=8)
        ax.xaxis.label.set_color(P["text_muted"])
        ax.yaxis.label.set_color(P["text_muted"])
        ax.title.set_color(P["text"])
        for spine in ax.spines.values():
            spine.set_color(P["border"])
        ax.grid(color=P["border"], alpha=0.4, linewidth=0.5)


def _draw_airfoil_outline(ax, upper, lower, fill=True):
    if fill:
        ax.fill(
            np.concatenate([upper[:, 0], lower[::-1, 0]]),
            np.concatenate([upper[:, 1], lower[::-1, 1]]),
            color=P["bg"], zorder=5,
        )
    ax.plot(upper[:, 0], upper[:, 1], color=P["text"], lw=1.2, zorder=6)
    ax.plot(lower[:, 0], lower[:, 1], color=P["text"], lw=1.2, zorder=6)


def render_field_plot(result, view: str = "cp"):
    """view: 'geometry' | 'velocity' | 'cp'"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _dark_style(fig, ax)

    if view == "geometry":
        _draw_airfoil_outline(ax, result.upper, result.lower, fill=False)
        ax.fill(
            np.concatenate([result.upper[:, 0], result.lower[::-1, 0]]),
            np.concatenate([result.upper[:, 1], result.lower[::-1, 1]]),
            color=P["accent"], alpha=0.25,
        )
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.35, 0.35)
        ax.set_title(f"Geometry — NACA-style m={result.m:.3f} p={result.p:.2f} t={result.t:.3f}, "
                     f"AoA={result.alpha_deg:.1f}°")
    elif view == "velocity":
        c = ax.contourf(result.X, result.Y, result.speed, levels=30, cmap="viridis")
        cbar = plt.colorbar(c, ax=ax, fraction=0.045)
        cbar.ax.yaxis.set_tick_params(color=P["text_muted"], labelcolor=P["text_muted"])
        cbar.outline.set_edgecolor(P["border"])
        _draw_airfoil_outline(ax, result.upper, result.lower)
        ax.set_aspect("equal")
        ax.set_xlabel("x / chord"); ax.set_ylabel("y / chord")
        ax.set_title("Predicted |velocity| / U_inf")
    else:  # cp
        c = ax.contourf(result.X, result.Y, result.cp_field, levels=30, cmap="RdBu_r")
        cbar = plt.colorbar(c, ax=ax, fraction=0.045)
        cbar.ax.yaxis.set_tick_params(color=P["text_muted"], labelcolor=P["text_muted"])
        cbar.outline.set_edgecolor(P["border"])
        _draw_airfoil_outline(ax, result.upper, result.lower)
        ax.set_aspect("equal")
        ax.set_xlabel("x / chord"); ax.set_ylabel("y / chord")
        ax.set_title("Predicted Cp")

    plt.tight_layout()
    return fig


def render_cp_overlay(case: dict, show_reference: bool = True):
    """Predicted (and optionally hidden SU2 reference) surface Cp, split upper/lower
    via the camber line (a NACA airfoil has two surface points per x station —
    sorting by x alone interleaves them into a jagged, physically-nonsensical
    zigzag). show_reference=False renders prediction-only, for the Unseen Geometry
    Challenge's progressive-disclosure flow (predict first, reveal reference after).
    """
    from aeropinn.geometry.naca import naca_camber

    yc = naca_camber(np.clip(case["surf_x"], 0, 1), case["m"], case["p"])
    is_upper = case["surf_y"] >= yc

    fig, ax = plt.subplots(figsize=(6, 4.5))
    _dark_style(fig, ax)
    for mask, name in [(is_upper, "upper"), (~is_upper, "lower")]:
        order = np.argsort(case["surf_x"][mask])
        x_s = case["surf_x"][mask][order]
        if show_reference:
            ax.plot(x_s, case["surf_cp_true"][mask][order], color=P["reference"], lw=2,
                    label="SU2 (hidden reference)" if name == "upper" else None)
        ax.plot(x_s, case["surf_cp_pred"][mask][order], color=P["prediction"], lw=2, linestyle="--",
                label="AeroPINN prediction" if name == "upper" else None)
    ax.invert_yaxis()
    ax.set_xlabel("x / chord"); ax.set_ylabel("Cp")
    legend = ax.legend(facecolor=P["panel_alt"], edgecolor=P["border"], labelcolor=P["text"])
    ax.set_title(f"m={case['m']:.3f} p={case['p']:.2f} t={case['t']:.3f}, "
                 f"AoA={case['alpha_deg']:.1f}°, Re={case['reynolds']:.0e}")
    plt.tight_layout()
    return fig


def plot_design_space(geometries: list, demo_cases: list, current: dict = None,
                       optimized: dict = None, color_by: str = "split", metric: str = "p_rel_l2"):
    """Same underlying scatter serves both the Design Space view (color_by='split')
    and the Failure Map (color_by='error') -- only 6 held-out points exist, so the
    Failure Map is honestly sparse, not a dense interpolated surface.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _dark_style(fig, ax)

    train_pts = np.array([[g["m"], g["t"]] for g in geometries if g["split"] == "train"])
    held_out_pts = np.array([[g["m"], g["t"]] for g in geometries if g["split"] == "held_out"])

    if color_by == "error":
        errs = np.array([c.get(metric, np.nan) for c in demo_cases])
        pts = np.array([[c["m"], c["t"]] for c in demo_cases])
        ax.scatter(train_pts[:, 0], train_pts[:, 1], c=P["train_geom"], s=40, marker="o",
                   alpha=0.5, label="Training geometry (no error shown)")
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=errs, cmap="YlOrRd", s=90, marker="D",
                        edgecolor=P["text"], linewidth=0.6, label="Held-out (colored by error)")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.045)
        cbar.set_label(f"{metric} (relative L2)", color=P["text_muted"])
        cbar.ax.yaxis.set_tick_params(color=P["text_muted"], labelcolor=P["text_muted"])
        ax.set_title("Failure Map — error over the 6 tested held-out geometries\n"
                     "(sparse, not a dense interpolated surface)")
    else:
        ax.scatter(train_pts[:, 0], train_pts[:, 1], c=P["train_geom"], s=40, marker="o",
                   label="Training geometry")
        ax.scatter(held_out_pts[:, 0], held_out_pts[:, 1], c=P["held_out_geom"], s=60, marker="D",
                   edgecolor=P["text"], linewidth=0.6, label="Held-out geometry")
        ax.set_title("Design Space — training vs. held-out geometries")

    if current is not None:
        ax.scatter([current["m"]], [current["t"]], c=P["current_design"], s=140, marker="*",
                   edgecolor=P["bg"], linewidth=0.8, label="Current design", zorder=10)
    if optimized is not None:
        ax.scatter([optimized["m"]], [optimized["t"]], c=P["optimized_design"], s=140, marker="*",
                   edgecolor=P["bg"], linewidth=0.8, label="Optimized design", zorder=10)

    ax.set_xlabel("m — max camber"); ax.set_ylabel("t — max thickness")
    ax.legend(facecolor=P["panel_alt"], edgecolor=P["border"], labelcolor=P["text"], fontsize=8, loc="best")
    plt.tight_layout()
    return fig


def plot_convergence(history: list):
    """history: list of {iter, cl, cd, cl_over_cd} dicts from a real optimization run."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _dark_style(fig, ax)
    iters = [h["iter"] for h in history]
    vals = [h["cl_over_cd"] for h in history]
    ax.plot(iters, vals, color=P["accent"], lw=1.8)
    ax.set_xlabel("Optimizer iteration"); ax.set_ylabel("CL / CD (predicted)")
    ax.set_title("Inverse design convergence")
    plt.tight_layout()
    return fig
