"""AeroPINN Gradio tab — professional workstation UI (Plan: quizzical-launching-fog.md
Phase 1). Imported by the top-level app.py and called once inside its
`with gr.Blocks(...) as demo:` context to add a tab alongside the flight-delay tabs.

Deploy-path module set: this file + aeropinn/app/{core,theme}.py +
aeropinn/visualization/plots.py + aeropinn/models/pinn.py +
aeropinn/geometry/{naca,sdf}.py + aeropinn/physics/{nondim,forces}.py +
aeropinn/evaluation/ood.py. Everything else in aeropinn/ (solvers, training,
optimization/cma_es) is dev/offline-only. Model/OOD/demo-case artifacts load lazily
on first use (aeropinn.app.core.load_state), never at import time, so this tab never
slows down app.py's startup or the other 5 tabs.

Scope note (see aeropinn/PLAN.md for the full research plan, and the "Phase 1" plan
for this UI/inverse-design pass): one trained geometry-conditioned PINN on ~22 NACA
4-digit airfoils, single training seed. Physics Ablation, Design Space/Failure Map,
and Inverse Design are real but bounded — not the full multi-seed, full-baseline
research study.
"""
import gradio as gr

from aeropinn.app import core
from aeropinn.app.theme import AERO_CSS

# --------------------------------------------------------------------------- helpers

def _status_chip(text: str, level: str = "ok") -> str:
    return f'<span class="status-chip status-chip-{level}">{text}</span>'


def _telemetry_html(items) -> str:
    cells = "".join(
        f'<div class="telemetry-item"><span class="telemetry-label">{label}</span>'
        f'<span class="telemetry-value mono">{value}</span></div>'
        for label, value in items
    )
    return f'<div class="telemetry-row">{cells}</div>'


def _domain_chip_html(m, p, t, alpha, reynolds) -> str:
    status = core.domain_status(m, p, t, alpha, reynolds)
    if not status.available:
        return _status_chip("DOMAIN CHECK N/A", "muted")
    if status.is_ood:
        return _status_chip("OUTSIDE VALIDATED ENVELOPE", "warn")
    return _status_chip("WITHIN VALIDATED DOMAIN", "ok")


def _header_status_html() -> str:
    state = core.load_state()
    dev = str(state.get("device", "n/a")).upper()
    model_chip = _status_chip("MODEL LOADED", "ok") if state.get("ready") else _status_chip("MODEL MISSING", "error")
    return (
        f'<div class="telemetry-row" style="justify-content:flex-end;border:none;background:transparent;padding:0;">'
        f'<div class="telemetry-item"><span class="telemetry-label">Model</span>{model_chip}</div>'
        f'<div class="telemetry-item"><span class="telemetry-label">Device</span>'
        f'<span class="telemetry-value mono">{dev}</span></div>'
        f'<div class="telemetry-item"><span class="telemetry-label">Version</span>'
        f'<span class="telemetry-value mono">{core.VERSION}</span></div>'
        f'</div>'
    )


# --------------------------------------------------------------------------- ANALYSIS

def _run_predict(m, p, t, alpha, reynolds, view):
    from aeropinn.visualization import plots

    result = core.predict_field(m, p, t, alpha, reynolds)
    forces = core.field_forces(result)
    fig = plots.render_field_plot(result, view=view)
    telemetry = _telemetry_html([
        ("CL", f"{forces['cl']:+.4f}"),
        ("CD (pressure)", f"{forces['cd_pressure']:.4f}"),
        ("CL/CD", f"{forces['cl_over_cd']:.2f}"),
        ("ERROR", "N/A (no reference)"),
        ("LATENCY", f"{result.latency_ms:.2f} ms"),
    ])
    chip = _domain_chip_html(m, p, t, alpha, reynolds)
    naca = core.naca_designation(m, p, t)
    return fig, telemetry, chip, naca, result


def _switch_view(result, view):
    from aeropinn.visualization import plots
    if result is None:
        return None
    return plots.render_field_plot(result, view=view)


# ------------------------------------------------------------------------- VALIDATION

def _validation_table():
    ev = core.load_held_out_eval()
    if ev is None:
        return []
    rows = []
    for c in ev["per_case"]:
        rows.append([
            c["geom_id"], f"{c['alpha_deg']:.1f}", f"{c['reynolds']:.0e}",
            f"{c['u_rel_l2']:.3f}", f"{c['v_rel_l2']:.3f}", f"{c['p_rel_l2']:.3f}",
            f"{c['inference_latency_ms']:.2f}",
        ])
    return rows


def _cl_consistency_table():
    """PINN Cp-integrated CL vs. SU2's own force-solver CL, on all held-out cases —
    an internal-consistency check for physics/forces.py, not a repeat of field error.
    """
    from aeropinn.geometry.naca import naca_camber
    from aeropinn.physics.forces import integrate_cl_cd_numpy
    import numpy as np

    state = core.load_state()
    rows = []
    for c in state.get("demo_cases", []):
        yc = naca_camber(np.clip(c["surf_x"], 0, 1), c["m"], c["p"])
        is_upper = c["surf_y"] >= yc
        lower_idx = np.where(~is_upper)[0]
        upper_idx = np.where(is_upper)[0]
        loop_idx = np.concatenate([
            lower_idx[np.argsort(c["surf_x"][lower_idx])],
            upper_idx[np.argsort(-c["surf_x"][upper_idx])],
        ])
        surf_xy = np.stack([c["surf_x"][loop_idx], c["surf_y"][loop_idx]], axis=-1)
        cl_pred_integrated, _ = integrate_cl_cd_numpy(surf_xy, c["surf_cp_pred"][loop_idx], c["alpha_deg"])
        cl_su2, _ = integrate_cl_cd_numpy(surf_xy, c["surf_cp_true"][loop_idx], c["alpha_deg"])
        rel_err = abs(cl_pred_integrated - c["cl_true"]) / (abs(c["cl_true"]) + 1e-6)
        rows.append([
            c["geom_id"], f"{c['alpha_deg']:.1f}",
            f"{cl_pred_integrated:+.4f}", f"{c['cl_true']:+.4f}", f"{cl_su2:+.4f}", f"{rel_err:.1%}",
        ])
    return rows


def _validate_case(case_label):
    from aeropinn.visualization import plots
    c = core.held_out_case(case_label)
    if c is None:
        return None, "No held-out validation cases available yet."
    fig = plots.render_cp_overlay(c, show_reference=True)
    text = (
        f"**Reference (SU2 CFD, independent):** CL={c['cl_true']:.4f}, CD={c['cd_true']:.4f}\n\n"
        f"**Field relative L2 error vs. SU2** — u: {c['u_rel_l2']:.3f}, v: {c['v_rel_l2']:.3f}, p: {c['p_rel_l2']:.3f}"
    )
    return fig, text


# ------------------------------------------------------ COMPARE > Unseen Geometry Challenge

def _challenge_predict(case_label):
    from aeropinn.visualization import plots
    c = core.held_out_case(case_label)
    if c is None:
        return None, "No case selected.", gr.update(visible=False)
    fig = plots.render_cp_overlay(c, show_reference=False)
    return fig, "Prediction generated. This geometry was **not present in training data**.", gr.update(visible=True)


def _challenge_reveal(case_label):
    from aeropinn.visualization import plots
    c = core.held_out_case(case_label)
    if c is None:
        return None, ""
    fig = plots.render_cp_overlay(c, show_reference=True)
    text = (
        f"**Revealed reference (SU2 CFD):** CL={c['cl_true']:.4f}, CD={c['cd_true']:.4f}\n\n"
        f"**Actual error** — u: {c['u_rel_l2']:.3f}, v: {c['v_rel_l2']:.3f}, p: {c['p_rel_l2']:.3f}"
    )
    return fig, text


# ------------------------------------------------------------ COMPARE > Physics Ablation

def _physics_ablation(case_label):
    from aeropinn.visualization import plots
    state = core.load_state()
    if "model_b2" not in state or "demo_cases_b2" not in state:
        return None, None, ("**Physics Ablation model not yet trained.** Run "
                             "`python -m aeropinn.training.train_pinn --lambda-pde 0 "
                             "--out-name pinn_v1_b2_dataonly` to enable this view.")

    c_physics = core.held_out_case(case_label, model_key="demo_cases")
    c_data = core.held_out_case(case_label, model_key="demo_cases_b2")
    if c_physics is None or c_data is None:
        return None, None, "Case not found in both models' evaluation sets."

    fig_physics = plots.render_cp_overlay(c_physics, show_reference=True)
    fig_data = plots.render_cp_overlay(c_data, show_reference=True)
    text = (
        f"| Metric | PHYSICS-GUIDED | DATA-ONLY |\n|---|---|---|\n"
        f"| u rel-L2 | {c_physics['u_rel_l2']:.3f} | {c_data['u_rel_l2']:.3f} |\n"
        f"| v rel-L2 | {c_physics['v_rel_l2']:.3f} | {c_data['v_rel_l2']:.3f} |\n"
        f"| p rel-L2 | {c_physics['p_rel_l2']:.3f} | {c_data['p_rel_l2']:.3f} |\n"
        f"| SU2 reference CL / CD | {c_physics['cl_true']:.4f} / {c_physics['cd_true']:.4f} "
        f"| (same reference) |\n"
    )
    return fig_physics, fig_data, text


# --------------------------------------------------------------------------- DESIGN SPACE

# -------------------------------------------------------------------- INVERSE DESIGN

def _run_inverse_design(case_label):
    from aeropinn.visualization import plots
    from aeropinn.optimization.gradient_design import optimize_gradient
    from aeropinn.optimization.grid_search import optimize_grid

    c = core.held_out_case(case_label)
    if c is None:
        return None, "No case selected.", None, None
    state = core.load_state()
    model, device = state["model"], state["device"]
    start_geom = {"m": c["m"], "p": c["p"], "t": c["t"], "alpha_deg": c["alpha_deg"]}
    reynolds = c["reynolds"]

    trace = optimize_gradient(model, device, start_geom, reynolds, n_iters=200, lr=0.02)
    grid = optimize_grid(model, device, start_geom, reynolds, resolution=10)

    fig = plots.plot_convergence(trace.history)
    final = trace.final_design
    predicted = {"cl": trace.history[-1]["cl"], "cd_pressure": trace.history[-1]["cd"],
                 "cl_over_cd": trace.history[-1]["cl_over_cd"]}

    text = (
        f"**Start** (held-out geometry): m={start_geom['m']:.3f} p={start_geom['p']:.2f} "
        f"t={start_geom['t']:.3f}, CL/CD={trace.history[0]['cl_over_cd']:.2f} (predicted)\n\n"
        f"**Gradient-optimized:** m={final['m']:.3f} p={final['p']:.2f} t={final['t']:.3f} — "
        f"predicted CL/CD={predicted['cl_over_cd']:.2f}\n\n"
        f"**Grid-search cross-check** (independent method, same surrogate): best CL/CD="
        f"{grid.best_cl_over_cd:.2f} at m={grid.best_design['m']:.3f} p={grid.best_design['p']:.2f} "
        f"t={grid.best_design['t']:.3f}\n\n"
        f"⚠️ **These are surrogate predictions only — not yet independently validated.** "
        f"Click VALIDATE WITH REFERENCE below."
    )
    candidate_state = {"design": final, "predicted": predicted, "reynolds": reynolds, "start": start_geom}
    return fig, text, candidate_state, final


def _validate_inverse_design(candidate_state):
    import json
    from aeropinn.evaluation.independent_validation import su2_is_available, validate_candidate, ensure_su2_on_path

    if candidate_state is None:
        return "Run an optimization first."

    ensure_su2_on_path()
    if su2_is_available():
        result = validate_candidate(candidate_state["design"], candidate_state["predicted"],
                                     candidate_state["reynolds"], mesh_fineness=1.0)
        source = "**LIVE independent SU2 run** (this deployment has SU2 installed)"
        elapsed_note = f" — measured {result.elapsed_s:.0f}s" if result.elapsed_s else ""
    else:
        # Production fallback: no live SU2 here, use the nearest precomputed real
        # SU2-validated example (built on the dev machine, never fabricated).
        cache_path = core.CHECKPOINT_DIR / "inverse_design_cache.json"
        if not cache_path.exists():
            return "**NOT YET VALIDATED** — SU2 unavailable in this deployment and no precomputed cache found."
        with open(cache_path) as f:
            cache = json.load(f)

        def dist(entry):
            d = entry["start_geom"]
            q = candidate_state["start"]
            return (d["m"] - q["m"]) ** 2 + (d["p"] - q["p"]) ** 2 + (d["t"] - q["t"]) ** 2

        nearest = min(cache, key=dist)
        v = nearest["su2_validation"]

        class _R:
            pass
        result = _R()
        result.status = v["status"]
        result.su2_reference = v["su2_reference"]
        result.agreement_pct = v["agreement_pct"]
        result.within_tolerance = v["within_tolerance"]
        source = ("**Showing precomputed offline SU2 validation** (SU2 not available in this "
                   "deployment) for the nearest cached example — not this exact optimization run")
        elapsed_note = ""

    if result.su2_reference is None:
        return f"{source}{elapsed_note}\n\n**NOT YET VALIDATED** — SU2 run unavailable or failed."

    verdict = "✓ VALIDATED (within tolerance)" if result.within_tolerance else "⚠️ VALIDATION DISAGREEMENT"
    pred_ratio = candidate_state["predicted"]["cl_over_cd"]
    su2_ratio = result.su2_reference["cl_over_cd"]
    if result.agreement_pct is None or result.agreement_pct < -100 or (pred_ratio * su2_ratio < 0):
        agreement_line = "Agreement: not meaningful (sign mismatch or near-zero reference — see raw values above)"
    else:
        agreement_line = f"Agreement: {result.agreement_pct:.0f}%"
    return (
        f"{source}{elapsed_note}\n\n"
        f"**{verdict}**\n\n"
        f"Surrogate predicted CL/CD: {pred_ratio:.2f}\n\n"
        f"**Independent SU2 CL/CD: {su2_ratio:.2f}** "
        f"(CL={result.su2_reference['cl']:.4f}, CD={result.su2_reference['cd']:.4f})\n\n"
        f"{agreement_line}\n\n"
        f"*The surrogate never validates its own optimum — this number comes from an independent "
        f"CFD run, per the project's hard rule (PLAN.md Section 20).*"
    )


def _design_space(color_by_label, optimized=None):
    from aeropinn.visualization import plots
    state = core.load_state()
    geometries = state.get("geometries", [])
    demo_cases = state.get("demo_cases", [])
    color_by = "error" if color_by_label.startswith("Error") else "split"
    return plots.plot_design_space(geometries, demo_cases, optimized=optimized, color_by=color_by)


# --------------------------------------------------------------------------------- build

def build_tab():
    with gr.Tab("AeroPINN — Airfoil Aerodynamics"):
        gr.HTML(f"<style>{AERO_CSS}</style>")

        with gr.Column(elem_id="aeropinn-root"):
            state = core.load_state()
            if not state.get("ready"):
                gr.Markdown(
                    "### AeroPINN\n\nNo trained checkpoint found yet — this tab activates once "
                    "`aeropinn/experiments/checkpoints/pinn_v1.pt` exists. See `aeropinn/PLAN.md` "
                    "for the full engineering plan."
                )
                return

            with gr.Row(elem_id="aeropinn-header"):
                gr.Markdown("**AEROPINN**", elem_classes="app-name")
                header_status = gr.HTML(_header_status_html())

            gr.Markdown(
                "Geometry-conditioned physics-informed surrogate for 2D NACA 4-digit airfoil flow. "
                "One trained model, no retraining per shape — predicts flow around geometries never "
                "seen during training. *Phase-1 scope: single training seed, "
                "6 held-out validation geometries, one physics-vs-data-only ablation. "
                "Full plan: `aeropinn/PLAN.md`.*"
            )

            optimized_design_state = gr.State(None)

            with gr.Tabs(elem_id="aeropinn-nav"):

                # ------------------------------------------------------------- ANALYSIS
                with gr.Tab("ANALYSIS"):
                    result_state = gr.State(None)
                    with gr.Row():
                        with gr.Column(scale=1):
                            m_in = gr.Slider(0.0, 0.06, value=0.02, step=0.005, label="m — max camber")
                            p_in = gr.Slider(0.10, 0.60, value=0.40, step=0.01, label="p — camber position")
                            t_in = gr.Slider(0.08, 0.20, value=0.12, step=0.005, label="t — max thickness")
                            alpha_in = gr.Slider(-4.0, 8.0, value=2.0, step=0.5, label="Angle of attack (deg)")
                            re_in = gr.Slider(1.0e4, 1.0e5, value=5.0e4, step=1.0e3, label="Reynolds number")
                            naca_display = gr.Textbox(label="NACA designation", interactive=False,
                                                       value=core.naca_designation(0.02, 0.40, 0.12))
                            domain_chip = gr.HTML(_domain_chip_html(0.02, 0.40, 0.12, 2.0, 5.0e4))
                            predict_btn = gr.Button("RUN PREDICTION", variant="primary")
                        with gr.Column(scale=2):
                            view_radio = gr.Radio(["geometry", "velocity", "cp"], value="cp", label="View")
                            field_plot = gr.Plot(label="Flow field")
                            telemetry_html = gr.HTML()

                    predict_btn.click(
                        _run_predict, inputs=[m_in, p_in, t_in, alpha_in, re_in, view_radio],
                        outputs=[field_plot, telemetry_html, domain_chip, naca_display, result_state],
                    )
                    view_radio.change(_switch_view, inputs=[result_state, view_radio], outputs=[field_plot])
                    for slider in (m_in, p_in, t_in, alpha_in, re_in):
                        slider.change(_domain_chip_html, inputs=[m_in, p_in, t_in, alpha_in, re_in], outputs=[domain_chip])
                    m_in.change(core.naca_designation, inputs=[m_in, p_in, t_in], outputs=[naca_display])
                    p_in.change(core.naca_designation, inputs=[m_in, p_in, t_in], outputs=[naca_display])
                    t_in.change(core.naca_designation, inputs=[m_in, p_in, t_in], outputs=[naca_display])

                # ----------------------------------------------------------- VALIDATION
                with gr.Tab("VALIDATION"):
                    gr.Markdown("#### Held-out geometry scoreboard\nAggregate error across all 6 geometries "
                                "never seen during training (`held_out_eval.json`).")
                    gr.Dataframe(
                        headers=["Geometry", "AoA (deg)", "Re", "u rel-L2", "v rel-L2", "p rel-L2", "Latency (ms)"],
                        value=_validation_table(), interactive=False,
                    )
                    gr.Markdown("#### CL consistency check\nPINN's own Cp-integrated CL vs. SU2's independent "
                                "force-solver CL — validates `physics/forces.py` before it's used anywhere else "
                                "(e.g. Inverse Design's objective).")
                    gr.Dataframe(
                        headers=["Geometry", "AoA (deg)", "CL (PINN pred, integrated)", "CL (SU2, true)",
                                 "CL (SU2 Cp, integrated)", "Rel. error"],
                        value=_cl_consistency_table(), interactive=False,
                    )
                    gr.Markdown("#### Cp overlay")
                    case_dropdown_v = gr.Dropdown(choices=core.case_labels(), value=core.case_labels()[0] if core.case_labels() else None,
                                                   label="Held-out case")
                    validate_btn = gr.Button("Compare vs. hidden CFD reference")
                    with gr.Row():
                        cp_plot_v = gr.Plot()
                        validate_text_v = gr.Markdown()
                    validate_btn.click(_validate_case, inputs=[case_dropdown_v], outputs=[cp_plot_v, validate_text_v])

                # --------------------------------------------------------------- COMPARE
                with gr.Tab("COMPARE"):
                    with gr.Tabs():
                        with gr.Tab("Unseen Geometry Challenge"):
                            gr.Markdown("Select a held-out airfoil, run the prediction, then reveal the hidden "
                                        "CFD reference to see the actual error.")
                            case_dropdown_c = gr.Dropdown(choices=core.case_labels(),
                                                           value=core.case_labels()[0] if core.case_labels() else None,
                                                           label="Held-out case")
                            gr.HTML(_status_chip("NOT PRESENT IN TRAINING DATA", "warn"))
                            predict_c_btn = gr.Button("RUN PREDICTION")
                            challenge_plot = gr.Plot()
                            challenge_text = gr.Markdown()
                            reveal_btn = gr.Button("REVEAL REFERENCE", visible=False, variant="primary")
                            predict_c_btn.click(_challenge_predict, inputs=[case_dropdown_c],
                                                 outputs=[challenge_plot, challenge_text, reveal_btn])
                            reveal_btn.click(_challenge_reveal, inputs=[case_dropdown_c],
                                              outputs=[challenge_plot, challenge_text])

                        with gr.Tab("Physics Ablation"):
                            gr.Markdown("DATA-ONLY neural network vs. PHYSICS-GUIDED (PDE-loss) neural network, "
                                        "same held-out airfoil, same training data. Equal visual weight — "
                                        "neither is biased.")
                            case_dropdown_a = gr.Dropdown(choices=core.case_labels(),
                                                           value=core.case_labels()[0] if core.case_labels() else None,
                                                           label="Held-out case")
                            ablation_btn = gr.Button("Compare")
                            with gr.Row():
                                fig_physics_out = gr.Plot(label="Physics-guided")
                                fig_data_out = gr.Plot(label="Data-only")
                            ablation_text = gr.Markdown()
                            ablation_btn.click(_physics_ablation, inputs=[case_dropdown_a],
                                                outputs=[fig_physics_out, fig_data_out, ablation_text])

                # ---------------------------------------------------------- DESIGN SPACE
                with gr.Tab("DESIGN SPACE"):
                    gr.Markdown("Training vs. held-out geometries in (camber, thickness) space, or the same "
                                "geometries colored by held-out error (Failure Map — 6 sparse points, "
                                "not a dense interpolated surface). Also shows the Inverse Design tab's "
                                "optimized candidate, once one has been run.")
                    color_by_radio = gr.Radio(["Split (train / held-out)", "Error magnitude"],
                                               value="Split (train / held-out)", label="Color by")
                    design_space_plot = gr.Plot(value=_design_space("Split (train / held-out)"))
                    color_by_radio.change(_design_space, inputs=[color_by_radio, optimized_design_state],
                                           outputs=[design_space_plot])

                # -------------------------------------------------------- INVERSE DESIGN
                with gr.Tab("INVERSE DESIGN"):
                    gr.Markdown(
                        "**Objective: maximize CL/CD.** Starting point must be a held-out geometry "
                        "(never a training geometry, per PLAN.md Section 19). Constraints: t ≥ 0.09, "
                        "m/p/t within the validated envelope (enforced via sigmoid reparameterization — "
                        "every iterate stays inside the box, not just penalized for leaving it), α fixed "
                        "at the starting geometry's value this phase.\n\n"
                        "**The surrogate never validates its own optimum** (PLAN.md Section 20) — "
                        "VALIDATE WITH REFERENCE always calls an independent SU2 run (live if this "
                        "deployment has SU2, otherwise the nearest precomputed real SU2 result)."
                    )
                    candidate_state = gr.State(None)
                    start_dropdown = gr.Dropdown(choices=core.case_labels(),
                                                  value=core.case_labels()[0] if core.case_labels() else None,
                                                  label="Starting geometry (held-out only)")
                    run_opt_btn = gr.Button("RUN OPTIMIZATION", variant="primary")
                    convergence_plot = gr.Plot(label="Convergence (gradient method)")
                    opt_text = gr.Markdown()
                    validate_ref_btn = gr.Button("VALIDATE WITH REFERENCE")
                    validate_ref_text = gr.Markdown()

                    run_opt_btn.click(_run_inverse_design, inputs=[start_dropdown],
                                       outputs=[convergence_plot, opt_text, candidate_state, optimized_design_state])
                    validate_ref_btn.click(_validate_inverse_design, inputs=[candidate_state],
                                            outputs=[validate_ref_text])
                    optimized_design_state.change(_design_space, inputs=[color_by_radio, optimized_design_state],
                                                   outputs=[design_space_plot])
