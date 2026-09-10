# AeroPINN: Geometry-Conditioned Physics-Informed Neural Network for Parametric Airfoil Flow Prediction and Inverse Design

**Engineering Research Plan — integrated into `haimahika-collab/aero-weather`**

---

## Context

You previously built `aero-weather` (live at aero-weather-production.up.railway.app): a Gradio app that generates synthetic weather/operational data and trains sklearn models (Decision Tree, small NN) plus some PyTorch time-series forecasters, to predict flight delays. You now want to move from that tabular/statistical ML work toward a genuinely physics-grounded aerodynamics project — AeroPINN — using a physics-informed neural network to predict flow around airfoils and, eventually, help design them.

You asked me (acting as tech lead) to read your full research brief (`AeroPINN_Claude_Research_Prompt.txt`), treat it as the master spec, do the literature/engineering research needed to ground it, and produce a defensible plan — then confirmed you want the result added directly to the existing `aero-weather` GitHub repo and its live Railway site, as a new tab alongside the existing flight-delay demo, rather than as a separate project or a separate deployment.

Three parallel research passes were run to ground this plan: (1) a literature/novelty audit, (2) CFD ground-truth generation and PINN software-stack research, (3) inverse-design and OOD/uncertainty-method research. The single most important finding: **the core "geometry-conditioned PINN for airfoils" idea is not novel** — arXiv 2412.01954 (Ghosh et al., NeurIPS 2024 ML4PS workshop) already does approximately this at small scale (8 airfoils). This plan is deliberately framed around what's still genuinely open — a rigorous benchmark against neural-operator baselines plus independently-validated inverse design — rather than a false "first-ever" claim.

---

## Repository Integration Decision

`aero-weather` is a single lightweight Gradio app (`app.py`, plus `data.py`/`models.py`/`train.py` for a separate PyTorch time-series demo), deployed as one Railway service, dependencies already include `torch>=2.0.0`. It has no CFD, no PDE solvers, no heavy compute in its deploy path. AeroPINN's forward-model training and CFD generation (SU2 sweeps, PINN/DeepONet training) are compute-heavy and must **never** run inside the Railway web container.

**Decision:** Add a new `aeropinn/` subpackage to the repo, and add a new **"AeroPINN" tab** to the existing `app.py` (it already uses a multi-tab Gradio pattern — this becomes tab 5, alongside Data Generator / Train & Analyze / Live Predictor / ML Academy). The deployed tab only loads a small pretrained checkpoint and runs CPU inference (milliseconds for an 8×128 MLP or a similarly small DeepONet) — all CFD generation and model training happens offline (your machine or a burst cloud instance), and only the resulting checkpoints + small cached demo assets are committed to the repo. `requirements.txt` gains a few small additions (e.g. `scipy` for Mahalanobis distance) but not SU2/OpenFOAM/DeepXDE — those are dev-time-only tools, never Railway dependencies. Note the repo's default branch is unusually named `initial-commit`, not `main` — worth confirming before pushing.

---

## 1. Executive Technical Assessment

The premise "geometry-conditioned PINN for airfoils" is scientifically sound but **not novel** — it has already been demonstrated at workshop scale (Ghosh et al., arXiv 2412.01954, NeurIPS 2024 ML4PS). Operator-learning methods (U-Net, GNN, DeepONet, neural fields) have been solving the *same* held-out-geometry generalization problem, often more accurately and faster, since at least 2019 (Thuerey U-Net) and as recently as 2024 (Aero-Nef, Geom-DeepONet). Any framing of this project as "inventing" geometry-conditioned physics-informed surrogates will not survive review.

What is genuinely open, and what this plan targets, is a **rigorous, reproducible, head-to-head benchmark**: geometry-conditioned RANS-PINN vs. a matched-budget neural-operator baseline, evaluated on the field's own standardized unseen-geometry protocol (AirfRANS), with an honest data-efficiency study and a downstream inverse-design capability that is independently re-validated (never self-validated). The interesting scientific question is not "does this work" (partially shown already) but "does the physics-informed inductive bias actually buy anything — accuracy, data efficiency, or generalization robustness — relative to a purely data-driven operator of comparable capacity and compute budget, and is that advantage still there in the standard laminar-limited regime this project can actually afford?" The honest expected outcome, based on the literature survey, is a mixed or negative result for pure accuracy and a positive result (if any) for data efficiency and/or differentiability for design. The project must be designed so that a negative result is still a complete, reportable, defensible piece of work — not a failure.

Feasibility for a solo builder is real but tight. Full turbulent RANS-PINN training (as in 2412.01954) is compute- and engineering-heavy; TSONN (2501.01165) needed 4.6 days to train a parametric laminar PINN at Re 100–5000, an order of magnitude below flight Reynolds numbers. This plan therefore explicitly recommends **narrowing physics scope** (Section 6) rather than chasing full-Re RANS, while still benchmarking against AirfRANS-derived data for the data-driven/operator baselines where turbulence closure is unavoidable.

---

## 2. Literature / Novelty Audit (Condensed)

| Ref | Contribution | Gap relative to AeroPINN |
|---|---|---|
| Ghosh et al. 2412.01954 (2024) | Geometry (shape params + local SDF) + Re/U∞ conditioned PINN, RANS k-ε, 8 NACA airfoils, unseen airfoils **and** unseen Re | Workshop-scale (8 shapes), no inverse design, no benchmark vs. AirfRANS or operator learning — the exact gap this project fills |
| Sun et al. 2401.07203 (2024) | Shape-param PINN + inverse design, inviscid only | No viscous physics, held-out protocol unclear |
| ScienceDirect S0021999124005333 (2024) | High-dim parametric inviscid PINN | Inviscid, paywalled |
| TSONN 2501.01165 (2025) | Parametric laminar PINN, many UIUC shapes/AoA/Re jointly | Re 100–5000 only (non-aerospace regime), 4.6-day training, no inverse design |
| Cambridge S0045782523001664 (2023) | AD+L-BFGS shape optimization through PINN | Per-shape optimization trajectories, not a pretrained geometry-general surrogate |
| **AirfRANS** 2212.07564 (2022, NeurIPS D&B) | 1000 OpenFOAM RANS sims, NACA 4/5-digit, Re 2–6M, explicit unseen-geometry split, GNN/CNN baselines | **The standard benchmark this project must evaluate against**; fully-turbulent Re range only |
| CFD-GCN 2007.04439 (2020) | GCN + differentiable coarse SU2 solver, explicit unseen-airfoil generalization | Non-physics-informed, motivates "pure DL doesn't generalize" concern |
| Thuerey U-Net 1810.08217 (2019) | ~1500 UIUC airfoils, held-out shape generalization | Purely data-driven, predates PINN work by years, strong non-PINN baseline |
| Aero-Nef 2407.19916 (2024) | SDF+INR+hypernetwork, ~3x better than MeshGraphNet/GraphUNet on unseen shapes | Not physics-informed |
| Geom-DeepONet 2403.14788 (2024) | SDF+point-cloud DeepONet for unseen 3D shapes | Not physics-informed, 3D not 2D |
| "Do Neural Operators Forget Geometry?" 2605.05862 (2026) | Documents genuine geometric-fidelity loss in DeepONet/FNO under shape shift | Motivates (as hypothesis, not fact) a PINN advantage under geometry extrapolation |
| Fixed-geometry PINN 2107.10711 (2021), condition-only PINN 2410.18917 (2024) | Establish condition-parametric (not geometry-parametric) PINNs as old news | Not novel by themselves |
| Turbine blade PINN 2605.07131 (2026) | Boundary-progressive training, curvature-weighted residual loss, ~20 configs | Tangential precedent for training tricks, narrow blade family |

**Bottom line:** condition-parametric PINNs = established; geometry+condition PINNs for NACA airfoils = demonstrated at small scale; operator-learning methods already routinely beat or match this on held-out-geometry accuracy, often with far less engineering effort and no physics loss at all. The open space is methodological rigor and honest comparison, not the core idea.

---

## 3. Exact Defensible Research Gap

No existing work combines, in one open reproducible pipeline: (a) viscous (not inviscid) physics-informed training, (b) joint geometry+condition conditioning (m, p, t, α, Re), (c) a rigorous held-out-geometry/held-out-Re study using the field's standardized benchmark protocol (AirfRANS), (d) a matched-budget neural-operator baseline tested under the identical protocol to directly probe the "Forgetting Hypothesis" (2605.05862), and (e) inverse design executed through the *same pretrained geometry-general* surrogate (not per-shape re-optimization, not inviscid-only), with independent re-validation of proposed designs.

---

## 4. Final Research Questions (Refined/Hedged)

**RQ1 (forward model, primary):** *Within a laminar-to-low-transitional Reynolds-number envelope where a steady incompressible NS-based PINN loss is physically defensible, does a geometry-conditioned PINN achieve equal or better held-out-geometry field/coefficient accuracy than a matched-parameter-budget, matched-CFD-data-budget neural-operator baseline (DeepONet or FNO), and does that advantage (if any) persist, shrink, or reverse under geometry extrapolation (outside the training shape envelope) versus interpolation (inside it)?*

**RQ2 (data efficiency, secondary):** *Does the PDE-residual loss allow the geometry-conditioned PINN to degrade more gracefully than the equivalent data-driven surrogate (B2) as labeled CFD data is reduced (100/50/25/10%)?*

**RQ3 (inverse design, tertiary, gated on RQ1):** *Can gradient-based optimization through the trained differentiable PINN surrogate, starting from an arbitrary held-out geometry, find a design whose predicted CL/CD improvement is confirmed (within a stated tolerance) by an independent solver, and how does it compare in efficiency/robustness to brute-force grid search and a derivative-free baseline (CMA-ES) over the same low-dimensional design space?*

These are deliberately hedged: RQ1 does **not** presuppose the PINN wins; RQ2 is the more defensible differentiator per the literature; RQ3 is explicitly conditioned on RQ1/RQ2 producing a forward model trustworthy enough to design from.

---

## 5. Testable Hypotheses (Falsifiable)

- **H1 (accuracy):** Held-out-geometry normalized L2 field error of PINN (B3) ≤ operator baseline (B4) at matched data budget. *Rejection outcome:* B4 error is statistically indistinguishably better or better — acceptable, reportable result consistent with 2605.05862's uncertainty and with Thuerey/Aero-Nef precedent.
- **H2 (extrapolation robustness):** The PINN's relative error degradation from interpolation-split to extrapolation-split geometries is smaller than the operator baseline's degradation (testing the Forgetting Hypothesis directly). *Rejection outcome:* both degrade comparably, or PINN degrades worse — also reportable, would falsify H2 straightforwardly.
- **H3 (data efficiency):** At 10% labeled data, PINN (B3) L2 field error increase relative to 100%-data PINN is smaller than B2's equivalent increase. *Rejection outcome:* physics loss provides no measurable data-efficiency benefit — a legitimate negative result given TSONN/2412.01954 don't report this comparison.
- **H4 (inverse design validity):** The independently-re-validated CL/CD of the PINN-optimizer's proposed geometry improves over the starting held-out geometry by more than the re-validated improvement found by grid search, within the same surrogate-query budget. *Rejection outcome:* gradient-based optimization is not better than grid search in this low-dim space — plausible, since the spec itself notes brute-force is a legitimate baseline here.
- **H5 (PDE residual as diagnostic, not proof):** Models with low PDE residual can still have high field error (loss-imbalance artifact) — must be actively demonstrated, not just asserted, via at least one ablation/seed showing this decoupling.

All five hypotheses have a defined, reportable failure mode. None assumes the PINN wins by default.

---

## 6. Physical Assumptions and Validity Envelope

Full turbulent RANS-PINN training across a geometry family (as in 2412.01954) is the most literature-aligned option but is compute-prohibitive for a solo project on the available hardware (Section 22) and introduces a turbulence closure (k-ε/k-ω SST) as an *additional* learned/embedded model inside the PDE residual — compounding uncertainty sources and making failure diagnosis much harder. **Recommendation: constrain the physics-informed (PINN) training regime to steady, 2D, incompressible, laminar flow (Re ≲ 5×10⁴–1×10⁵), pre-stall AoA (roughly −4° to +8°, geometry-dependent), attached flow only.** This is a deliberate, justified scope reduction, not evasion:

- Steady-state assumption breaks down near/past stall (vortex shedding, unsteady separation) — excluded by AoA bound.
- Laminar assumption is invalid at flight Reynolds numbers (10⁶–10⁷, where AirfRANS lives) — this is the single biggest scope compromise and must be stated up front, repeatedly, in any write-up. It means direct field-level comparison to AirfRANS ground truth is **not apples-to-apples**; AirfRANS is used only for (a) pretraining/testing the *data-driven baselines* (B2, B4) at their native turbulent regime, and (b) a clearly-labeled secondary/stretch turbulent-regime PINN experiment using an eddy-viscosity closure, attempted only if the laminar-regime study is complete and time remains (Section 26).
- Incompressible assumption requires M∞ ≲ 0.3, enforced as an input bound.
- Pressure is only defined up to a constant in incompressible NS — a reference point (far-field p=0) must be fixed (Section 8).
- Domain restricted to bounded geometry family: NACA 4-digit, m ∈ [0, 0.06], p ∈ [0.1, 0.6], t ∈ [0.08, 0.20] — a defensible bounded design envelope, not the full NACA space.

This means the PINN forward-model study and the AirfRANS-based benchmark study are **two separate, clearly labeled regimes**, not one continuous experiment — a point that must be stated explicitly in any results write-up to avoid an invalid implied comparison.

---

## 7. Governing Equations

Nondimensionalized steady 2D incompressible Navier–Stokes, characteristic length L = chord c, characteristic velocity U = U∞:

x* = x/c, y* = y/c, u* = u/U∞, v* = v/U∞, p* = p/(ρU∞²), Re = ρU∞c/μ

Continuity:
∂u*/∂x* + ∂v*/∂y* = 0

x-momentum:
u*∂u*/∂x* + v*∂u*/∂y* = −∂p*/∂x* + (1/Re)(∂²u*/∂x*² + ∂²u*/∂y*²)

y-momentum:
u*∂v*/∂x* + v*∂v*/∂y* = −∂p*/∂y* + (1/Re)(∂²v*/∂x*² + ∂²v*/∂y*²)

with Re defined on chord c (not on a fixed reference geometry), consistent across all geometries in the training family so Re is comparable across shapes. All star notation dropped below for brevity; the network is trained and evaluated entirely in nondimensional variables. Pressure coefficient for reporting: Cp = (p − p∞)/(½ρU∞²) = 2p* (since p* already carries the ½ρU∞² normalization convention used above — this must be fixed consistently in code and documented once, as unit/normalization drift is a common, embarrassing PINN bug).

---

## 8. Boundary Conditions

- **Airfoil surface Γ_s(m,p,t):** no-slip + no-penetration, u=v=0.
- **Far-field Γ_∞** (recommend circular/rectangular domain radius ≈ 15–20c, informed by SU2 C-grid practice): u→cos α, v→sin α (freestream direction rotated by AoA), p→0 (pressure reference fixed here, gauge choice).
- **Inflow:** subset of Γ_∞ upstream, Dirichlet velocity = freestream.
- **Outflow:** subset of Γ_∞ downstream, recommend soft/Neumann-consistent treatment (zero-gradient normal stress) rather than hard Dirichlet, matching standard CFD outflow practice and avoiding over-constraining the pressure field.
- **Pressure reference:** fixed at far-field (p∞=0); without this the incompressible pressure field is only defined up to an additive constant and the network can learn an arbitrary offset — must be enforced as a soft constraint or hard output shift.

**How the boundary moves with (m,p,t):** this is the central technical difficulty the spec flags, and it is where most naive implementations fail silently (network trained on Cartesian collocation points that don't actually sit on/around the *current* geometry's surface as m,p,t vary within a batch).

Geometry-handling strategies compared:

| Strategy | Pros | Cons | Verdict |
|---|---|---|---|
| Body-fitted coordinate transform (map each airfoil to unit circle/canonical domain) | Exact geometry conformance | Requires a valid conformal/algebraic map per shape, expensive to make differentiable and general across the full (m,p,t) family, adds another source of error | Reject — too much engineering for early stage |
| Cartesian collocation + geometry-as-extra-input only (no SDF) | Simplest | BC loss must be evaluated exactly on Γ_s(m,p,t); with only scalar params the network has no *local* signal for "how far am I from the surface at this point," empirically harder to enforce sharp near-wall behavior | Reject as sole strategy — insufficient per 2412.01954's own design choice |
| Point-cloud / mesh-based geometry embedding (GNN-style) | Matches operator-learning literature strength | Abandons the PINN collocation-based training loop's simplicity, conflates the PINN and operator-baseline architectures | Reject for the PINN; appropriate for the operator baseline itself |
| **Signed-distance function (SDF) as additional per-point input, computed analytically per NACA 4-digit shape at each collocation point** | Exact for this specific shape family (NACA 4-digit has an analytic thickness/camber definition, so exact and cheap surface distance is computable without meshing), directly matches 2412.01954's already-validated approach, gives the network explicit near-wall locality without a coordinate transform, cheap to batch across mixed geometries in one training step | Only cheap because of the analytic NACA parameterization — would not generalize to arbitrary/free-form shapes | **Recommended** |

**Recommended geometry-handling strategy:** SDF(x,y; m,p,t) computed analytically (distance to the NACA camber+thickness curve, signed by upper/lower surface), added as an extra scalar input alongside (x,y,m,p,t,α,Re). Surface/no-slip BC loss evaluated at points sampled directly on Γ_s (SDF=0 by construction) rather than only enforced implicitly through the SDF field. This mirrors 2412.01954's validated choice rather than inventing a new mechanism — the novelty in this project is the benchmarking, not the geometry encoding trick.

---

## 9. Geometry Representation Strategy

Baseline input: X = [x, y, SDF(x,y,m,p,t), m, p, t, α, Re]. Rejected alternatives already covered in Section 8. Additional candidate features evaluated as an **ablation**, not baked into the default architecture (Section 15): local surface curvature at nearest surface point, surface-normal components at nearest surface point. Rationale for treating these as ablations rather than defaults: 2412.01954 shows SDF+raw params is already sufficient to generalize to unseen shapes; adding curvature/normal features is an empirical question of diminishing/uncertain marginal return, not a settled requirement — testing it as an ablation is more honest than assuming it helps.

---

## 10. Network Architecture

Justified against SA-PINN (McClenny & Braga-Neto, JCP 2022), NTK-weighting (Wang/Yu/Perdikaris), and Fourier-feature literature, and against the field-standard depth/width range (6–10 layers, 64–256 units):

- **Depth/width:** 8 hidden layers × 128 units. Reasoning: the input dimensionality here (8 scalars including geometry+condition) is higher than a typical single-geometry PINN (2–3 inputs), and the function class needs to represent a *manifold* of solutions, not one field — favors width toward the upper-middle of the standard range rather than the low end used in single-airfoil papers. Not going to 256 units / >10 layers by default because published parametric-PINN work at similar scope (TSONN, 2412.01954) do not report needing it and compute budget is solo-scale (Section 22); treat width/depth as a tunable swept in a small ablation, not hand-picked once and frozen.
- **Activation:** tanh as default (smooth, C^∞, standard for 2nd-order PDE residuals since viscous stress terms need twice-differentiable activations — ReLU is explicitly disallowed). Swish/sine (SIREN-style) evaluated as an ablation given documented spectral-bias mitigation near sharp leading-edge curvature, but tanh kept as default for reproducibility with the bulk of PINN literature.
- **Input encoding:** Fourier-feature / positional encoding on (x,y) only (not on geometry/condition scalars) to combat spectral bias near the leading edge and thin boundary layer. Geometry/condition scalars passed raw (after normalization), not Fourier-encoded, since they are low-frequency conditioning signals, not spatial coordinates needing high-frequency resolution.
- **Normalization:** all inputs normalized to roughly [-1,1] or zero-mean/unit-variance per dimension (m,p,t,α,Re on very different native scales — this is a real bug risk if skipped); outputs (u,v,p*) normalized using training-set statistics, de-normalized at inference.
- **Initialization:** Glorot/Xavier (standard for tanh networks, preserves activation variance).
- **Autodiff:** PyTorch autograd with `create_graph=True` for the 2nd-order residual derivatives — this is the concrete reason MATLAB's `dlgradient` (single-scalar-loss-oriented) is rejected as the delivery framework (Section 24).
- **Optimizer:** Adam (1e-3, cosine or step decay) for the bulk of training, switched to L-BFGS for late-stage fine-tuning — standard two-phase PINN optimization practice, since Adam is robust early but L-BFGS's better final-convergence behavior matters once residuals are small and second-order curvature information helps.
- **Sampling/batching:** mixed batches per step containing (a) PDE collocation points sampled across the *interior* of multiple randomly-drawn training geometries simultaneously (not one geometry per batch — this is what makes it a parametric/manifold learner rather than N independent fixed-geometry PINNs), (b) BC points on Γ_s and Γ_∞ for those same geometries, (c) supervised CFD data points if L_data is active. Residual-based adaptive sampling (resample high-residual regions more densely, e.g. near leading edge / stagnation point) added as an explicit technique to try, per SA-PINN precedent, not assumed necessary from step one.
- **Loss weighting:** SA-PINN-style self-adaptive per-point weights for BC terms (known to help near-wall vs. far-field imbalance) plus NTK-informed *global* term weighting (λ_PDE, λ_BC, λ_data) recomputed periodically (e.g. every N epochs) rather than fixed constants — directly motivated by the "PDE residual can be low while field error is high" failure mode (H5) that fixed/miscalibrated weights are known to cause.
- **Convergence criteria:** stop on plateau of validation-split field error (not training loss) over a patience window, cross-checked against PDE/BC residual norms — using training loss alone as stopping criterion is explicitly rejected since it doesn't detect the H5 failure mode.

---

## 11. Full Mathematical Loss Formulation

L_total = λ_PDE·L_PDE + λ_BC·L_BC + λ_data·L_data + λ_aux·L_aux

**L_PDE** (mean squared residual over N_f collocation points sampled across all training geometries/conditions in the batch):

L_PDE = (1/N_f) Σᵢ [ (∂u/∂x + ∂v/∂y)ᵢ² + (u∂u/∂x + v∂u/∂y + ∂p/∂x − (1/Re)∇²u)ᵢ² + (u∂v/∂x + v∂v/∂y + ∂p/∂y − (1/Re)∇²v)ᵢ² ]

**L_BC** (sum of surface, far-field, pressure-reference terms, N_s/N_∞/N_p points respectively):

L_BC = (1/N_s)Σ(u,v)ᵢ² [no-slip, on Γ_s] + (1/N_∞)Σ‖(u,v)ᵢ − (cosα, sinα)‖² [freestream, on Γ_∞ inflow] + (1/N_p)Σ pᵢ² [pressure reference at far-field]

**L_data** (supervised regression to CFD ground truth at N_d labeled points, only present for B2/B3 where labeled data is used):

L_data = (1/N_d) Σⱼ [ (u−û)ⱼ² + (v−v̂)ⱼ² + (p−p̂)ⱼ² ]

**L_aux** (optional, only if empirically justified — e.g. surface Cp-matching term or a small L2 weight-decay regularizer). This term is **not** allowed to be an arbitrary "physics-looking" addition per the spec's explicit constraint — the only admitted L_aux candidates are (a) a direct Cp-matching term derived from the same momentum equation already in L_PDE evaluated on the surface, or (b) standard weight regularization for numerical stability. If unused, λ_aux = 0 and this is stated explicitly rather than silently omitted.

Loss weights λ_* initialized at O(1) parity and re-balanced via the NTK/SA-PINN procedure in Section 10, logged every evaluation step so weight drift itself becomes a diagnosed, reported quantity (directly supports detecting the H5 failure mode).

---

## 12. CFD/Reference-Data Generation Strategy

**Decision: SU2 as primary in-house CFD generator for the laminar-regime training/validation family; AirfRANS as a pre-existing, non-regenerated dataset used only for the turbulent-regime data-driven baselines and the (optional, stretch) turbulent PINN experiment; XFOIL as a cheap pre-filter and coefficient-only sanity check, never as field ground truth.**

Reasoning: SU2's config-file-per-case structure is trivially scriptable (Python/Jinja templating) across the (m,p,t,α,Re) sweep; existing open NACA→SU2 C-grid generators (NACA-SU2-MESH, mikfoil) remove most of the meshing engineering burden that would otherwise dominate a solo timeline; SU2_DEF/SU2_GEO give shape-deformation tooling reusable later for adjoint-based cross-checks. OpenFOAM (what AirfRANS itself used) has more mature automated meshing for very heterogeneous shape libraries but is heavier to script per-case for a bounded, analytically-parameterized shape family like NACA 4-digit — not worth the extra engineering here given SU2 already covers the need.

**Concrete pipeline (runs entirely offline/local — never in the Railway deploy path):**
1. Sample geometry family (Section 6 bounds) via Latin Hypercube over (m,p,t) — target ~150–300 distinct geometries for the laminar study (small enough to be feasible solo, large enough to have real held-out splits).
2. For each geometry, generate C-grid mesh (structured, far-field ~15–20c, wall-normal clustering appropriate for laminar y+ — far less aggressive than RANS y+≈1 requirement, a genuine cost saving from the laminar scope decision) via templated script.
3. Sweep α ∈ [−4°,8°] (5–7 values) × Re ∈ [1×10⁴, 1×10⁵] (3–4 values) per geometry — full factorial where affordable, else LHS over the joint (geometry,condition) space if factorial is too large for the compute budget.
4. Run SU2 laminar (no turbulence model) steady solver; convergence = 5–6 orders residual drop **and** Cauchy convergence on CL/CD (both criteria required, not either/or — a common false-convergence trap).
5. Mesh-independence check on a representative subset (3–5 geometries) at 2–3 mesh densities before committing to the production mesh density — this is a non-negotiable CFD-standards item the spec calls out explicitly and it is often skipped under time pressure; budget real time for it (Section 25).
6. Extract u,v,p on a fixed set of interior sample points (for L_data/eval) plus surface Cp, CL, CD.
7. XFOIL run in parallel on the same (m,p,t,α,Re) grid, restricted to cases within its validity envelope (attached, pre-stall, Re 5×10⁵–2×10⁷ nominal — note this is *above* our laminar Re window, so XFOIL's applicability here is limited and must be explicitly checked case-by-case, not assumed) — used only as an independent coefficient-level sanity check on the SU2 runs themselves, and later as the fast independent check in inverse-design re-validation (Section 20).
8. AirfRANS loaded via `pip install airfrans` unmodified, used strictly for B2/B4 turbulent-regime training/eval and clearly labeled as a separate regime from the SU2 laminar dataset in every result table.
9. Only the final processed tensors (interior sample points, Cp/CL/CD, trained checkpoints) are committed to `aero-weather`'s `aeropinn/data/processed/` and `aeropinn/experiments/checkpoints/`; raw meshes/solver working directories stay local and gitignored.

---

## 13. Geometry-Based Train/Validation/Test Split Methodology

Split at the **geometry** level, never at the point or condition level, to prevent leakage (a network that has seen other points on the same airfoil at nearby α/Re can trivially interpolate even at a "held-out" point — this is not a real generalization test).

- **Training set:** ~70% of sampled geometries, all their α/Re sweep points.
- **Interpolation test set:** ~15% of geometries whose (m,p,t) lie *inside* the convex hull of the training geometries' parameter values (nearest-neighbor and hull-membership check on (m,p,t) to confirm) — tests whether the manifold is learned smoothly inside the seen envelope.
- **Extrapolation test set:** ~15% of geometries whose (m,p,t) lie *outside* or at the boundary of the training convex hull (e.g. highest-camber or highest-thickness shapes held out entirely) — tests true extrapolation, expected to be materially harder, and required to be **reported separately** from the interpolation result, never pooled into one aggregate "test accuracy" number (pooling here would hide exactly the failure mode this project is designed to probe).
- Validation set (for early stopping / hyperparameter selection) carved from the training-geometry pool as a further geometry-level split, not from the test sets, to keep test sets fully untouched until final evaluation.
- Same split reused identically across B0–B4 for fairness (Section 14).
- α/Re values within a held-out geometry's sweep are themselves reported both as "seen α/Re range, unseen geometry" and, if compute allows, "unseen α/Re range, unseen geometry" (compounding extrapolation) as a stretch analysis.

---

## 14. Baseline Models

| ID | Model | Purpose | Fairness controls |
|---|---|---|---|
| B0 | Reference CFD (SU2 laminar / AirfRANS OpenFOAM) | Ground truth | N/A |
| B1 | Fixed-geometry PINN, one model per test geometry (feasible subset only, e.g. 3–5 held-out geometries) | Upper-bound sanity check: how much accuracy is "lost" by generalizing vs. training bespoke | Same architecture family, same physics loss, same collocation density per case |
| B2 | Purely data-driven parametric surrogate (same MLP capacity/inputs as B3, no L_PDE/L_BC, λ_PDE=λ_BC=0) | Isolates the value of the physics loss term | Identical architecture, identical CFD data budget, identical splits |
| B3 | Geometry-conditioned PINN (proposed) | Primary model under test | — |
| B4 | Neural-operator baseline (DeepONet, or FNO if a fixed-grid formulation fits the C-grid data better) | Tests whether operator learning already solves this without physics loss — directly addresses the "not novel" finding and the Forgetting Hypothesis | Same train/test geometry split, same CFD data budget, comparable parameter count, identical metrics suite |

B4 architecture choice reasoning: DeepONet's branch/trunk structure maps naturally onto (geometry+condition) → branch, (x,y) → trunk, without requiring a fixed regular grid, making it a more natural match to the SU2 C-grid/scattered evaluation points than FNO (which wants a uniform grid and would require an extra regridding step, introducing its own error source). FNO added only as a stretch/optional second operator baseline if time permits, not required for the core comparison.

Neural-operator baseline is **required**, not optional, given the literature audit — omitting it would leave the project's central claim untestable.

---

## 15. Ablation Studies

All ablations run on B3 (and where noted, mirrored on B4 for a fair operator-side comparison), same splits as Section 13:

1. **Remove L_PDE** (λ_PDE=0) — reduces B3 toward B2; quantifies physics-loss contribution directly (complements the standalone B2 comparison with a within-model toggle).
2. **Remove L_data** (λ_data=0, pure physics-informed, zero labeled interior points, only BCs) — tests whether physics+BCs alone can reconstruct the field without any CFD supervision; expected to degrade substantially, informative failure case.
3. **Remove geometry conditioning** (drop m,p,t from input, keep only x,y,α,Re) — collapses toward a fixed-shape-family-averaged model; sanity check that geometry conditioning is actually being used (a real risk: network could learn to ignore weakly-influential inputs).
4. **Remove SDF feature** (keep raw m,p,t only, no SDF) — directly tests the Section 8/9 geometry-representation decision rather than assuming it.
5. **Reduce number of training geometries** (e.g. 150 → 75 → 30) — generalization-vs-shape-diversity curve.
6. **Reduce labeled CFD points** — see Section 17 (dedicated experiment, not just an ablation line item).
7. **Vary PDE collocation density** (e.g. 2k/8k/32k points per geometry) — cost/accuracy trade-off, informs final production setting.
8. **Vary λ_PDE:λ_data ratio** (fixed-ratio sweep, e.g. 10:1, 1:1, 1:10) alongside the adaptive-weighting default — establishes whether adaptive weighting (Section 10/11) actually beats a well-chosen fixed ratio, an honest check since adaptive schemes are not guaranteed wins.

---

## 16. Evaluation Metrics

- Normalized L2 field error (separately for u, v, p) over interior evaluation points.
- MAE / RMSE for velocity components and pressure.
- Cp distribution error (L2 and max-abs) along the surface, at matched x/c stations.
- CL relative error (%), reported **only** where CFD convergence/mesh-independence checks passed cleanly — flagged/excluded otherwise, not silently included.
- CD relative error (%) — reported with explicit caveat that CD is far more sensitive to viscous/mesh resolution than CL, per standard CFD practice; lower confidence weighting in any summary claim.
- CL/CD ratio error, since this is the inverse-design objective quantity.
- PDE residual norms and BC residual norms — reported as **diagnostic panel items only**, explicitly never cited alone as evidence of correctness (per H5 and the spec's explicit constraint).
- Inference latency (measured, wall-clock, stated hardware — including a specific measurement on Railway's actual CPU instance for the deployed demo, separate from any GPU dev-machine number) — no projected/theoretical numbers.
- Training cost (wall-clock hours, GPU-hours) per model class.
- CFD computational cost (wall-clock hours for the SU2 sweep) — reported once as a fixed cost baseline for the "surrogate is cheaper after upfront investment" argument.
- Variability across ≥3 training seeds for B2/B3/B4 (mean ± std on primary metrics) — required, not optional, since single-seed PINN results are well known in the literature to be noisy/non-representative.
- Explicitly **not** relying on R² as a headline metric (spec constraint) — R² reported at most as a secondary sanity item.

---

## 17. Data-Efficiency Experiment

Train B2 and B3 at 100%, 50%, 25%, 10% of the labeled CFD training points (subsampled at the *point* level within the fixed training-geometry set — geometry split itself unchanged, per Section 13, so this isolates data density, not shape diversity). λ_PDE held fixed for B3 across all four data levels (i.e., physics loss density is *not* reduced when labeled data is reduced — this is the whole point of the experiment: does physics loss compensate for missing labels). Report interpolation- and extrapolation-test error vs. data fraction as a curve for both B2 and B3, 3 seeds each. H3 is confirmed only if B3's degradation slope is measurably shallower than B2's, with error bars that don't overlap at 10%; otherwise H3 is rejected and reported as such.

---

## 18. Interpolation vs. Extrapolation Experiment Design

Directly built on the Section 13 split. Primary result table structure (required format, not optional):

| Model | Interp. L2 error | Interp. Cp error | Interp. CL err % | Extrap. L2 error | Extrap. Cp error | Extrap. CL err % | Δ(extrap−interp) |
|---|---|---|---|---|---|---|---|
| B1 (per-geom, subset) | ... | | | n/a (by construction) | | | |
| B2 | | | | | | | |
| B3 | | | | | | | |
| B4 | | | | | | | |

The **Δ(extrap−interp) column is the primary quantity for testing H2** (Forgetting Hypothesis) — reported for every model, with confidence intervals across seeds, not just point estimates. A secondary qualitative panel (rendered field-error heatmaps for 2–3 representative extrapolation geometries) is required to visually communicate *where* error concentrates (typically near leading edge / suction peak for shape extrapolation) — numbers alone tend to hide this and a skeptical reviewer will ask for it.

---

## 19. Inverse-Design Methodology

**Objective:** maximize CL/CD (or CL at fixed CD, as a secondary formulation) subject to:
- t ≥ t_min (structural/manufacturability floor, e.g. t ≥ 0.09)
- m, p, t within the validated training envelope (Section 6) — enforced as box constraints, not merely hoped for
- α within the validated pre-stall range
- Starting point: an arbitrary **held-out** geometry (not a training geometry) — required, since starting from a training-set shape would understate the difficulty and overstate the surrogate's real design usefulness.

**Optimizer comparison** (all three required, per spec, not just the gradient method):
1. **Gradient-based through the differentiable PINN** (Adam or L-BFGS on design params, backprop through the frozen trained surrogate) — cheap per-iteration, but only finds a local optimum and is sensitive to initialization; box constraints enforced via clipping/reparameterization (sigmoid-squashed params), not penalty terms, to guarantee validity of every candidate evaluated.
2. **Brute-force grid search** over the low-dimensional (m,p,t) space (and α if included) — legitimate and cheap here specifically *because* the design space is low-dimensional (≤4 dims), serves as an easy independent sanity check on the gradient method's result, not a strawman.
3. **Derivative-free / global method (CMA-ES)** as the middle ground — more robust to local optima than gradient descent, far cheaper than CFD-based optimization, tests whether the gradient method's local optimum is actually close to a global one.
4. CFD-based optimization (SU2 adjoint or finite-difference gradient) run on a **small validation subset only** (e.g. 2–3 starting geometries) — not the primary optimization loop (too expensive to run at full scale solo), but essential as an independent third-party check on whether the surrogate-based optimizers are finding genuinely good designs or artifacts of surrogate error.

H4 is evaluated by comparing the **independently re-validated** (Section 20) CL/CD improvement across methods 1–3, at matched surrogate-query budgets, with method 4 as the trust anchor on the small subset.

---

## 20. Independent Validation of Optimized Designs (Hard Rule)

**Rule, stated explicitly and enforced in code, not just in prose:** any geometry proposed by an optimizer running against the PINN/operator surrogate is *never* reported as an improvement until it has been re-evaluated by a solver that was not part of the optimization loop. Concretely:

1. Every candidate geometry that survives optimization (the final proposed design from each of methods 1–3 in Section 19) is automatically queued for XFOIL re-evaluation first (fast, seconds-scale, immediate feedback) **if** it falls within XFOIL's validity envelope (attached, pre-stall) — if it does not, this is itself flagged as a red flag on the proposed design, not silently skipped.
2. The top 1–2 candidates overall are re-run through full SU2 CFD (same mesh/convergence standards as Section 12) — this is the authoritative check.
3. A design is only claimed as "validated" if the SU2-computed CL/CD improvement is within a pre-stated tolerance (e.g. ±15%, chosen before running the check, not after seeing the result) of the surrogate's predicted improvement, **and** the improvement itself is positive and outside CFD noise/mesh-sensitivity bounds established in Section 12's mesh-independence check.
4. Any case where the independent solver disagrees materially with the surrogate is reported as a finding, not discarded — this is exactly the kind of result the spec requires ("report failed experiments").
5. The optimizer's own surrogate is architecturally prevented from being used as its own final judge: the re-validation code path is a separate module (`aeropinn/evaluation/independent_validation.py`) that never calls the trained PINN/operator model, only XFOIL/SU2 wrappers.

---

## 21. Uncertainty / OOD Strategy

**Recommended: small deep ensemble (3–5 independently seeded models) for epistemic uncertainty + Mahalanobis (or kNN) distance check in the low-dimensional (m,p,t,α,Re) input parameter space.**

Reasoning against alternatives: MC dropout is explicitly **rejected** — its fixed-Bernoulli-mask posterior is documented to trend toward near-uniform uncertainty rather than properly growing under extrapolation, which would silently defeat the exact purpose of this mechanism. Evidential deep learning is **rejected** — harder optimization landscape, incomplete aleatoric/epistemic disentanglement, not worth the added complexity for a solo project. Ensemble cost is trivial at this project's scale (3–5x training cost of a single already-small MLP/operator model). Mahalanobis/kNN distance is specifically favorable *because* the conditioning space here is low-dimensional (5 scalars: m,p,t,α,Re) — distance-based OOD famously breaks down in high-dimensional image-like spaces but is well-behaved and interpretable here, directly matching domain precedent (SmOOD, arXiv 2209.03438, smoothness-based OOD specifically for aircraft-design surrogates).

PDE-residual magnitude is retained only as a **tertiary diagnostic panel item**, never a standalone OOD/correctness signal, per H5 — a poorly-weighted PINN can show artificially low residual due to loss-imbalance optimization artifacts, which is precisely why Section 10/11 build in periodic weight-rebalancing diagnostics rather than trusting residual alone.

**Operationalization:** every inference-time query computes (a) ensemble variance on (u,v,p) at query points → single scalar summary (e.g. mean predictive std over the field) and (b) Mahalanobis distance of the query (m,p,t,α,Re) to the training-set distribution in that 5-D space (cheap: covariance matrix is precomputed offline and shipped with the checkpoint, so the deployed Gradio tab only does a fast `scipy`/`numpy` matrix-vector computation, no heavy dependency). Either exceeding a pre-calibrated threshold (calibrated on the held-out extrapolation set, not on training data) triggers a visible **"outside validated domain"** flag in the demo (Section 27) — a warning, never a silently-degraded prediction.

---

## 22. Computational Requirements (Solo-Realistic)

- **CFD generation (SU2 laminar sweep):** ~150–300 geometries × ~20 (α,Re) combos ≈ 3,000–6,000 cases; laminar 2D cases converge in minutes each on a few CPU cores → estimate 100–300 CPU-hours total, parallelizable across cores/cloud burst instances over ~1–2 weeks of calendar time run in the background. Mesh-independence subset adds a small fixed overhead (~20–40 extra runs).
- **AirfRANS:** no generation cost, ~few GB download, used as-is.
- **PINN/operator training (B1–B4, 3 seeds each):** single consumer/prosumer GPU (e.g. RTX 3090/4090 class or equivalent cloud instance) sufficient given the laminar-scope reduction and modest network size (8×128 MLP); estimate 2–8 GPU-hours per B2/B3 training run, more for B4 depending on DeepONet branch/trunk sizing — total training budget across all baselines/ablations/seeds realistically **80–200 GPU-hours**, i.e. days, not the multi-day-per-run TSONN reported for a harder joint-Re-range problem, precisely because this scope is intentionally narrower.
- **Storage:** CFD field data (u,v,p on interior points + surface Cp per case) at this scale is on the order of a few GB to a few tens of GB — kept local/offline, not committed to the repo (Section 24); only final checkpoints (small, a few MB each) and cached demo assets are committed.
- **Deployed inference (Railway):** CPU-only, single small forward pass (8×128 MLP or similarly small DeepONet) — sub-10ms expected but must be *measured on the actual Railway instance*, not assumed from GPU dev numbers, per Section 16's honesty requirement.
- **Inverse-design/optimization runs:** negligible additional compute (surrogate inference is the dominant cost, milliseconds per query); CFD re-validation subset (Section 20) adds a small, bounded number of extra SU2 runs.
- **Realistic total solo timeline implication:** this budget is achievable within the 6–8 week roadmap (Section 25) only if the laminar-scope narrowing (Section 6) is respected and the turbulent/AirfRANS-PINN extension (Section 26) is treated as optional stretch, not core-path.

---

## 23. Main Technical Failure Modes and Mitigations

1. **PDE loss and data loss fight each other, network converges to a low-total-loss but physically wrong or over-smoothed solution** (classic PINN gradient-pathology failure). *Mitigation:* NTK/SA-PINN adaptive weighting (Section 10), monitor loss-term *ratios* not just total loss, H5-style deliberate check for the low-residual/high-error decoupling.
2. **Geometry conditioning is effectively ignored** — network finds it easier to fit an "average" flow field and treat m,p,t as near-irrelevant. *Mitigation:* Section 15 ablation #3 (remove geometry conditioning) as a required sanity check; inspect predicted-field sensitivity to m,p,t directly (finite-difference the trained model's output w.r.t. geometry inputs) as a diagnostic.
3. **Extrapolation performance is simply bad for all models**, PINN included — the Forgetting Hypothesis paper's finding could apply to *both* PINN and operator baselines, not just operator learning. *Mitigation:* this is an acceptable, reportable outcome (H2 rejection case), not something to paper over.
4. **CFD ground truth itself is under-converged or mesh-dependent**, contaminating every downstream comparison. *Mitigation:* mandatory mesh-independence subset (Section 12 step 5) and dual convergence criteria — the single most common way "PINN vs CFD" comparisons quietly become meaningless.
5. **Laminar-regime scope reduction makes the whole study less relevant to real aerospace Reynolds numbers**, inviting fair criticism of "why should I care about Re 10⁴–10⁵." *Mitigation:* stated explicitly and repeatedly (Section 6), framed as a deliberate finishability trade-off with a labeled stretch extension (Section 26).
6. **DeepONet/FNO baseline is under-tuned relative to the PINN**, producing an unfair comparison. *Mitigation:* fix an explicit, time-boxed tuning budget applied equally to B3 and B4 before final comparison runs; document the tuning process/budget so this bias is auditable.
7. **Inverse-design optimizer exploits surrogate error** — finds a geometry where the surrogate over-predicts CL/CD due to a local surrogate blind spot rather than genuine aerodynamic improvement. *Mitigation:* Section 20's hard independent-validation rule exists specifically for this.
8. **Single-seed noise mistaken for a real effect** in any headline comparison. *Mitigation:* mandatory ≥3-seed reporting (Section 16), no single-run claims in the final write-up's headline table.
9. **XFOIL misused as if it were field-resolved ground truth** by mistake under time pressure. *Mitigation:* XFOIL's role is fixed and documented (Section 12) as filter/sanity-check/design-recheck only; code-level separation (XFOIL wrapper never feeds L_data, only used in `evaluation/independent_validation.py` and the pre-CFD-filter step).
10. **The demo tab silently breaks the existing flight-delay tabs** by adding a heavy dependency, a slow import, or a Railway memory/cold-start regression. *Mitigation:* keep AeroPINN's runtime dependencies minimal (torch is already present; avoid adding SciPy-heavy or GPU-only libraries to the production path), lazy-load the checkpoint only when the AeroPINN tab is first opened, and smoke-test the full existing app (all 4 original tabs + the new one) locally before every deploy.

---

## 24. Recommended Software Architecture

**Stack decision:** PyTorch (directly, with DeepXDE as the PINN scaffolding layer for offline training where it saves engineering time) — not MATLAB — for the delivered pipeline. MATLAB's Deep Learning Toolbox `dlgradient` is built around single-scalar-loss backward passes; the 2nd-order, multi-input derivatives needed for the 2D NS residual are materially more cumbersome there than PyTorch's `create_graph=True` autograd, which is what essentially all published PINN reference implementations (DeepXDE, Modulus) actually use. `aero-weather` already depends on `torch>=2.0.0`, so this is also the path of least friction for the existing repo. MATLAB's role, if any, is limited to your own early personal prototyping — explicitly **not** part of the delivered/reproducible pipeline, avoiding a two-language reproducibility burden.

DeepXDE vs. NVIDIA Modulus/PhysicsNeMo: DeepXDE recommended as the default — lightweight, sufficient custom-residual and hard/soft BC support for this scope, and (crucially) a **dev/training-only dependency**, never imported by the deployed `app.py`. Modulus is heavier/production-cluster-oriented — revisit only if DeepXDE's parametric-geometry workaround becomes a genuine scaling bottleneck.

**Repository layout** — `aeropinn/` added as a subpackage of the existing repo, existing files untouched except `app.py` (new tab) and `requirements.txt` (small additions):

```
aero-weather/                    # existing repo root
  app.py                          # EXISTING — gains one import + one new gr.Tab("AeroPINN", ...)
                                   # calling aeropinn.app.build_tab(); nothing else in app.py changes
  data.py, models.py, train.py    # existing flight-delay / time-series code — untouched
  requirements.txt                # + scipy (Mahalanobis), no SU2/OpenFOAM/DeepXDE (dev-only)
  README.md                       # + a short "AeroPINN" section describing the new tab
  aeropinn/
    configs/          # YAML configs per experiment (geometry bounds, Re/AoA ranges,
                       # architecture, loss weights, seeds) — one file = one reproducible run
    geometry/          # NACA 4-digit parametric generator, analytic SDF, mesh templating
    physics/            # nondimensional NS residual definitions, BC residual definitions
    models/              # MLP (B2/B3), DeepONet/FNO (B4), ensemble wrapper
    solvers/              # SU2 config templating + runner, XFOIL wrapper, AirfRANS loader
                          # (dev-only: never imported by app.py)
    training/              # training loops, adaptive loss weighting (SA-PINN/NTK), seeds
                          # (dev-only: never imported by app.py)
    evaluation/            # metrics (Section 16), split logic (Section 13),
                          # independent_validation.py (Section 20 hard-rule module, isolated)
    optimization/          # gradient-based / grid-search / CMA-ES inverse-design routines
    visualization/         # field plots, Cp plots, error heatmaps, latency logging
    app/                    # build_tab() — Gradio Blocks fragment imported by top-level app.py;
                            # sliders (m,p,t,alpha,Re), field/Cp/streamline plots, OOD flag,
                            # measured-latency display. ONLY this module + models/ + a checkpoint
                            # are on the deploy path.
    experiments/             # run scripts/logs, local-only (gitignored except final summary)
    experiments/checkpoints/ # small trained checkpoints (MLP/DeepONet state_dict, a few MB) —
                            # committed; this is what the deployed tab actually loads
    data/processed/          # small cached demo assets (a handful of precomputed geometries'
                            # fields for the "hidden reference" comparison in the demo) — committed
    data/raw/                 # gitignored: raw SU2/OpenFOAM meshes, solver working dirs, AirfRANS cache
    tests/                     # unit tests: residual correctness (finite-diff check vs autograd),
                            # SDF correctness, split leakage checks, unit-normalization checks
```

`.gitignore` additions: `aeropinn/data/raw/`, SU2/OpenFOAM working-directory patterns, local experiment logs — keeps the repo lean despite the CFD-heavy dev workflow.

Reproducibility: fixed seeds per config, experiment logs (loss curves, weight-rebalancing history) checkpointed per run, config hash stored alongside results to prevent silent config drift between reported numbers and code state.

---

## 25. 6–8 Week Implementation Roadmap

- **Week 0 (setup):** Clone `aero-weather` locally, confirm default branch (`initial-commit`), create the `aeropinn/` package skeleton (empty modules per Section 24 layout), add `.gitignore` entries, open a feature branch.
- **Week 1:** Geometry module (NACA generator + analytic SDF), nondimensionalization utilities, unit tests for both. SU2 case templating + single-geometry smoke test run (local/offline).
- **Week 2:** Full SU2 sweep automation across sampled (m,p,t,α,Re); launch batch runs in background; mesh-independence subset run in parallel. AirfRANS download/cache + loader.
- **Week 3:** CFD sweep completes/validated (residual+Cauchy convergence, mesh-independence confirmed); data extraction pipeline (interior points, surface Cp/CL/CD) finalized; geometry-level train/interp-test/extrap-test split implemented with leakage checks (unit-tested).
- **Week 4:** B2 (data-only) and B3 (PINN) architecture + training loop implemented (DeepXDE/PyTorch), including adaptive loss weighting; first end-to-end training runs on a small geometry subset for debugging.
- **Week 5:** Full B2/B3 training across all seeds/splits; B1 (fixed-geometry PINN subset) run for sanity-bound comparison; begin B4 (DeepONet) implementation.
- **Week 6:** B4 training complete, matched tuning budget applied; full metrics suite (Section 16) computed across B1–B4; interpolation/extrapolation result tables (Section 18) assembled; data-efficiency experiment (Section 17) run.
- **Week 7:** Ablation suite (Section 15) run; ensemble + Mahalanobis OOD calibration (Section 21); inverse-design optimizers (gradient/grid/CMA-ES) implemented and run; independent re-validation (Section 20) executed on top candidates.
- **Week 8:** Build `aeropinn/app/build_tab()`, wire it into the existing `app.py` as a new tab, smoke-test the *entire* app (all 5 tabs) locally, measure real Railway-CPU inference latency post-deploy, results write-up, failure-mode/negative-result documentation (Section 23 items checked against actual observed results), final reproducibility pass (config hashes, checkpoint archival, README update), push and deploy.

Buffer note: Weeks 2–3 (CFD generation) are the most schedule-risk-prone step — if slipping, the first scope cut is the extrapolation-set size and ablation count (Section 26), not the mesh-independence check itself.

---

## 26. Minimum Viable Research Version (Fallback Scope)

If full scope is infeasible in the timeline:
- Shrink geometry family to ~60–80 geometries (still enough for a real geometry-level split, per Section 13's proportions).
- Drop B1 (fixed-geometry per-airfoil PINNs) entirely except 2 cases, purely as a qualitative sanity anchor, not a full baseline row.
- Reduce data-efficiency experiment (Section 17) to two points (100% and 25%) instead of four.
- Reduce ablation suite (Section 15) to the three highest-value items: remove-L_PDE, remove-geometry-conditioning, remove-SDF.
- Keep B4 (operator baseline) and the interpolation/extrapolation split as **non-negotiable core**, since they are what makes the project's central claim (Section 3/4) testable at all — cutting these would collapse the project back into "yet another geometry-conditioned PINN," reproducing the non-novel case (Section 2).
- Keep the inverse-design independent-validation rule (Section 20) non-negotiable even in MVP scope, on a reduced candidate set (1 optimizer method instead of 3, gradient-based only, still grid-search-checked as the one required comparison since it's cheap).
- Defer the turbulent-regime/AirfRANS-PINN stretch experiment entirely; AirfRANS is used only for B2/B4 in MVP scope.
- Defer curvature/normal-feature ablation (Section 9) entirely.
- The new Gradio tab and Railway deploy still ship even in MVP scope — it's cheap once a single trained B3 checkpoint exists, and it's the most visible/demonstrable artifact of the project.

---

## 27. Competition Demonstration Plan (2–3 minutes, live on aero-weather-production.up.railway.app)

1. Show reference CFD result (SU2 field plot) for a known training-set airfoil — establish what "ground truth" looks like and how it was produced (mesh, solver, convergence — one sentence, sets credibility).
2. Select a **held-out** airfoil from a dropdown/slider (m,p,t combination not in training set) — state explicitly on screen that this geometry was never seen during training.
3. Run the B3 surrogate on this geometry — no retraining step, emphasize this visually (no "training..." spinner, just inference).
4. Display predicted velocity-magnitude field, pressure field, Cp distribution, streamlines side-by-side with a (pre-computed, hidden-until-now) reference CFD result for the same held-out geometry.
5. Show quantitative error panel: L2 field error, Cp error, CL/CD relative error (if within validated regime) — actual numbers, not just visual similarity.
6. Show **measured** inference latency (milliseconds, on the actual deployed Railway CPU instance) next to the CFD wall-clock time for the same case — the efficiency argument stated in real numbers, not implied.
7. Interactively change geometry sliders (m,p,t) and/or α/Re live — show the field updating in near-real-time, reinforcing "one model, many geometries."
8. Deliberately push a slider **outside** the validated envelope — show the OOD flag (Section 21) triggering, ensemble-variance/Mahalanobis-distance indicator turning red — demonstrate the system knows what it doesn't know, rather than silently extrapolating.
9. Switch to inverse-design mode: state an objective (maximize CL/CD) starting from the same held-out geometry, with constraints shown on screen (t≥t_min, envelope bounds).
10. Run the gradient-based optimizer live (or show a fast-forwarded/cached version if real-time optimization is too slow for a live demo — stated honestly as such, not disguised as live if it isn't).
11. Show the proposed optimized geometry, then show its **independent re-validation** result (XFOIL and/or SU2 re-run number) side-by-side with the surrogate's prediction — explicitly state the agreement/disagreement, closing the loop on "the surrogate does not validate itself."

Framing throughout: engineering evidence (numbers, measured latency, independent re-checks, explicit OOD flags) over visual polish — matches the NVIDIA Omniverse digital-twin precedent of always showing surrogate/reference/difference together, never surrogate output alone as truth.

---

## 28. Claims That May Reasonably Be Made (If Results Support Them)

- "A geometry- and condition-conditioned PINN, trained once on a bounded laminar NACA 4-digit family, produces near-real-time flow predictions for held-out geometries without retraining, at measured latency X ms vs. CFD wall-clock Y."
- "Within this laminar, pre-stall, bounded-geometry envelope, the PDE-residual loss provides [a measured, quantified] data-efficiency advantage over an equivalent purely data-driven surrogate at reduced label fractions" — *only* if H3 is actually confirmed by the data-efficiency experiment.
- "In a matched-budget comparison, [PINN/operator] achieved lower held-out-geometry error under [interpolation/extrapolation], with the gap [narrowing/widening] under extrapolation" — stated as the actually-observed direction, whichever way the data falls.
- "A design proposed by gradient-based optimization through the differentiable surrogate was independently confirmed by [XFOIL/SU2] to improve CL/CD by Z%, within the pre-stated validation tolerance" — only for candidates that actually pass Section 20's check.
- "This is, to our knowledge, the first evaluation of a geometry-conditioned PINN against a matched-budget neural-operator baseline under a standardized held-out-geometry protocol, alongside independently-validated inverse design from an unseen starting geometry" — the actual, narrow, defensible novelty claim.

---

## 29. Claims That Must Be Avoided

- "First geometry-conditioned PINN for airfoils" — **false**, 2412.01954 predates this.
- "First PINN generalizing across geometries without retraining" — **false**, non-PINN work (Thuerey 2019, CFD-GCN 2020) predates this by years.
- "PINNs generalize better than operator learning" — **not established either way** in the literature; must never be asserted as a general conclusion, only as this specific study's measured, scoped result.
- "Differentiable PINN surrogate enables inverse design" as a general novelty claim — **not new by itself** (Cambridge 2023, 2401.07203 already show this); only the specific combination with geometry-general pretraining + independent re-validation is novel.
- "Low PDE residual proves the prediction is physically correct" — explicitly false per H5 and Section 21; residual is a diagnostic, not proof.
- Any claim that the surrogate's own re-query of itself constitutes validation of an optimized design (Section 20 hard rule).
- Any implied claim that the laminar-regime results transfer to flight Reynolds numbers without the explicitly-labeled, separately-scoped turbulent extension (Section 6, Section 26).
- Any accuracy claim pooling interpolation and extrapolation results into one aggregate number (Section 13/18).
- Any latency claim not backed by an actual measured number on stated hardware (spec's explicit "no invented speedups" constraint).

---

## 30. Proposed Paper Title

**"Does Physics-Informed Conditioning Help Airfoil Surrogates Generalize? A Benchmarked Comparison of Geometry-Conditioned PINNs and Neural Operators for Held-Out Geometry Prediction and Validated Inverse Design"**

---

## 31. Proposal Abstract (150–200 words)

Physics-informed neural networks (PINNs) conditioned on airfoil geometry and operating conditions have recently been shown to predict flow around unseen NACA airfoils without retraining, but no existing study benchmarks this approach against the neural-operator methods (DeepONet, FNO, GNN-based surrogates) that have solved the same held-out-geometry generalization problem, often without any physics loss, for several years. AeroPINN addresses this gap directly. We train a geometry-conditioned PINN and a matched-budget DeepONet baseline on a bounded, laminar, pre-stall NACA 4-digit family, using an analytic signed-distance function to handle geometry-dependent boundaries, and evaluate both under a rigorous geometry-level interpolation/extrapolation split. We measure whether the PDE-residual loss yields quantifiable data-efficiency gains as labeled CFD data is reduced, and whether it changes robustness under geometry extrapolation — testing, rather than assuming, the recently-raised concern that operator methods can "forget geometry" under shape shift. A validated forward model is then used for gradient-based inverse design from an arbitrary unseen airfoil, with every proposed design independently re-evaluated by XFOIL/CFD before any performance claim is made. We report honest, possibly negative, results alongside a reproducible open pipeline.

---

## Verification Plan

- **Unit tests** (`aeropinn/tests/`): NS-residual autograd values checked against finite-difference approximation on a known analytic flow (e.g. potential-flow-around-cylinder closed form) before ever training on real geometry; SDF sign/zero-crossing correctness on a known NACA shape; geometry-split leakage check (assert no test geometry's (m,p,t) appears in training set); input-normalization round-trip check.
- **CFD validation**: mesh-independence check (Section 12 step 5) must show CL/CD changing by <1–2% between the two finest mesh levels before that mesh density is used for the production sweep; XFOIL cross-check on a handful of attached-flow cases as an independent sanity check on early SU2 results.
- **Training validation**: for each of B1–B4, training/validation loss curves and loss-term-ratio logs must be inspected (Section 10's H5 check) — a model with suspiciously low PDE residual but high held-out field error should be caught here, before it reaches the result tables.
- **End-to-end app check**: after wiring the new tab into `app.py`, run the full app locally (`python app.py`) and manually exercise all 5 tabs (the 4 existing ones plus AeroPINN) to confirm no regression; then deploy to Railway and re-verify the same 5 tabs load and the AeroPINN tab's measured latency and OOD-flag behavior work against the live deployment, not just localhost.
- **Final claim check**: before writing any result into Section 28/29-style claims in the eventual report, cross-check it against the exact hedged hypothesis (Section 5) it's answering and confirm it isn't one of the explicitly-forbidden Section 29 claims.
