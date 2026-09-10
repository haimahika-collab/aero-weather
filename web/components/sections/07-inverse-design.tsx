"use client";

import { useState } from "react";
import { SectionShell } from "@/components/SectionShell";
import { Figure } from "@/components/Figure";
import { ConvergenceChart } from "@/components/ConvergenceChart";
import { DesignSpaceScatter, type GeometryPoint } from "@/components/DesignSpaceScatter";
import { StatusChip } from "@/components/StatusChip";
import inverseDesign from "@/data/inverse_design.json";
import designSpace from "@/data/design_space_error.json";

export function InverseDesignSection() {
  const [idx, setIdx] = useState(0);
  const run = inverseDesign[idx];
  const v = run.su2_validation;

  return (
    <SectionShell id="07-inverse-design" number="07" title="Inverse Design" typography="engineering">
      <p className="mb-4 max-w-2xl leading-relaxed" data-typography="editorial">
        Objective: maximize CL/CD. Starting geometry is always a held-out airfoil, never a training
        geometry. Constraints: t ≥ 0.09, m/p/t within the validated envelope, enforced by construction
        (sigmoid-reparameterized optimization variables), not by penalty. The surrogate never validates
        its own optimum — every candidate is independently re-evaluated with SU2.
      </p>

      <select
        value={idx}
        onChange={(e) => setIdx(Number(e.target.value))}
        className="mono text-xs bg-panel-alt border border-border rounded-[4px] px-2 py-1.5 mb-4 w-full sm:w-auto max-w-full"
      >
        {inverseDesign.map((r, i) => (
          <option key={i} value={i}>
            Run {i + 1}: start m={r.start_geom.m.toFixed(3)} p={r.start_geom.p.toFixed(2)} t={r.start_geom.t.toFixed(3)}
          </option>
        ))}
      </select>

      <Figure
        number="07"
        title="Inverse Design Convergence"
        provenance="real"
        sourceNote="aeropinn/experiments/checkpoints/inverse_design_cache.json"
        caption={`Gradient-based optimization, ${run.gradient_history.length} real iterations, starting from held-out geometry m=${run.start_geom.m.toFixed(3)} p=${run.start_geom.p.toFixed(2)} t=${run.start_geom.t.toFixed(3)}. Dashed red line: independent SU2-measured CL/CD for the final candidate.`}
      >
        <ConvergenceChart
          history={run.gradient_history}
          su2ClOverCd={v.su2_reference ? v.su2_reference.cl_over_cd : null}
        />
      </Figure>

      <div className="border border-border rounded-[var(--radius)] bg-panel p-4 my-6" data-typography="editorial">
        <div className="flex items-center gap-3 mb-2 flex-wrap">
          <StatusChip
            text={v.within_tolerance ? "VALIDATED" : "VALIDATION DISAGREEMENT"}
            level={v.within_tolerance ? "ok" : "warn"}
          />
          <span className="mono text-xs text-text-muted">independent SU2 re-evaluation</span>
        </div>
        <p className="text-[15px] leading-relaxed mb-2">
          Surrogate predicted CL/CD:{" "}
          <span data-numeric>{run.gradient_predicted.cl_over_cd.toFixed(2)}</span>. Independent SU2
          CL/CD: <span data-numeric>{v.su2_reference ? v.su2_reference.cl_over_cd.toFixed(2) : "N/A"}</span>.
        </p>
        {!v.within_tolerance && (
          <p className="text-[15px] leading-relaxed text-text-muted">
            This is reported as a finding, not hidden: the optimizer found a design the surrogate
            over- or under-rated near the edge of its validated envelope. Independent validation
            caught it — that is exactly what the hard validation rule is for. No candidate from this
            project has yet been confirmed as a genuine aerodynamic improvement.
          </p>
        )}
      </div>

      <Figure
        number="06"
        title="Model Error Across Design Space"
        provenance="real"
        sourceNote="geometries.json ⋈ held_out_eval.json per_case, joined by geometry"
        caption="Held-out error (pressure field relative L2) across the 6 tested geometries — sparse, not a dense interpolated surface. Training geometries shown for reference, without error (they were seen during training)."
      >
        <DesignSpaceScatter points={designSpace as GeometryPoint[]} colorBy="error" />
      </Figure>
    </SectionShell>
  );
}
