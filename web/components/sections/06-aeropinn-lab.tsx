"use client";

import { useState } from "react";
import { SectionShell } from "@/components/SectionShell";
import { Figure } from "@/components/Figure";
import { CpChart } from "@/components/CpChart";
import { DesignSpaceScatter, type GeometryPoint } from "@/components/DesignSpaceScatter";
import { ExperimentMetadataBlock } from "@/components/ExperimentMetadataBlock";
import { StatusChip } from "@/components/StatusChip";
import { LAB_URL } from "@/components/NavHeader";
import cpExamples from "@/data/cp_examples.json";
import designSpace from "@/data/design_space_error.json";
import physicsEval from "@/data/held_out_eval.json";

export function AeropinnLabSection() {
  const cases = cpExamples.physics_guided;
  const [idx, setIdx] = useState(0);
  const c = cases[idx];

  return (
    <SectionShell id="06-aeropinn-lab" number="06" title="AeroPINN Lab" typography="engineering">
      <p className="mb-4 max-w-2xl leading-relaxed" data-typography="editorial">
        The AeroPINN Lab is the live engineering interface: a three-panel workstation — geometry and
        operating-condition controls on the left, scientific visualization in the center, model
        telemetry (CL, CD, CL/CD, uncertainty, error, inference time, domain status) on the right.
      </p>

      <ExperimentMetadataBlock
        items={[
          { label: "Model", value: "pinn_v1" },
          { label: "Device", value: physicsEval.device },
          { label: "Held-out cases", value: String(physicsEval.n_cases) },
          { label: "Status", value: "live" },
          { label: "Reference method", value: "SU2 (laminar, incompressible)" },
        ]}
      />

      <div className="my-4">
        <a href={LAB_URL} target="_blank" rel="noopener noreferrer"
          className="mono text-xs border border-accent text-accent rounded-[4px] px-4 py-2.5 inline-block hover:bg-accent hover:text-bg">
          OPEN AEROPINN LAB ↗
        </a>
      </div>

      <h3 className="text-lg mt-8 mb-2" data-typography="editorial">Unseen Airfoil Challenge</h3>
      <p className="mb-3 max-w-2xl leading-relaxed" data-typography="editorial">
        Select a held-out airfoil — <StatusChip text="NOT PRESENT IN TRAINING DATA" level="warn" /> — and
        compare the model&apos;s prediction against the hidden SU2 reference.
      </p>

      <select
        value={idx}
        onChange={(e) => setIdx(Number(e.target.value))}
        className="mono text-xs bg-panel-alt border border-border rounded-[4px] px-2 py-1.5 mb-4 w-full sm:w-auto max-w-full"
      >
        {cases.map((cc, i) => (
          <option key={cc.geom_id + i} value={i}>
            {cc.geom_id} · m={cc.m.toFixed(3)} p={cc.p.toFixed(2)} t={cc.t.toFixed(3)} · AoA={cc.alpha_deg}° Re={cc.reynolds.toExponential(0)}
          </option>
        ))}
      </select>

      <Figure
        number="05"
        title="Pressure Coefficient Prediction"
        provenance="real"
        sourceNote="aeropinn/experiments/checkpoints/held_out_demo_cases.npz"
        caption={`Surface Cp, predicted vs. SU2 reference, for held-out geometry ${c.geom_id} at AoA=${c.alpha_deg}°, Re=${c.reynolds.toExponential(1)}. Reference CL=${c.cl_true.toFixed(4)}, CD=${c.cd_true.toFixed(4)}. Field relative L2 error — u: ${c.u_rel_l2.toFixed(3)}, v: ${c.v_rel_l2.toFixed(3)}, p: ${c.p_rel_l2.toFixed(3)}.`}
      >
        <CpChart c={c} showReference />
      </Figure>

      <Figure
        number="Design Space"
        title="Training vs. Held-Out Geometries"
        provenance="real"
        sourceNote="aeropinn/data/processed/sweep_v1/geometries.json"
        caption="16 training geometries, 6 held-out test geometries in (camber, thickness) space. A single Latin-Hypercube sample, single seed."
      >
        <DesignSpaceScatter points={designSpace as GeometryPoint[]} colorBy="split" />
      </Figure>
    </SectionShell>
  );
}
