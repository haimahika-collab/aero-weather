"use client";

import { useState } from "react";
import { SectionShell } from "@/components/SectionShell";
import { AirfoilDiagram } from "@/components/AirfoilDiagram";
import { CpChart } from "@/components/CpChart";
import { StatusChip } from "@/components/StatusChip";
import { LAB_URL } from "@/components/NavHeader";
import { nacaDesignation } from "@/lib/naca";
import cpExamples from "@/data/cp_examples.json";

type ViewMode = "geometry" | "prediction" | "reference" | "comparison";

const MODES: { key: ViewMode; label: string }[] = [
  { key: "geometry", label: "GEOMETRY" },
  { key: "prediction", label: "PREDICTION" },
  { key: "reference", label: "REFERENCE" },
  { key: "comparison", label: "COMPARISON" },
];

export function ComputationalAerodynamicsSection() {
  const cases = cpExamples.physics_guided;
  const [idx, setIdx] = useState(0);
  const [mode, setMode] = useState<ViewMode>("comparison");
  const c = cases[idx];

  return (
    <SectionShell id="03-computational-aerodynamics" number="03" title="Computational Aerodynamics" typography="engineering">
      <p className="mb-4 max-w-2xl leading-relaxed" data-typography="editorial">
        These are {cases.length} real, pre-evaluated held-out cases — not a live model. Geometry and
        condition are selected from the menu below, not free sliders, since these are cached results,
        not a running inference server.{" "}
        <a href={LAB_URL} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">
          Open the AeroPINN Lab for live inference on any geometry →
        </a>
      </p>

      <StatusChip text="MODEL OUTPUT (PRECOMPUTED)" level="ok" />

      <div className="flex flex-wrap gap-3 mt-4 mb-4">
        <select
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
          className="mono text-xs bg-panel-alt border border-border rounded-[4px] px-2 py-1.5 w-full sm:w-auto max-w-full"
        >
          {cases.map((cc, i) => (
            <option key={cc.geom_id + i} value={i}>
              {cc.geom_id} · AoA={cc.alpha_deg}° Re={cc.reynolds.toExponential(0)}
            </option>
          ))}
        </select>
        <div className="flex gap-1">
          {MODES.map((m) => (
            <button
              key={m.key}
              onClick={() => setMode(m.key)}
              className={`mono text-[11px] rounded-[4px] px-2.5 py-1.5 border ${
                mode === m.key ? "border-accent text-accent" : "border-border text-text-muted"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="border border-border rounded-[var(--radius)] bg-panel-alt p-4 flex justify-center">
        {mode === "geometry" ? (
          <div className="text-center">
            <AirfoilDiagram m={c.m} p={c.p} t={c.t} width={420} height={180} />
            <p className="mono text-xs text-accent mt-2">{nacaDesignation(c.m, c.p, c.t)}</p>
          </div>
        ) : (
          <CpChart c={c} showReference={mode !== "prediction"} />
        )}
      </div>
      <p className="mono text-[11px] text-text-caption mt-2">
        NACA-style m={c.m.toFixed(3)} p={c.p.toFixed(2)} t={c.t.toFixed(3)}, AoA={c.alpha_deg}°,
        Re={c.reynolds.toExponential(1)} — held-out, never seen during training.
      </p>
    </SectionShell>
  );
}
