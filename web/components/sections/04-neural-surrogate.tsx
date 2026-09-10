"use client";

import { useState } from "react";
import { SectionShell } from "@/components/SectionShell";

const LAYERS = {
  geometry: {
    label: "GEOMETRY INPUTS",
    detail: "m, p, t — max camber, camber position, max thickness. Bounded to the validated envelope: m ∈ [0, 0.06], p ∈ [0.10, 0.60], t ∈ [0.08, 0.20].",
  },
  conditions: {
    label: "OPERATING CONDITIONS",
    detail: "α (angle of attack, −4° to 8°, pre-stall) and Re (Reynolds number, 3×10⁴–1×10⁵, laminar regime).",
  },
  network: {
    label: "NEURAL SURROGATE",
    detail: "A geometry-conditioned physics-informed network (PDE-residual loss term for steady, 2D, incompressible, laminar Navier–Stokes) — an 8-layer, 128-width MLP with Fourier-feature spatial encoding.",
  },
  outputs: {
    label: "AERODYNAMIC OUTPUTS",
    detail: "CL, CD, Cp distribution, and a binary in/out-of-validated-domain flag (Mahalanobis distance in the 5D parameter space).",
  },
} as const;

type LayerKey = keyof typeof LAYERS;

export function NeuralSurrogateSection() {
  const [active, setActive] = useState<LayerKey>("network");

  return (
    <SectionShell id="04-neural-surrogate" number="04" title="Neural Surrogate" typography="editorial">
      <p className="mb-6 max-w-2xl leading-relaxed">
        Click a stage below for detail. This is a static architecture diagram — there is no live
        backend connection on this website (real inference happens in the AeroPINN Lab).
      </p>

      <div className="flex flex-col md:flex-row items-center gap-3 md:gap-2 justify-center mb-6" data-typography="engineering">
        {(["geometry", "conditions"] as LayerKey[]).map((k) => (
          <button
            key={k}
            onClick={() => setActive(k)}
            className={`mono text-xs border rounded-[4px] px-3 py-3 text-center ${active === k ? "border-accent text-accent" : "border-border text-text-muted"}`}
          >
            {LAYERS[k].label}
          </button>
        ))}
        <span className="text-text-muted hidden md:inline">→</span>
        <button
          onClick={() => setActive("network")}
          className={`mono text-xs border rounded-[4px] px-4 py-3 text-center ${active === "network" ? "border-accent text-accent bg-panel-alt" : "border-border text-text-muted"}`}
        >
          {LAYERS.network.label}
        </button>
        <span className="text-text-muted hidden md:inline">→</span>
        <button
          onClick={() => setActive("outputs")}
          className={`mono text-xs border rounded-[4px] px-3 py-3 text-center ${active === "outputs" ? "border-accent text-accent" : "border-border text-text-muted"}`}
        >
          {LAYERS.outputs.label}
        </button>
      </div>

      <div className="border border-border rounded-[var(--radius)] bg-panel p-4 max-w-2xl">
        <p className="mono text-xs text-accent mb-1">{LAYERS[active].label}</p>
        <p className="text-[15px] leading-relaxed">{LAYERS[active].detail}</p>
      </div>
    </SectionShell>
  );
}
