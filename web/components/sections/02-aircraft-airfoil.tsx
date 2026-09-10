"use client";

import { useState } from "react";
import { SectionShell } from "@/components/SectionShell";
import { AircraftSilhouette } from "@/components/AircraftSilhouette";
import { AirfoilDiagram } from "@/components/AirfoilDiagram";
import { Figure } from "@/components/Figure";
import { StatusChip } from "@/components/StatusChip";
import { nacaDesignation } from "@/lib/naca";
import geometries from "@/data/geometries.json";

function ParameterizationFigure() {
  const [m, setM] = useState(0.02);
  const [p, setP] = useState(0.4);
  const [t, setT] = useState(0.12);

  return (
    <Figure
      number="01"
      title="Airfoil Parameterization"
      provenance="real"
      sourceNote="web/lib/naca.ts (ported from aeropinn/geometry/naca.py), pure geometry — no model inference"
      caption="NACA 4-digit parameterization: camber (m), camber position (p), thickness (t). This is the only figure with free sliders — it's pure geometry math, not model inference."
    >
      <div className="flex flex-col items-center">
        <AirfoilDiagram m={m} p={p} t={t} width={420} height={170} />
        <p className="mono text-xs text-accent mt-2 mb-4">{nacaDesignation(m, p, t)}</p>
        <div className="w-full max-w-xs space-y-3 mono text-[11px]">
          <label className="block">
            m — max camber: <span data-numeric>{m.toFixed(3)}</span>
            <input type="range" min={0} max={0.06} step={0.005} value={m} onChange={(e) => setM(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
          </label>
          <label className="block">
            p — camber position: <span data-numeric>{p.toFixed(2)}</span>
            <input type="range" min={0.1} max={0.6} step={0.01} value={p} onChange={(e) => setP(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
          </label>
          <label className="block">
            t — max thickness: <span data-numeric>{t.toFixed(3)}</span>
            <input type="range" min={0.08} max={0.2} step={0.005} value={t} onChange={(e) => setT(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
          </label>
        </div>
      </div>
    </Figure>
  );
}

const SHOTS = [
  { label: "Aircraft", caption: "A conventional fixed-wing aircraft — conceptual diagram." },
  { label: "Wing", caption: "The wing is isolated. A 2D cross-section is taken here, spanwise." },
  { label: "Airfoil cross-section", caption: "The extracted cross-section — a real NACA 4-digit geometry, the same parameterization used throughout this project." },
];

export function AircraftAirfoilSection() {
  const [shot, setShot] = useState(0);
  const trainGeom = geometries.find((g) => g.split === "train") ?? geometries[0];

  return (
    <SectionShell id="02-aircraft-airfoil" number="02" title="From Aircraft to Airfoil" typography="editorial">
      <p className="mb-6 max-w-2xl leading-relaxed">
        AeroPINN studies 2D flow around a single spanwise cross-section of a wing — the airfoil — not
        the full 3D aircraft. This is the standard simplification that makes the aerodynamic prediction
        problem tractable while remaining physically meaningful.
      </p>

      <div className="flex items-center justify-center border border-border rounded-[var(--radius)] bg-panel-alt py-10 mb-4 min-h-[280px]">
        {shot === 0 && <AircraftSilhouette width={440} height={240} />}
        {shot === 1 && <AircraftSilhouette width={440} height={240} highlightWing />}
        {shot === 2 && (
          <div className="text-center">
            <AirfoilDiagram m={trainGeom.m} p={trainGeom.p} t={trainGeom.t} width={440} height={180} fillOpacity={0.35} />
            <p className="mono text-xs text-accent mt-2">{nacaDesignation(trainGeom.m, trainGeom.p, trainGeom.t)}</p>
          </div>
        )}
      </div>

      <div className="flex items-center justify-center gap-2 mb-3">
        {SHOTS.map((s, i) => (
          <button
            key={s.label}
            onClick={() => setShot(i)}
            className={`mono text-xs rounded-[4px] px-3 py-1.5 border ${
              shot === i ? "border-accent text-accent" : "border-border text-text-muted"
            }`}
          >
            {i + 1}. {s.label}
          </button>
        ))}
      </div>

      <p className="text-sm text-text-muted text-center mb-2">{SHOTS[shot].caption}</p>
      <div className="flex justify-center mb-8">
        <StatusChip text={shot === 2 ? "REAL GEOMETRY" : "CONCEPTUAL DIAGRAM"} level={shot === 2 ? "ok" : "muted"} />
      </div>

      <ParameterizationFigure />
    </SectionShell>
  );
}
