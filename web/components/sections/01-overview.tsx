import { SectionShell } from "@/components/SectionShell";
import { AircraftSilhouette } from "@/components/AircraftSilhouette";
import { LAB_URL } from "@/components/NavHeader";

export function OverviewSection() {
  return (
    <SectionShell id="01-overview" number="01" title="Overview" typography="editorial">
      <div className="grid md:grid-cols-2 gap-8 items-center">
        <div>
          <h1 className="text-4xl md:text-5xl font-normal tracking-tight mb-4">AEROPINN</h1>
          <p className="text-lg text-text-muted mb-6">
            Physics-Guided Neural Surrogates for Aerodynamic Prediction
          </p>
          <p className="text-base leading-relaxed mb-8 max-w-lg">
            Investigating whether aerodynamic physics can improve neural-network generalization
            across unseen airfoil geometries and flight conditions.
          </p>
          <div className="flex flex-wrap gap-3">
            <a
              href="#02-aircraft-airfoil"
              className="mono text-xs border border-border text-text rounded-[4px] px-4 py-2.5 hover:border-accent hover:text-accent"
            >
              EXPLORE RESEARCH
            </a>
            <a
              href={LAB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="mono text-xs border border-accent text-accent rounded-[4px] px-4 py-2.5 hover:bg-accent hover:text-bg"
            >
              LAUNCH AEROPINN LAB
            </a>
          </div>
        </div>
        <div className="flex justify-center">
          <div>
            <AircraftSilhouette width={420} height={230} />
            <p className="mono text-[11px] text-text-caption text-center mt-2">conceptual diagram</p>
          </div>
        </div>
      </div>
    </SectionShell>
  );
}
