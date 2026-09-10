import { NavHeader } from "@/components/NavHeader";
import { OverviewSection } from "@/components/sections/01-overview";
import { AircraftAirfoilSection } from "@/components/sections/02-aircraft-airfoil";
import { ComputationalAerodynamicsSection } from "@/components/sections/03-computational-aerodynamics";
import { NeuralSurrogateSection } from "@/components/sections/04-neural-surrogate";
import { ExperimentSection } from "@/components/sections/05-experiment";
import { AeropinnLabSection } from "@/components/sections/06-aeropinn-lab";
import { InverseDesignSection } from "@/components/sections/07-inverse-design";
import { ResultsLimitationsSection } from "@/components/sections/08-results-limitations";
import { AboutSection } from "@/components/sections/09-about";

export default function Home() {
  return (
    <>
      <NavHeader />
      <main className="flex-1">
        <OverviewSection />
        <AircraftAirfoilSection />
        <ComputationalAerodynamicsSection />
        <NeuralSurrogateSection />
        <ExperimentSection />
        <AeropinnLabSection />
        <InverseDesignSection />
        <ResultsLimitationsSection />
        <AboutSection />
      </main>
      <footer className="px-6 md:px-12 py-6 text-center mono text-[11px] text-text-caption">
        AeroPINN — a physics-guided neural surrogate research project.
      </footer>
    </>
  );
}
