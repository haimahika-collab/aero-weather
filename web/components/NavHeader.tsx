const SECTIONS = [
  { id: "01-overview", label: "01", title: "Overview" },
  { id: "02-aircraft-airfoil", label: "02", title: "From Aircraft to Airfoil" },
  { id: "03-computational-aerodynamics", label: "03", title: "Computational Aerodynamics" },
  { id: "04-neural-surrogate", label: "04", title: "Neural Surrogate" },
  { id: "05-experiment", label: "05", title: "Experiment" },
  { id: "06-aeropinn-lab", label: "06", title: "AeroPINN Lab" },
  { id: "07-inverse-design", label: "07", title: "Inverse Design" },
  { id: "08-results-limitations", label: "08", title: "Results & Limitations" },
  { id: "09-about", label: "09", title: "About" },
];

const LAB_URL = process.env.NEXT_PUBLIC_LAB_URL ?? "https://aero-weather-production.up.railway.app/";

export function NavHeader() {
  return (
    <header className="sticky top-0 z-50 bg-bg/95 backdrop-blur-sm border-b border-border">
      <div className="max-w-6xl mx-auto px-4 md:px-6 h-12 flex items-center justify-between gap-4">
        <a href="#01-overview" className="mono text-sm font-bold tracking-widest text-text shrink-0">
          AEROPINN
        </a>
        <nav className="hidden lg:flex items-center gap-3">
          {SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              title={s.title}
              className="mono text-[11px] text-text-muted hover:text-accent whitespace-nowrap"
            >
              {s.label}
            </a>
          ))}
        </nav>
        <a
          href={LAB_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="mono text-[11px] border border-accent text-accent rounded-[4px] px-3 py-1.5 whitespace-nowrap hover:bg-accent hover:text-bg"
        >
          LAUNCH AEROPINN LAB
        </a>
      </div>
    </header>
  );
}

export { LAB_URL };
