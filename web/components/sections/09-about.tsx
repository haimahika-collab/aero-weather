import { SectionShell } from "@/components/SectionShell";

export function AboutSection() {
  return (
    <SectionShell id="09-about" number="09" title="About" typography="editorial">
      <h3 className="text-xl mb-1">Mahika Modigunta</h3>
      <p className="mono text-xs text-text-muted mb-6">Student Researcher</p>
      <div className="space-y-4 max-w-2xl leading-relaxed text-[15px]">
        <p>
          Hi, I&apos;m Mahika Modigunta, a high school student interested in aerospace engineering,
          computational modeling, and machine learning.
        </p>
        <p>
          I started exploring neural networks through aviation data and flight prediction. That work
          led me to a bigger question: instead of asking a neural network to simply recognize patterns
          in data, could we incorporate what we already know about aerodynamics to help it make better
          predictions?
        </p>
        <p>That question became AeroPINN.</p>
        <p>
          AeroPINN explores physics-guided neural networks for predicting aerodynamic performance
          across different airfoil geometries and flight conditions. My goal is to investigate how
          physics-guided learning compares with conventional data-driven neural networks, particularly
          when predicting airfoils the model has not previously encountered or when training data are
          limited.
        </p>
        <p>
          I built this project not only to create a fast aerodynamic prediction tool, but to better
          understand where machine learning can and cannot be useful in aerospace engineering.
        </p>
        <p>
          This website shows the model, experiments, validation results, limitations, and design tools
          developed throughout the research.
        </p>
      </div>
    </SectionShell>
  );
}
