import { SectionShell } from "@/components/SectionShell";
import physicsEval from "@/data/held_out_eval.json";
import inverseDesign from "@/data/inverse_design.json";
import limitations from "@/data/limitations.json";

export function ResultsLimitationsSection() {
  const disagreements = inverseDesign.filter((r) => r.su2_validation.status === "validated_disagreement").length;

  return (
    <SectionShell id="08-results-limitations" number="08" title="Results & Limitations" typography="editorial">
      <h3 className="text-lg mb-3">Results</h3>
      <ul className="space-y-2 mb-8 max-w-2xl leading-relaxed list-disc pl-5">
        <li>
          Held-out geometry performance: mean relative L2 error of{" "}
          <span data-numeric>{physicsEval.mean_u_rel_l2.toFixed(3)}</span> (u),{" "}
          <span data-numeric>{physicsEval.mean_v_rel_l2.toFixed(3)}</span> (v),{" "}
          <span data-numeric>{physicsEval.mean_p_rel_l2.toFixed(3)}</span> (p) across{" "}
          {physicsEval.n_cases} held-out cases.
        </li>
        <li>
          Physics-guided vs. data-only: a mixed result — physics guidance improved streamwise
          velocity error but not cross-flow velocity or pressure error at this training scale.
          See Figure 03.
        </li>
        <li>Data efficiency: not yet measured (Figure 04, pending).</li>
        <li>
          Inference speed: warm-call latency ~{physicsEval.latency_breakdown.warm_mean_ms?.toFixed(2)}ms;
          the first call per geometry is slower (~{physicsEval.latency_breakdown.cold_mean_ms?.toFixed(1)}ms,
          model/OOD load path) — reported separately rather than blended into one average.
        </li>
        <li>
          Inverse-design validation: {inverseDesign.length} optimization runs were independently
          re-evaluated with SU2. <strong>{disagreements} of {inverseDesign.length}</strong> came back
          in disagreement with the surrogate&apos;s own prediction — the optimizer found designs the
          model over- or under-rated near the edges of its validated envelope, and independent
          validation caught every one. That is the safety mechanism working as intended, not a
          failure to hide: no inverse-design candidate from this project has yet been confirmed as a
          genuine aerodynamic improvement.
        </li>
      </ul>

      <h3 className="text-lg mb-3">Limitations</h3>
      <div className="space-y-3 max-w-2xl">
        {limitations.map((l) => (
          <div key={l.topic} className="border-l-2 border-border pl-4">
            <p className="mono text-xs text-accent mb-1">{l.topic}</p>
            <p className="text-[15px] leading-relaxed">{l.text}</p>
            <p className="mono text-[11px] text-text-caption mt-1">{l.source}</p>
          </div>
        ))}
      </div>
    </SectionShell>
  );
}
