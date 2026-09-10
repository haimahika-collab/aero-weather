import { SectionShell } from "@/components/SectionShell";
import { Figure } from "@/components/Figure";
import physicsEval from "@/data/held_out_eval.json";
import dataOnlyEval from "@/data/held_out_eval_b2.json";

function betterWorse(physics: number, dataOnly: number): string {
  if (physics < dataOnly) return "physics guidance lower error";
  if (physics > dataOnly) return "data-only lower error";
  return "no difference";
}

export function ExperimentSection() {
  const rows: { label: string; key: "mean_u_rel_l2" | "mean_v_rel_l2" | "mean_p_rel_l2" }[] = [
    { label: "u (streamwise velocity)", key: "mean_u_rel_l2" },
    { label: "v (cross-flow velocity)", key: "mean_v_rel_l2" },
    { label: "p (pressure)", key: "mean_p_rel_l2" },
  ];

  return (
    <SectionShell id="05-experiment" number="05" title="Experiment" typography="editorial">
      <p className="mb-2 max-w-2xl leading-relaxed">
        The research pipeline: airfoil geometry → reference aerodynamic data (SU2 CFD) → model
        training → physics guidance → prediction → held-out geometry test → independent validation.
      </p>
      <h3 className="text-lg mt-8 mb-3">Does physics guidance actually improve unseen-airfoil prediction?</h3>
      <p className="mb-4 max-w-2xl leading-relaxed">
        A data-only neural network and a physics-guided neural network (identical architecture,
        identical training data, identical seed — the only difference is the PDE-residual loss term)
        were evaluated on the same {physicsEval.n_cases} held-out geometries never seen during
        training. Neither model is visually or numerically biased in this comparison.
      </p>

      <Figure
        number="03"
        title="Held-Out Geometry Evaluation"
        provenance="real"
        sourceNote="aeropinn/experiments/checkpoints/held_out_eval.json, held_out_eval_b2.json"
        caption={`Mean relative L2 field error across ${physicsEval.n_cases} held-out geometries, physics-guided vs. data-only. The honest result: ${rows
          .map((r) => `${r.label} — ${betterWorse(physicsEval[r.key], dataOnlyEval[r.key])}`)
          .join("; ")}. Physics guidance was not clearly better at this training scale.`}
      >
        <table className="w-full mono text-xs">
          <thead>
            <tr className="text-text-muted text-left border-b border-border">
              <th className="py-2 font-normal">Metric</th>
              <th className="py-2 font-normal">Physics-guided</th>
              <th className="py-2 font-normal">Data-only</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-b border-border/50">
                <td className="py-2 text-text-muted">{r.label}</td>
                <td className="py-2" data-numeric>{physicsEval[r.key].toFixed(4)}</td>
                <td className="py-2" data-numeric>{dataOnlyEval[r.key].toFixed(4)}</td>
              </tr>
            ))}
            <tr>
              <td className="py-2 text-text-muted">Mean latency (warm)</td>
              <td className="py-2" data-numeric>{physicsEval.latency_breakdown.warm_mean_ms?.toFixed(2)} ms</td>
              <td className="py-2" data-numeric>{dataOnlyEval.latency_breakdown.warm_mean_ms?.toFixed(2)} ms</td>
            </tr>
          </tbody>
        </table>
      </Figure>

      <Figure
        number="04"
        title="Performance Under Limited Training Data"
        provenance="pending"
        pendingReason="A data-efficiency sweep (10% / 25% / 50% / 100% of labeled training data, physics-guided vs. data-only) has not yet been run. This figure will populate once that experiment completes — no estimated or interpolated values are shown in its place."
        sourceNote="not yet run"
        caption="Planned: physics-guided vs. data-only test error at 10/25/50/100% labeled training data."
      />
    </SectionShell>
  );
}
