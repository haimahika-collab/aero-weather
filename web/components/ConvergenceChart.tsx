import { AERO_PALETTE } from "@/lib/palette";

export interface ConvergenceHistoryPoint {
  iter: number;
  cl_over_cd: number;
}

const W = 480;
const H = 260;
const MARGIN = { top: 16, right: 16, bottom: 32, left: 48 };

/** Optimizer convergence (real, gradient_history from inverse_design_cache.json),
 * with the independent SU2 result plotted as a separate annotated reference line —
 * the surrogate's own convergence trace never gets to stand in for validation. */
export function ConvergenceChart({
  history,
  su2ClOverCd,
}: {
  history: ConvergenceHistoryPoint[];
  su2ClOverCd?: number | null;
}) {
  const iters = history.map((h) => h.iter);
  const vals = history.map((h) => h.cl_over_cd);
  const allVals = su2ClOverCd != null ? [...vals, su2ClOverCd] : vals;
  const yMin = Math.min(...allVals);
  const yMax = Math.max(...allVals);
  const pad = (yMax - yMin) * 0.1 || 0.5;

  const xScale = (i: number) => MARGIN.left + (i / Math.max(...iters, 1)) * (W - MARGIN.left - MARGIN.right);
  const yScale = (v: number) =>
    H - MARGIN.bottom - ((v - (yMin - pad)) / (yMax - yMin + 2 * pad)) * (H - MARGIN.top - MARGIN.bottom);

  const path = history.map((h, i) => `${i === 0 ? "M" : "L"} ${xScale(h.iter)} ${yScale(h.cl_over_cd)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Inverse design optimizer convergence">
      <line x1={MARGIN.left} y1={H - MARGIN.bottom} x2={W - MARGIN.right} y2={H - MARGIN.bottom} stroke={AERO_PALETTE.border} />
      <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={H - MARGIN.bottom} stroke={AERO_PALETTE.border} />
      <text x={W / 2} y={H - 8} fill={AERO_PALETTE.text_muted} fontSize="11" textAnchor="middle" className="mono">
        optimizer iteration
      </text>
      <text x={14} y={H / 2} fill={AERO_PALETTE.text_muted} fontSize="10" textAnchor="middle" className="mono" transform={`rotate(-90 14 ${H / 2})`}>
        CL/CD (predicted)
      </text>

      <path d={path} fill="none" stroke={AERO_PALETTE.accent} strokeWidth={2} />
      <circle cx={xScale(history[history.length - 1].iter)} cy={yScale(history[history.length - 1].cl_over_cd)} r={4} fill={AERO_PALETTE.accent} />

      {su2ClOverCd != null && (
        <>
          <line
            x1={MARGIN.left} x2={W - MARGIN.right}
            y1={yScale(su2ClOverCd)} y2={yScale(su2ClOverCd)}
            stroke={AERO_PALETTE.error} strokeWidth={1.5} strokeDasharray="4,3"
          />
          <text x={W - MARGIN.right} y={yScale(su2ClOverCd) - 4} fill={AERO_PALETTE.error} fontSize="10" textAnchor="end" className="mono">
            independent SU2: {su2ClOverCd.toFixed(2)}
          </text>
        </>
      )}
    </svg>
  );
}
