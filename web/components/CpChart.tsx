import { nacaCamber } from "@/lib/naca";
import { AERO_PALETTE } from "@/lib/palette";

export interface CpCase {
  m: number;
  p: number;
  surf_x: number[];
  surf_y: number[];
  surf_cp_true?: number[];
  surf_cp_pred: number[];
}

const W = 480;
const H = 340;
const MARGIN = { top: 16, right: 16, bottom: 36, left: 44 };

function buildPath(xs: number[], ys: number[], xScale: (v: number) => number, yScale: (v: number) => number) {
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b]);
  return order.map((i, k) => `${k === 0 ? "M" : "L"} ${xScale(xs[i])} ${yScale(ys[i])}`).join(" ");
}

/** Surface Cp: predicted (dashed accent) vs. reference (solid gray) if present.
 * Splits upper/lower via the camber line — a NACA airfoil has two points per x
 * station, so sorting by x alone would zigzag between them (matches
 * aeropinn/visualization/plots.py::render_cp_overlay's exact logic). */
export function CpChart({ c, showReference = true }: { c: CpCase; showReference?: boolean }) {
  const yc = c.surf_x.map((x) => nacaCamber(Math.max(0, Math.min(1, x)), c.m, c.p));
  const isUpper = c.surf_y.map((y, i) => y >= yc[i]);

  const allCp = [...c.surf_cp_pred, ...(showReference && c.surf_cp_true ? c.surf_cp_true : [])];
  const cpMin = Math.min(...allCp);
  const cpMax = Math.max(...allCp);
  const pad = (cpMax - cpMin) * 0.1 || 0.1;

  const xScale = (x: number) => MARGIN.left + x * (W - MARGIN.left - MARGIN.right);
  // Cp axis inverted (standard convention: negative Cp / suction plotted upward)
  const yScale = (cp: number) =>
    MARGIN.top +
    ((cp - (cpMin - pad)) / (cpMax - cpMin + 2 * pad)) * (H - MARGIN.top - MARGIN.bottom);

  const upperIdx = isUpper.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
  const lowerIdx = isUpper.map((v, i) => (!v ? i : -1)).filter((i) => i >= 0);

  const pick = (idx: number[], arr: number[]) => idx.map((i) => arr[i]);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Surface pressure coefficient plot">
      {/* axes */}
      <line x1={MARGIN.left} y1={H - MARGIN.bottom} x2={W - MARGIN.right} y2={H - MARGIN.bottom} stroke={AERO_PALETTE.border} />
      <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={H - MARGIN.bottom} stroke={AERO_PALETTE.border} />
      <text x={W / 2} y={H - 8} fill={AERO_PALETTE.text_muted} fontSize="11" textAnchor="middle" className="mono">
        x / chord
      </text>
      <text x={14} y={H / 2} fill={AERO_PALETTE.text_muted} fontSize="11" textAnchor="middle" className="mono" transform={`rotate(-90 14 ${H / 2})`}>
        Cp
      </text>

      {[upperIdx, lowerIdx].map((idx, seg) => (
        <g key={seg}>
          {showReference && c.surf_cp_true && (
            <path
              d={buildPath(pick(idx, c.surf_x), pick(idx, c.surf_cp_true), xScale, yScale)}
              fill="none"
              stroke={AERO_PALETTE.reference}
              strokeWidth={2}
            />
          )}
          <path
            d={buildPath(pick(idx, c.surf_x), pick(idx, c.surf_cp_pred), xScale, yScale)}
            fill="none"
            stroke={AERO_PALETTE.prediction}
            strokeWidth={2}
            strokeDasharray="6,4"
          />
        </g>
      ))}

      {/* legend */}
      <g transform={`translate(${W - 170}, ${MARGIN.top + 4})`}>
        {showReference && c.surf_cp_true && (
          <>
            <line x1={0} y1={0} x2={20} y2={0} stroke={AERO_PALETTE.reference} strokeWidth={2} />
            <text x={26} y={4} fill={AERO_PALETTE.text} fontSize="10" className="mono">SU2 reference</text>
          </>
        )}
        <line x1={0} y1={16} x2={20} y2={16} stroke={AERO_PALETTE.prediction} strokeWidth={2} strokeDasharray="6,4" />
        <text x={26} y={20} fill={AERO_PALETTE.text} fontSize="10" className="mono">AeroPINN prediction</text>
      </g>
    </svg>
  );
}
