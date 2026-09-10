import { AERO_PALETTE } from "@/lib/palette";

export interface GeometryPoint {
  geom_id: string;
  m: number;
  t: number;
  split: "train" | "held_out";
  mean_p_rel_l2?: number;
}

const W = 480;
const H = 380;
const MARGIN = { top: 16, right: 70, bottom: 40, left: 48 };

/** Training vs. held-out geometries in (m, t) space, or the same points colored by
 * held-out error (Failure Map). Mirrors aeropinn/visualization/plots.py::plot_design_space —
 * same axes, same marker conventions (circle=train, diamond=held-out). */
export function DesignSpaceScatter({
  points,
  colorBy = "split",
  current,
  optimized,
}: {
  points: GeometryPoint[];
  colorBy?: "split" | "error";
  current?: { m: number; t: number };
  optimized?: { m: number; t: number };
}) {
  const mVals = points.map((p) => p.m);
  const tVals = points.map((p) => p.t);
  const mMin = 0, mMax = 0.06;
  const tMin = 0.08, tMax = 0.2;

  const xScale = (m: number) => MARGIN.left + ((m - mMin) / (mMax - mMin)) * (W - MARGIN.left - MARGIN.right);
  const yScale = (t: number) => H - MARGIN.bottom - ((t - tMin) / (tMax - tMin)) * (H - MARGIN.top - MARGIN.bottom);

  const errorVals = points.filter((p) => p.mean_p_rel_l2 !== undefined).map((p) => p.mean_p_rel_l2!);
  const errMin = errorVals.length ? Math.min(...errorVals) : 0;
  const errMax = errorVals.length ? Math.max(...errorVals) : 1;
  const errorColor = (v: number) => {
    const t = errMax > errMin ? (v - errMin) / (errMax - errMin) : 0.5;
    // pale yellow -> orange -> red
    const r = 255;
    const g = Math.round(220 - t * 140);
    const b = Math.round(150 - t * 150);
    return `rgb(${r},${g},${Math.max(b, 0)})`;
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Design space scatter plot">
      <line x1={MARGIN.left} y1={H - MARGIN.bottom} x2={W - MARGIN.right} y2={H - MARGIN.bottom} stroke={AERO_PALETTE.border} />
      <line x1={MARGIN.left} y1={MARGIN.top} x2={MARGIN.left} y2={H - MARGIN.bottom} stroke={AERO_PALETTE.border} />
      <text x={W / 2} y={H - 10} fill={AERO_PALETTE.text_muted} fontSize="11" textAnchor="middle" className="mono">
        m — max camber
      </text>
      <text x={14} y={H / 2} fill={AERO_PALETTE.text_muted} fontSize="11" textAnchor="middle" className="mono" transform={`rotate(-90 14 ${H / 2})`}>
        t — max thickness
      </text>

      {points.map((pt) => {
        const cx = xScale(pt.m);
        const cy = yScale(pt.t);
        if (colorBy === "error") {
          if (pt.split === "train") {
            return <circle key={pt.geom_id} cx={cx} cy={cy} r={4} fill={AERO_PALETTE.train_geom} fillOpacity={0.5} />;
          }
          const color = pt.mean_p_rel_l2 !== undefined ? errorColor(pt.mean_p_rel_l2) : AERO_PALETTE.text_muted;
          return (
            <rect key={pt.geom_id} x={cx - 6} y={cy - 6} width={12} height={12} fill={color}
              stroke={AERO_PALETTE.text} strokeWidth={1} transform={`rotate(45 ${cx} ${cy})`} />
          );
        }
        if (pt.split === "train") {
          return <circle key={pt.geom_id} cx={cx} cy={cy} r={4} fill={AERO_PALETTE.train_geom} />;
        }
        return (
          <rect key={pt.geom_id} x={cx - 6} y={cy - 6} width={12} height={12} fill={AERO_PALETTE.held_out_geom}
            stroke={AERO_PALETTE.text} strokeWidth={1} transform={`rotate(45 ${cx} ${cy})`} />
        );
      })}

      {current && (
        <text x={xScale(current.m)} y={yScale(current.t)} fontSize={18} fill={AERO_PALETTE.current_design} textAnchor="middle" dominantBaseline="middle">★</text>
      )}
      {optimized && (
        <text x={xScale(optimized.m)} y={yScale(optimized.t)} fontSize={18} fill={AERO_PALETTE.optimized_design} textAnchor="middle" dominantBaseline="middle">★</text>
      )}

      <g transform={`translate(${W - MARGIN.right + 8}, ${MARGIN.top})`}>
        <circle cx={5} cy={6} r={4} fill={AERO_PALETTE.train_geom} fillOpacity={colorBy === "error" ? 0.5 : 1} />
        <text x={14} y={10} fontSize="9" fill={AERO_PALETTE.text_muted} className="mono">train</text>
        <rect x={1} y={22} width={8} height={8}
          fill={colorBy === "error" ? errorColor((errMin + errMax) / 2) : AERO_PALETTE.held_out_geom}
          stroke={colorBy === "error" ? AERO_PALETTE.text : "none"} strokeWidth={1}
          transform="rotate(45 5 26)" />
        <text x={14} y={30} fontSize="9" fill={AERO_PALETTE.text_muted} className="mono">
          {colorBy === "error" ? "= error" : "held-out"}
        </text>
      </g>
    </svg>
  );
}
