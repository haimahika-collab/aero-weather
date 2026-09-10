import { naca4SvgPath } from "@/lib/naca";
import { AERO_PALETTE } from "@/lib/palette";

export function AirfoilDiagram({
  m,
  p,
  t,
  width = 420,
  height = 160,
  fillOpacity = 0.25,
}: {
  m: number;
  p: number;
  t: number;
  width?: number;
  height?: number;
  fillOpacity?: number;
}) {
  const path = naca4SvgPath(m, p, t, 100);
  // Airfoil geometry lives in [~-0.02, 1.02] x [-0.15, 0.15] roughly; map to viewBox with margin.
  const viewBox = "-0.15 -0.35 1.3 0.7";

  return (
    <svg
      viewBox={viewBox}
      width={width}
      height={height}
      style={{ maxWidth: "100%", height: "auto" }}
      role="img"
      aria-label={`Airfoil outline m=${m} p=${p} t=${t}`}
    >
      <path d={path} fill={AERO_PALETTE.accent} fillOpacity={fillOpacity} stroke={AERO_PALETTE.text} strokeWidth={0.006} />
      <line x1={0} y1={0} x2={1} y2={0} stroke={AERO_PALETTE.border} strokeWidth={0.003} strokeDasharray="0.01,0.01" />
    </svg>
  );
}
