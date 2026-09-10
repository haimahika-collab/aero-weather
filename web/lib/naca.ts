/**
 * NACA 4-digit airfoil geometry — direct TypeScript port of aeropinn/geometry/naca.py.
 * Pure math, no dependencies. Used wherever the site renders a real airfoil outline
 * (Figure 01, Section 02's scroll scene, Section 03's precomputed-case views) so
 * geometry is always the same real parameterization the trained model was built on,
 * never a generic/stock airfoil shape.
 *
 * Verify against the Python source before trusting in a figure: sample a few (m,p,t,x)
 * points and diff naca_thickness/naca_camber against aeropinn/geometry/naca.py's output.
 */

export interface Point {
  x: number;
  y: number;
}

const THICKNESS_COEFFS = [0.2969, -0.126, -0.3516, 0.2843, -0.1015] as const;

export function nacaThickness(x: number, t: number): number {
  const [a0, a1, a2, a3, a4] = THICKNESS_COEFFS;
  return 5 * t * (a0 * Math.sqrt(x) + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4);
}

export function nacaCamber(x: number, m: number, p: number): number {
  if (m === 0 || p === 0) return 0;
  if (x < p) {
    return (m / p ** 2) * (2 * p * x - x ** 2);
  }
  return (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x ** 2);
}

export function nacaCamberSlope(x: number, m: number, p: number): number {
  if (m === 0 || p === 0) return 0;
  if (x < p) {
    return ((2 * m) / p ** 2) * (p - x);
  }
  return ((2 * m) / (1 - p) ** 2) * (p - x);
}

function cosineSpacing(nPoints: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < nPoints; i++) {
    const beta = (Math.PI * i) / (nPoints - 1);
    out.push(0.5 * (1 - Math.cos(beta)));
  }
  return out;
}

/** Upper and lower surface coordinates, leading edge to trailing edge, cosine-spaced. */
export function naca4Surface(m: number, p: number, t: number, nPoints = 150): { upper: Point[]; lower: Point[] } {
  const xs = cosineSpacing(nPoints);
  const upper: Point[] = [];
  const lower: Point[] = [];

  for (const x of xs) {
    const yt = nacaThickness(x, t);
    const yc = nacaCamber(x, m, p);
    const dyc = nacaCamberSlope(x, m, p);
    const theta = Math.atan(dyc);

    upper.push({ x: x - yt * Math.sin(theta), y: yc + yt * Math.cos(theta) });
    lower.push({ x: x + yt * Math.sin(theta), y: yc - yt * Math.cos(theta) });
  }

  return { upper, lower };
}

/** Closed boundary loop: lower LE->TE then upper TE->LE (matches naca4_boundary_loop /
 * physics/forces.py's convention). */
export function naca4BoundaryLoop(m: number, p: number, t: number, nPoints = 150): Point[] {
  const { upper, lower } = naca4Surface(m, p, t, nPoints);
  const upperReversed = [...upper].reverse().slice(1);
  return [...lower, ...upperReversed];
}

/** SVG path `d` string for the closed airfoil outline, chord normalized to [0,1]. */
export function naca4SvgPath(m: number, p: number, t: number, nPoints = 100): string {
  const loop = naca4BoundaryLoop(m, p, t, nPoints);
  const [first, ...rest] = loop;
  const cmds = [`M ${first.x.toFixed(5)} ${first.y.toFixed(5)}`];
  for (const pt of rest) {
    cmds.push(`L ${pt.x.toFixed(5)} ${pt.y.toFixed(5)}`);
  }
  cmds.push("Z");
  return cmds.join(" ");
}

export function nacaDesignation(m: number, p: number, t: number): string {
  const d1 = Math.round(m * 100);
  const d2 = Math.round(p * 10);
  const d34 = Math.round(t * 100);
  return `NACA ${d1}${d2}${String(d34).padStart(2, "0")}`;
}
