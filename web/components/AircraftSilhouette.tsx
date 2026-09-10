import { AERO_PALETTE } from "@/lib/palette";

/**
 * Schematic top-down aircraft line-art — deliberately simple/technical, not a stock
 * photo or decorative render (Plan: "avoid stock airplane images"). CONCEPTUAL
 * visualization, not derived from any real aircraft dataset — always paired with a
 * "conceptual diagram" caption where used. Shared between Section 01 (hero) and
 * Section 02 (aircraft->wing->airfoil story), per the plan's reuse note.
 */
export function AircraftSilhouette({
  width = 480,
  height = 260,
  highlightWing = false,
}: {
  width?: number;
  height?: number;
  highlightWing?: boolean;
}) {
  return (
    <svg
      viewBox="0 0 480 260"
      width={width}
      height={height}
      style={{ maxWidth: "100%", height: "auto" }}
      role="img"
      aria-label="Schematic aircraft diagram (conceptual)"
    >
      {/* fuselage */}
      <path
        d="M 40 130 Q 80 118 180 122 L 400 128 Q 430 129 440 130 Q 430 131 400 132 L 180 138 Q 80 142 40 130 Z"
        fill="none"
        stroke={AERO_PALETTE.text_muted}
        strokeWidth={1.5}
      />
      {/* wings */}
      <path
        d="M 190 124 L 90 40 L 105 38 L 215 120 Z"
        fill={highlightWing ? AERO_PALETTE.accent : "none"}
        fillOpacity={highlightWing ? 0.25 : 0}
        stroke={highlightWing ? AERO_PALETTE.accent : AERO_PALETTE.text_muted}
        strokeWidth={highlightWing ? 2 : 1.5}
      />
      <path
        d="M 190 136 L 90 220 L 105 222 L 215 140 Z"
        fill={highlightWing ? AERO_PALETTE.accent : "none"}
        fillOpacity={highlightWing ? 0.25 : 0}
        stroke={highlightWing ? AERO_PALETTE.accent : AERO_PALETTE.text_muted}
        strokeWidth={highlightWing ? 2 : 1.5}
      />
      {/* tail */}
      <path d="M 360 126 L 420 108 L 424 112 L 372 129 Z" fill="none" stroke={AERO_PALETTE.text_muted} strokeWidth={1.5} />
      <path d="M 360 134 L 420 152 L 424 148 L 372 131 Z" fill="none" stroke={AERO_PALETTE.text_muted} strokeWidth={1.5} />
      <path d="M 420 122 L 445 130 L 420 138 Z" fill="none" stroke={AERO_PALETTE.text_muted} strokeWidth={1.5} />
    </svg>
  );
}
