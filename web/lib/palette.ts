/**
 * Typed re-export of web/data/palette.json, generated from
 * aeropinn/app/theme.py::AERO_PALETTE by aeropinn/export_site_data.py.
 * Import this instead of hardcoding hex values in chart/SVG code.
 */
import paletteJson from "@/data/palette.json";

export interface AeroPalette {
  bg: string;
  panel: string;
  panel_alt: string;
  border: string;
  text: string;
  text_muted: string;
  accent: string;
  warn: string;
  error: string;
  prediction: string;
  reference: string;
  train_geom: string;
  held_out_geom: string;
  current_design: string;
  optimized_design: string;
}

export const AERO_PALETTE: AeroPalette = paletteJson as AeroPalette;
