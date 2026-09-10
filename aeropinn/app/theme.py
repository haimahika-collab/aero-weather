"""Dark engineering visual theme for the AeroPINN tab, scoped to that tab only.

Gradio's theme system compiles to CSS custom properties inherited down the DOM. We
don't touch app.py's global gr.themes.Soft() (used by the unrelated flight-delay
tabs) — instead we redeclare the same variable names under a scoped #aeropinn-root
selector, injected via a <style> block at the top of the tab. Every descendant
Gradio component already consumes these variables, so this reskins only this tab.
"""

# Single source of truth for color semantics, shared between the CSS below and
# every matplotlib figure in aeropinn/visualization/plots.py — so UI chrome and
# plots never disagree about what a color means.
AERO_PALETTE = {
    "bg": "#0B0F14",
    "panel": "#111820",
    "panel_alt": "#151D26",
    "border": "#26313D",
    "text": "#F2F5F7",
    "text_muted": "#8F9BA8",
    "accent": "#4EA1D3",
    "warn": "#D9A441",
    "error": "#D3564E",
    "prediction": "#4EA1D3",
    "reference": "#C7CDD3",
    "train_geom": "#5A6A78",
    "held_out_geom": "#4EA1D3",
    "current_design": "#F2F5F7",
    "optimized_design": "#D9A441",
}

_P = AERO_PALETTE

AERO_CSS = f"""
#aeropinn-root {{
  --body-background-fill: {_P['bg']};
  --background-fill-primary: {_P['panel']};
  --background-fill-secondary: {_P['panel_alt']};
  --border-color-primary: {_P['border']};
  --border-color-accent: {_P['border']};
  --body-text-color: {_P['text']};
  --body-text-color-subdued: {_P['text_muted']};
  --block-background-fill: {_P['panel']};
  --block-border-color: {_P['border']};
  --block-border-width: 1px;
  --block-radius: 6px;
  --panel-background-fill: {_P['panel_alt']};
  --panel-border-color: {_P['border']};
  --input-background-fill: {_P['bg']};
  --input-border-color: {_P['border']};
  --button-primary-background-fill: {_P['accent']};
  --button-primary-background-fill-hover: #3D84B0;
  --button-primary-text-color: {_P['bg']};
  --button-secondary-background-fill: {_P['panel_alt']};
  --button-secondary-border-color: {_P['border']};
  --button-secondary-text-color: {_P['text']};
  --slider-color: {_P['accent']};
  --shadow-drop: none;
  background: {_P['bg']};
  color: {_P['text']};
  padding: 4px;
}}

#aeropinn-root, #aeropinn-root * {{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
}}

#aeropinn-root .mono,
#aeropinn-root .telemetry-value {{
  font-family: ui-monospace, "JetBrains Mono", "SF Mono", Consolas, monospace !important;
}}

/* Workstation header */
#aeropinn-header {{
  border-bottom: 1px solid {_P['border']};
  padding-bottom: 6px;
  margin-bottom: 6px;
  align-items: center;
}}
#aeropinn-header .app-name {{
  font-weight: 700;
  letter-spacing: 0.06em;
  color: {_P['text']};
}}

/* Nav tab strip */
#aeropinn-nav > .tab-nav {{
  border-bottom: 1px solid {_P['border']};
  background: {_P['bg']};
  gap: 0;
}}
#aeropinn-nav > .tab-nav button {{
  color: {_P['text_muted']};
  border-radius: 0;
  border: none;
  padding: 8px 14px;
  font-size: 0.85em;
  letter-spacing: 0.03em;
}}
#aeropinn-nav > .tab-nav button.selected {{
  color: {_P['accent']};
  border-bottom: 2px solid {_P['accent']};
  background: transparent;
}}

/* Status chips (Model Status / Device / Version / Domain Validity) */
#aeropinn-root .status-chip {{
  border: 1px solid {_P['border']};
  border-radius: 4px;
  padding: 3px 8px;
  font-size: 0.8em;
  display: inline-block;
}}
#aeropinn-root .status-chip-ok {{ color: {_P['accent']}; border-color: {_P['accent']}; }}
#aeropinn-root .status-chip-warn {{ color: {_P['warn']}; border-color: {_P['warn']}; }}
#aeropinn-root .status-chip-error {{ color: {_P['error']}; border-color: {_P['error']}; }}
#aeropinn-root .status-chip-muted {{ color: {_P['text_muted']}; }}

/* Telemetry strip */
#aeropinn-root .telemetry-row {{
  display: flex;
  gap: 18px;
  border: 1px solid {_P['border']};
  border-radius: 6px;
  padding: 8px 14px;
  background: {_P['panel_alt']};
  flex-wrap: wrap;
}}
#aeropinn-root .telemetry-item {{ display: flex; flex-direction: column; gap: 2px; }}
#aeropinn-root .telemetry-label {{
  font-size: 0.7em;
  color: {_P['text_muted']};
  letter-spacing: 0.05em;
  text-transform: uppercase;
}}
#aeropinn-root .telemetry-value {{ font-size: 1.05em; color: {_P['text']}; }}

/* Allowed motion only: subtle transitions, no glow/particles/parallax */
#aeropinn-root button,
#aeropinn-root .gr-panel,
#aeropinn-root .status-chip {{
  transition: background-color 180ms ease, border-color 180ms ease, color 180ms ease;
}}
"""
