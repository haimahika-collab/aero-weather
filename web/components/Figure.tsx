import Link from "next/link";
import { ProvenanceBadge, type Provenance } from "./ProvenanceBadge";

/**
 * Every figure on the site goes through this component. The discriminated union
 * below makes it a TYPE ERROR to pass chart children alongside provenance="pending" —
 * a figure with no real data literally cannot render a chart under its badge, so
 * Figure 04 (data-efficiency curve, not yet run) can never be accidentally given
 * fabricated content. See Plan "Figure system (non-negotiable scientific-integrity
 * mechanism)".
 */
type FigureBaseProps = {
  number: string;
  title: string;
  caption: string;
  sourceNote?: string;
};

type FigureRealProps = FigureBaseProps & {
  provenance: Exclude<Provenance, "pending">;
  children: React.ReactNode;
};

type FigurePendingProps = FigureBaseProps & {
  provenance: "pending";
  children?: never;
  pendingReason?: string;
};

export type FigureProps = FigureRealProps | FigurePendingProps;

export function Figure(props: FigureProps) {
  const { number, title, caption, sourceNote, provenance } = props;
  return (
    <figure
      id={`figure-${number}`}
      className="border border-border rounded-[var(--radius)] bg-panel p-5 my-6 scroll-mt-24"
    >
      <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
        <span className="mono text-xs tracking-wide text-text-muted uppercase">
          Figure {number} — {title}
        </span>
        <div className="flex items-center gap-2">
          <ProvenanceBadge provenance={provenance} />
          <Link href={`/figures/${number}`} className="mono text-xs text-accent hover:underline">
            Open ↗
          </Link>
        </div>
      </div>

      {provenance === "pending" ? (
        <div className="border border-dashed border-warn rounded-[var(--radius)] py-10 px-4 text-center">
          <p className="mono text-sm text-warn mb-1">EXPERIMENT PENDING</p>
          <p className="text-sm text-text-muted" data-typography="editorial">
            {(props as FigurePendingProps).pendingReason ??
              "This experiment has not been run yet. No data exists for this figure — nothing is shown here rather than an estimate or placeholder."}
          </p>
        </div>
      ) : (
        <div className="bg-panel-alt border border-border rounded-[var(--radius)] p-3">
          {(props as FigureRealProps).children}
        </div>
      )}

      <figcaption className="mt-3 text-sm text-text-muted" data-typography="editorial">
        {caption}
      </figcaption>
      {sourceNote && (
        <p className="mono text-[11px] text-text-caption mt-2">source: {sourceNote}</p>
      )}
    </figure>
  );
}
