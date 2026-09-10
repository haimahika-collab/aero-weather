export function ExperimentMetadataBlock({
  items,
}: {
  items: { label: string; value: string }[];
}) {
  return (
    <div className="border border-border rounded-[var(--radius)] bg-panel-alt p-3 flex flex-wrap gap-x-6 gap-y-2 mono text-xs">
      {items.map((item) => (
        <div key={item.label} className="flex flex-col gap-0.5">
          <span className="text-text-muted uppercase tracking-wide text-[10px]">{item.label}</span>
          <span className="text-text" data-numeric>{item.value}</span>
        </div>
      ))}
    </div>
  );
}
