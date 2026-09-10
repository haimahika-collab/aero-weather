export type ChipLevel = "ok" | "warn" | "error" | "muted";

export function StatusChip({ text, level }: { text: string; level: ChipLevel }) {
  return <span className={`status-chip status-chip-${level}`}>{text}</span>;
}
