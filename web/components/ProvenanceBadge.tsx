import { StatusChip } from "./StatusChip";

export type Provenance = "real" | "conceptual" | "pending" | "reference";

const LABELS: Record<Provenance, string> = {
  real: "REAL DATA",
  conceptual: "CONCEPTUAL VISUALIZATION",
  pending: "EXPERIMENT PENDING",
  reference: "INDEPENDENT REFERENCE",
};

const LEVELS: Record<Provenance, "ok" | "warn" | "muted"> = {
  real: "ok",
  conceptual: "muted",
  pending: "warn",
  reference: "ok",
};

export function ProvenanceBadge({ provenance }: { provenance: Provenance }) {
  return <StatusChip text={LABELS[provenance]} level={LEVELS[provenance]} />;
}
