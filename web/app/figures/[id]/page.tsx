import { redirect } from "next/navigation";

/** Citable per-figure permalink. Redirects to the anchor on the main page rather than
 * duplicating each chart's rendering logic in a second place. */
export default async function FigurePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  redirect(`/#figure-${encodeURIComponent(id)}`);
}
