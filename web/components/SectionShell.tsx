export function SectionShell({
  id,
  number,
  title,
  typography,
  children,
}: {
  id: string;
  number: string;
  title: string;
  typography: "editorial" | "engineering";
  children: React.ReactNode;
}) {
  return (
    <section id={id} data-typography={typography} className="scroll-mt-20 py-16 border-b border-border px-6 md:px-12">
      <div className="max-w-4xl mx-auto">
        <h2 className="mono text-sm tracking-widest text-accent mb-6">
          {number} / {title.toUpperCase()}
        </h2>
        {children}
      </div>
    </section>
  );
}
