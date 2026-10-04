import { useId, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";

/** Landing section: a rule, a display-face heading and an optional lead paragraph. No eyebrow label. */
export function Section({
  id,
  title,
  lead,
  children,
  tone = "parchment",
}: {
  id: string;
  title: string;
  lead?: string;
  children: ReactNode;
  tone?: "parchment" | "ink";
}) {
  return (
    <section
      id={id}
      tabIndex={-1}
      aria-labelledby={`${id}-h`}
      className={`scroll-mt-16 border-t border-ink ${tone === "ink" ? "bg-ink text-parchment" : ""}`}
    >
      <div className="mx-auto max-w-[1440px] px-4 py-12 md:px-8 md:py-16">
        <h2 id={`${id}-h`} className="type-display-l max-w-[24ch]">
          {title}
        </h2>
        {lead && <p className="measure mt-4 type-body-l">{lead}</p>}
        <div className="mt-8 md:mt-10">{children}</div>
      </div>
    </section>
  );
}

/** Accessible tabs with roving tabindex, arrow keys, Home and End. */
export function Tabs({
  label,
  tabs,
  render,
  initial = 0,
}: {
  label: string;
  tabs: ReadonlyArray<{ id: string; label: string }>;
  render: (index: number) => ReactNode;
  initial?: number;
}) {
  const base = useId();
  const [index, setIndex] = useState(initial);
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const move = (next: number) => {
    const n = (next + tabs.length) % tabs.length;
    setIndex(n);
    refs.current[n]?.focus();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "ArrowDown") move(index + 1);
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") move(index - 1);
    else if (e.key === "Home") move(0);
    else if (e.key === "End") move(tabs.length - 1);
    else return;
    e.preventDefault();
  };
  return (
    <div>
      <div role="tablist" aria-label={label} onKeyDown={onKey} className="flex flex-wrap gap-x-4 gap-y-1 border-b border-ink">
        {tabs.map((t, i) => {
          const on = i === index;
          return (
            <button
              key={t.id}
              ref={(el) => {
                refs.current[i] = el;
              }}
              type="button"
              role="tab"
              id={`${base}-t-${t.id}`}
              aria-selected={on}
              aria-controls={`${base}-p-${t.id}`}
              tabIndex={on ? 0 : -1}
              onClick={() => setIndex(i)}
              className={`-mb-px min-h-11 border-b-2 px-2 type-label ${on ? "border-ember" : "border-transparent"}`}
            >
              {t.label}
            </button>
          );
        })}
      </div>
      {tabs.map((t, i) => (
        <div
          key={t.id}
          role="tabpanel"
          id={`${base}-p-${t.id}`}
          aria-labelledby={`${base}-t-${t.id}`}
          hidden={i !== index}
          tabIndex={0}
          className="pt-6"
        >
          {i === index && render(i)}
        </div>
      ))}
    </div>
  );
}
