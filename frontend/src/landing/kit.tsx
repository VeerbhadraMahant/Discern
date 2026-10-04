import { useId, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";

export type Tone = "parchment" | "bone" | "ink";

const TONE: Record<Tone, string> = {
  parchment: "",
  bone: "bg-bone",
  ink: "bg-ink text-parchment",
};

/**
 * One landing band: a full-bleed surface, a display-face heading and an optional lead. No eyebrow label.
 * `split` puts the heading in a left column and the content in a wider right column; otherwise they stack.
 */
export function Band({
  id,
  title,
  lead,
  children,
  tone = "parchment",
  split = false,
}: {
  id: string;
  title: string;
  lead?: string;
  children: ReactNode;
  tone?: Tone;
  split?: boolean;
}) {
  const head = (
    <div className={split ? "lg:sticky lg:top-24 lg:self-start" : ""}>
      <h2 id={`${id}-h`} className="type-display-l max-w-[18ch] text-balance">
        {title}
      </h2>
      {lead && <p className="measure mt-5 type-body-l">{lead}</p>}
    </div>
  );
  return (
    <section id={id} tabIndex={-1} aria-labelledby={`${id}-h`} className={`scroll-mt-16 ${TONE[tone]}`}>
      <div className="page py-14 md:py-24">
        {split ? (
          <div className="grid gap-10 lg:grid-cols-12 lg:gap-12">
            <div className="lg:col-span-4">{head}</div>
            <div className="min-w-0 lg:col-span-8">{children}</div>
          </div>
        ) : (
          <>
            {head}
            <div className="mt-10 md:mt-14">{children}</div>
          </>
        )}
      </div>
    </section>
  );
}

const PANEL: Record<Tone, string> = {
  parchment: "bg-parchment text-ink",
  bone: "bg-bone text-ink",
  ink: "bg-ink text-parchment",
};

/** A flat panel with the one directional card shadow. Use the tone that differs from the band it sits on. */
export function Panel({ tone = "bone", className = "", children }: { tone?: Tone; className?: string; children: ReactNode }) {
  return <div className={`rounded-card p-6 shadow-card ${PANEL[tone]} ${className}`}>{children}</div>;
}

/** Four corner ticks around a figure: the frame-bracket motif. The parent must be `relative`. */
export function Corners({ tone = "ink" }: { tone?: "ink" | "parchment" }) {
  const c = tone === "ink" ? "border-ink" : "border-parchment";
  const base = `pointer-events-none absolute size-4 ${c}`;
  return (
    <>
      <span aria-hidden="true" className={`${base} -left-2 -top-2 border-l-[3px] border-t-[3px]`} />
      <span aria-hidden="true" className={`${base} -right-2 -top-2 border-r-[3px] border-t-[3px]`} />
      <span aria-hidden="true" className={`${base} -bottom-2 -left-2 border-b-[3px] border-l-[3px]`} />
      <span aria-hidden="true" className={`${base} -bottom-2 -right-2 border-b-[3px] border-r-[3px]`} />
    </>
  );
}

/** Accessible tabs with roving tabindex, arrow keys, Home and End. */
export function Tabs({
  label,
  tabs,
  render,
  initial = 0,
  vertical = false,
}: {
  label: string;
  tabs: ReadonlyArray<{ id: string; label: string }>;
  render: (index: number) => ReactNode;
  initial?: number;
  /** Tab list beside the panel on wide screens, with Up and Down as the main keys. */
  vertical?: boolean;
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
    <div className={vertical ? "lg:grid lg:grid-cols-12 lg:gap-12" : ""}>
      <div
        role="tablist"
        aria-label={label}
        aria-orientation={vertical ? "vertical" : "horizontal"}
        onKeyDown={onKey}
        className={
          vertical
            ? "flex flex-wrap gap-x-4 gap-y-1 border-b border-ink lg:col-span-5 lg:flex-col lg:flex-nowrap lg:gap-0 lg:self-start lg:border-b-0 lg:border-t"
            : "flex flex-wrap gap-x-4 gap-y-1 border-b border-ink"
        }
      >
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
              className={
                vertical
                  ? `-mb-px min-h-11 border-b-2 px-2 type-label lg:mb-0 lg:flex lg:min-h-14 lg:items-center lg:border-b lg:border-l-4 lg:border-b-ink lg:px-4 lg:text-left lg:type-h3 ${on ? "border-b-ember lg:border-l-ember lg:bg-parchment" : "border-b-transparent lg:border-l-transparent"}`
                  : `-mb-px min-h-11 border-b-2 px-2 type-label ${on ? "border-ember" : "border-transparent"}`
              }
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
          className={vertical ? "pt-6 lg:col-span-7 lg:pt-0" : "pt-6"}
        >
          {i === index && render(i)}
        </div>
      ))}
    </div>
  );
}

export const APP_HREF = "#/clean";
export const DEMO_HREF = "?mock=1#/clean";

export function PrimaryLink({ href, children, onInk = false }: { href: string; children: ReactNode; onInk?: boolean }) {
  return (
    <a
      href={href}
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-tag px-5 py-2 type-label ${onInk ? "bg-parchment text-ink" : "bg-ink text-parchment"}`}
    >
      {children}
    </a>
  );
}

/** A small ember dot: the "found" mark. Drawn as SVG because ember is never a CSS fill on a box. */
export function EmberDot({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true" focusable="false" className="shrink-0">
      <circle cx="6" cy="6" r="5" className="fill-ember" />
    </svg>
  );
}
