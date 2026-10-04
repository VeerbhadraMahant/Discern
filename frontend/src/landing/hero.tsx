import { CaretLeft, CaretRight, PlayCircle } from "@phosphor-icons/react";
import { useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import { Stamp } from "../components/ui";
import { HERO_STATUS } from "./facts";
import { prefersReducedMotion } from "./hooks";
import { APP_HREF, DEMO_HREF, EmberDot, PrimaryLink } from "./kit";
import { Brackets, DEFAULT_PLACEMENT, DegradeLayer, SCENE_H, SCENE_W, SceneArt, SceneDefs, boxes, useSafeId } from "./scene";

/** The objects the evidence brackets mark. A bracket appears once the divider has passed the object's centre. */
const OBJECTS = (() => {
  const b = boxes(DEFAULT_PLACEMENT);
  return [
    { id: "person", label: "person", box: b.person, end: false },
    { id: "far", label: "car", box: b.far, end: false },
    { id: "car", label: "car", box: b.car, end: true },
  ] as const;
})();

const TRACKS: ReadonlyArray<{ id: string; label: string; from: number; to: number }> = [
  { id: "person", label: "person, 0:01 to 0:05", from: 1, to: 5 },
  { id: "car", label: "car, 0:03 to 0:07", from: 3, to: 7 },
];
const CLIP = 8;

/** Where the divider sits, 0 to 100, clamped to whole percents. */
function clamp(n: number): number {
  return Math.min(100, Math.max(0, Math.round(n)));
}

export function describeFocus(pos: number): string {
  const x = (pos / 100) * SCENE_W;
  const names = OBJECTS.filter((o) => x < o.box.x + o.box.w / 2).map((o) => o.label);
  return `Divider at ${pos} percent. Outlined on the cleaned side: ${names.length ? names.join(", ") : "nothing yet"}.`;
}

/** The signature figure: drag from fog to clean. Operable by pointer and by keyboard. Illustration only. */
export function FocusPull() {
  const id = useSafeId();
  const [pos, setPos] = useState(45);
  const frame = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const x = (pos / 100) * SCENE_W;
  const found = OBJECTS.filter((o) => x < o.box.x + o.box.w / 2);

  const fromPointer = (e: PointerEvent) => {
    const r = frame.current?.getBoundingClientRect();
    if (!r || r.width === 0) return;
    setPos(clamp(((e.clientX - r.left) / r.width) * 100));
  };
  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 5;
    const k = e.key;
    if (k === "ArrowRight" || k === "ArrowUp") setPos((v) => clamp(v + step));
    else if (k === "ArrowLeft" || k === "ArrowDown") setPos((v) => clamp(v - step));
    else if (k === "PageUp") setPos((v) => clamp(v + 10));
    else if (k === "PageDown") setPos((v) => clamp(v - 10));
    else if (k === "Home") setPos(0);
    else if (k === "End") setPos(100);
    else return;
    e.preventDefault();
  };

  return (
    <figure>
      <div ref={frame} className="relative mx-2 border-2 border-ink bg-parchment">
        <svg viewBox={`0 0 ${SCENE_W} ${SCENE_H}`} className="scene block h-auto w-full" aria-hidden="true" focusable="false">
          <SceneDefs id={id} />
          <clipPath id={`${id}clip`}>
            <rect x="0" y="0" width={x} height={SCENE_H} />
          </clipPath>
          <SceneArt id={id} />
          <g clipPath={`url(#${id}clip)`}>
            <DegradeLayer id={id} />
          </g>
          {found.map((o) => (
            <Brackets key={o.id} {...o.box} />
          ))}
          <line x1={x} y1="0" x2={x} y2={SCENE_H} className="stroke-parchment" strokeWidth="7" />
          <line x1={x} y1="0" x2={x} y2={SCENE_H} className="stroke-ink" strokeWidth="3" />
        </svg>

        {found.map((o) => (
          <span
            key={o.id}
            aria-hidden="true"
            className="pointer-events-none absolute bg-ink px-1.5 py-0.5 type-caption font-medium text-parchment"
            style={{ left: `${((o.end ? o.box.x + o.box.w : o.box.x) / SCENE_W) * 100}%`, top: `${(o.box.y / SCENE_H) * 100}%`, transform: o.end ? "translate(calc(-100% - 12px), -100%)" : "translateY(-100%)" }}
          >
            {o.label}
          </span>
        ))}
        <span
          aria-hidden="true"
          className={`pointer-events-none absolute left-2 top-2 border border-ink bg-parchment px-2 py-0.5 type-label ${pos < 22 ? "invisible" : ""}`}
        >
          Degraded input
        </span>
        <span
          aria-hidden="true"
          className={`pointer-events-none absolute right-2 top-2 border border-ink bg-parchment px-2 py-0.5 type-label ${pos > 78 ? "invisible" : ""}`}
        >
          Cleaned, with evidence
        </span>

        <div
          role="slider"
          tabIndex={0}
          aria-label="Focus pull: move the divider between the degraded and the cleaned frame"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pos}
          aria-valuetext={describeFocus(pos)}
          onKeyDown={onKey}
          onPointerDown={(e) => {
            if (e.button !== 0) return;
            dragging.current = true;
            e.currentTarget.setPointerCapture?.(e.pointerId);
            fromPointer(e);
          }}
          onPointerMove={(e) => {
            if (dragging.current) fromPointer(e);
          }}
          onPointerUp={() => {
            dragging.current = false;
          }}
          onPointerCancel={() => {
            dragging.current = false;
          }}
          className="absolute inset-0 cursor-ew-resize touch-pan-y"
        >
          <span
            aria-hidden="true"
            className="absolute top-1/2 flex size-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-tag border-2 border-ink bg-parchment"
            style={{ left: `${pos}%` }}
          >
            <CaretLeft size={16} aria-hidden="true" />
            <CaretRight size={16} aria-hidden="true" />
          </span>
        </div>
      </div>

      <div className="mx-2 mt-5" aria-hidden="true">
        {TRACKS.map((t) => (
          <div key={t.id} className="relative h-9">
            <span className="absolute top-0 whitespace-nowrap type-caption num" style={{ left: `${(t.from / CLIP) * 100}%` }}>
              {t.label}
            </span>
            <span className="absolute bottom-1.5 h-1.5 bg-ink" style={{ left: `${(t.from / CLIP) * 100}%`, width: `${((t.to - t.from) / CLIP) * 100}%` }} />
            <span className="absolute bottom-0 -translate-x-1/2" style={{ left: `${(t.from / CLIP) * 100}%` }}>
              <EmberDot size={10} />
            </span>
          </div>
        ))}
        <div className="relative mt-1 h-9 border-t-2 border-ink">
          {[0, 2, 4, 6, 8].map((s) => (
            <span
              key={s}
              className={`absolute top-0 flex flex-col ${s === 0 ? "items-start" : s === CLIP ? "items-end" : "-translate-x-1/2 items-center"}`}
              style={s === CLIP ? { right: 0 } : { left: `${(s / CLIP) * 100}%` }}
            >
              <span className="h-2 w-0.5 bg-ink" />
              <span className="type-caption num">{`0:0${s}`}</span>
            </span>
          ))}
        </div>
      </div>

      <figcaption className="mx-2 mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 type-body-s">
        <Stamp>Illustration, not a recorded result</Stamp>
        <span>Drag the divider, or focus it and use the arrow keys, to pull the frame into focus.</span>
      </figcaption>
    </figure>
  );
}

export function Hero() {
  // The headline resolves from a degraded look once. Skipped entirely under reduced motion; the text is readable either way.
  const [resolve] = useState(() => !prefersReducedMotion());
  return (
    <section aria-label="Introduction" id="top" tabIndex={-1}>
      <div className="page pb-14 pt-10 md:pb-24 md:pt-16">
        <div className="relative max-w-[1000px]">
          <h1 className={`type-display-xl ${resolve ? "hero-resolve" : ""}`}>Ask a video a question and see the evidence.</h1>
          {resolve && <span aria-hidden="true" className="hero-grain" />}
        </div>
        <div className="mt-10 grid gap-10 lg:mt-14 lg:grid-cols-12 lg:gap-14">
          <div className="lg:col-span-5">
            <p className="measure type-body-l">
              Discern cleans fog, night and rain from each shot, finds and tracks what is there, and answers with boxes and timestamps you can check.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3">
              <PrimaryLink href={APP_HREF}>Open the app</PrimaryLink>
              <a href={DEMO_HREF} className="link">
                <PlayCircle size={20} aria-hidden="true" />
                Try with demo data
              </a>
            </div>
            <p className="mt-6 type-body-s">{HERO_STATUS}</p>
          </div>
          <div className="min-w-0 lg:col-span-7">
            <FocusPull />
          </div>
        </div>
      </div>
    </section>
  );
}
