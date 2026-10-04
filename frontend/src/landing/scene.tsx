import { useId } from "react";
import type { ReactNode } from "react";
import type { SheetFrame } from "./facts";

/**
 * The page's illustrated road scene, drawn only from SVG primitives (no dataset photo is redistributed).
 * Fog and noise are patterns (hatch and stipple), never a filter, so the pattern carries the meaning and colour does not.
 */

export const SCENE_W = 720;
export const SCENE_H = 460;

export function useSafeId(): string {
  return useId().replace(/[^a-zA-Z0-9]/g, "");
}

export interface Placement {
  carX: number;
  farX: number;
  personX: number;
}

export const DEFAULT_PLACEMENT: Placement = { carX: 500, farX: 410, personX: 230 };

const CAR_Y = 396;
const FAR_Y = 292;
const PERSON_Y = 372;
const FAR_S = 0.34;

/** Evidence boxes for the three objects, in scene units. */
export function boxes(p: Placement) {
  return {
    car: { x: p.carX - 88, y: CAR_Y - 86, w: 176, h: 108 },
    far: { x: p.farX - 32, y: FAR_Y - 31, w: 64, h: 44 },
    person: { x: p.personX - 30, y: PERSON_Y - 128, w: 60, h: 140 },
  };
}

export function SceneDefs({ id }: { id: string }) {
  return (
    <defs>
      <pattern id={`${id}win`} width="9" height="12" patternUnits="userSpaceOnUse">
        <rect x="2" y="3" width="4" height="5" className="fill-ink" opacity="0.55" />
      </pattern>
      <pattern id={`${id}hatch`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
        <line x1="0" y1="0" x2="0" y2="7" className="stroke-ink" strokeWidth="1.2" opacity="0.4" />
      </pattern>
      <pattern id={`${id}stip`} width="7" height="7" patternUnits="userSpaceOnUse">
        <circle cx="1.6" cy="1.6" r="1" className="fill-ink" opacity="0.6" />
        <circle cx="5.2" cy="4.8" r="0.8" className="fill-ink" opacity="0.5" />
      </pattern>
    </defs>
  );
}

const SKYLINE: ReadonlyArray<readonly [number, number, number]> = [
  [14, 62, 84],
  [84, 52, 126],
  [144, 72, 62],
  [226, 44, 104],
  [278, 52, 70],
  [438, 62, 92],
  [508, 52, 134],
  [568, 74, 72],
  [650, 66, 112],
];

function Car({ x, y, s }: { x: number; y: number; s: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <line x1="-86" y1="14" x2="86" y2="14" className="stroke-ink" strokeWidth="2" opacity="0.5" />
      <rect x="-75" y="-40" width="150" height="40" rx="5" className="fill-parchment stroke-ink" strokeWidth="2.5" />
      <polygon points="-55,-40 -42,-72 42,-72 55,-40" className="fill-parchment stroke-ink" strokeWidth="2.5" strokeLinejoin="round" />
      <polygon points="-45,-44 -35,-66 35,-66 45,-44" className="fill-ink" />
      <rect x="-70" y="-34" width="22" height="10" className="fill-ink" />
      <rect x="48" y="-34" width="22" height="10" className="fill-ink" />
      <rect x="-16" y="-24" width="32" height="12" className="fill-parchment stroke-ink" strokeWidth="2" />
      <rect x="-66" y="-2" width="24" height="14" className="fill-ink" />
      <rect x="42" y="-2" width="24" height="14" className="fill-ink" />
    </g>
  );
}

function Person({ x, y }: { x: number; y: number }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <line x1="-6" y1="-52" x2="-9" y2="0" className="stroke-ink" strokeWidth="6" />
      <line x1="6" y1="-52" x2="10" y2="0" className="stroke-ink" strokeWidth="6" />
      <line x1="-14" y1="-92" x2="-20" y2="-60" className="stroke-ink" strokeWidth="5" />
      <line x1="14" y1="-92" x2="20" y2="-62" className="stroke-ink" strokeWidth="5" />
      <polygon points="-14,-96 14,-96 16,-50 -16,-50" className="fill-ink" />
      <circle cx="0" cy="-108" r="10" className="fill-parchment stroke-ink" strokeWidth="2.5" />
    </g>
  );
}

/** The scene itself, with no overlay. */
export function SceneArt({ id, p = DEFAULT_PLACEMENT }: { id: string; p?: Placement }) {
  return (
    <g>
      <rect width={SCENE_W} height={SCENE_H} className="fill-parchment" />
      {SKYLINE.map(([x, w, h]) => (
        <g key={x}>
          <rect x={x} y={232 - h} width={w} height={h} className="fill-bone stroke-ink" strokeWidth="2" />
          <rect x={x} y={232 - h} width={w} height={h} fill={`url(#${id}win)`} />
        </g>
      ))}
      <rect x="0" y="232" width={SCENE_W} height={SCENE_H - 232} className="fill-parchment" />
      <polygon points="330,232 390,232 770,460 -50,460" className="fill-bone stroke-ink" strokeWidth="2" />
      <line x1="0" y1="232" x2={SCENE_W} y2="232" className="stroke-ink" strokeWidth="2" />
      {(
        [
          [240, 248, 2],
          [258, 272, 3],
          [288, 312, 4],
          [336, 372, 5],
          [402, 452, 7],
        ] as const
      ).map(([a, b, w]) => (
        <line key={a} x1="360" y1={a} x2="360" y2={b} className="stroke-ink" strokeWidth={w} />
      ))}
      <line x1="620" y1="330" x2="620" y2="120" className="stroke-ink" strokeWidth="4" />
      <line x1="620" y1="120" x2="592" y2="120" className="stroke-ink" strokeWidth="4" />
      <rect x="576" y="118" width="18" height="8" className="fill-ink" />
      <Car x={p.farX} y={FAR_Y} s={FAR_S} />
      <Person x={p.personX} y={PERSON_Y} />
      <Car x={p.carX} y={CAR_Y} s={1} />
    </g>
  );
}

/** Fog and noise: translucent bands, hatching, stipple and a few dropped scanlines. */
export function DegradeLayer({ id }: { id: string }) {
  return (
    <g>
      <rect width={SCENE_W} height={SCENE_H} className="fill-parchment" opacity="0.5" />
      <rect y="150" width={SCENE_W} height="110" className="fill-parchment" opacity="0.55" />
      <rect y="292" width={SCENE_W} height="44" className="fill-parchment" opacity="0.32" />
      <rect width={SCENE_W} height={SCENE_H} fill={`url(#${id}hatch)`} />
      <rect width={SCENE_W} height={SCENE_H} fill={`url(#${id}stip)`} />
      {[94, 187, 301, 378].map((y) => (
        <g key={y}>
          <rect y={y} width={SCENE_W} height="3" className="fill-parchment" opacity="0.85" />
          <line x1="0" y1={y + 6} x2={SCENE_W} y2={y + 6} className="stroke-ink" strokeWidth="1" opacity="0.3" />
        </g>
      ))}
    </g>
  );
}

/** Frame brackets (four corner ticks) plus the ember dot: "this was found and can be inspected". */
export function Brackets({ x, y, w, h, len = 18 }: { x: number; y: number; w: number; h: number; len?: number }) {
  const l = Math.min(len, w / 3, h / 3);
  return (
    <g>
      <path
        d={`M${x} ${y + l}V${y}H${x + l}M${x + w - l} ${y}H${x + w}V${y + l}M${x + w} ${y + h - l}V${y + h}H${x + w - l}M${x + l} ${y + h}H${x}V${y + h - l}`}
        fill="none"
        className="stroke-ink"
        strokeWidth="3.5"
        strokeLinecap="square"
      />
      <circle cx={x + w} cy={y} r="6" className="fill-ember stroke-parchment" strokeWidth="2" />
    </g>
  );
}

/** A scene as its own SVG. Children draw on top, in scene units. */
export function Scene({ p = DEFAULT_PLACEMENT, degraded = false, view, children }: { p?: Placement; degraded?: boolean; view?: string; children?: ReactNode }) {
  const id = useSafeId();
  return (
    <svg viewBox={view ?? `0 0 ${SCENE_W} ${SCENE_H}`} className="scene block h-auto w-full" aria-hidden="true" focusable="false">
      <SceneDefs id={id} />
      <SceneArt id={id} p={p} />
      {degraded && <DegradeLayer id={id} />}
      {children}
    </svg>
  );
}

/** Stills are cropped to the road, so the car and its evidence are large enough to read at contact-sheet size. */
const STILL_VIEW = "130 130 520 330";

const STILL_PLACEMENT: Record<SheetFrame["id"], Placement> = {
  degraded: { carX: 380, farX: 410, personX: 200 },
  cleaned: { carX: 420, farX: 410, personX: 210 },
  fused: { carX: 460, farX: 410, personX: 220 },
  tracked: { carX: 500, farX: 410, personX: 230 },
  answered: { carX: 540, farX: 410, personX: 240 },
};

/** One still of the contact sheet. Decorative: the caption beside it carries the meaning. */
export function Still({ kind }: { kind: SheetFrame["id"] }) {
  const p = STILL_PLACEMENT[kind];
  const b = boxes(p).car;
  const cy = b.y + b.h / 2;
  return (
    <Scene p={p} degraded={kind === "degraded"} view={STILL_VIEW}>
      {kind === "fused" && (
        <g fill="none" className="stroke-ink" strokeWidth="3">
          <rect x={b.x - 6} y={b.y + 4} width={b.w} height={b.h - 6} strokeDasharray="12 6" />
          <rect x={b.x + 8} y={b.y - 4} width={b.w - 4} height={b.h + 2} strokeDasharray="3 6" />
          <rect x={b.x} y={b.y + 8} width={b.w + 6} height={b.h - 10} strokeDasharray="18 5 3 5" />
        </g>
      )}
      {kind === "tracked" && (
        <g>
          <polyline points={`${b.x + b.w / 2 - 120},${cy} ${b.x + b.w / 2 - 80},${cy} ${b.x + b.w / 2 - 40},${cy}`} fill="none" className="stroke-ink" strokeWidth="3" strokeDasharray="2 10" strokeLinecap="round" />
          {[-120, -80, -40].map((dx) => (
            <circle key={dx} cx={b.x + b.w / 2 + dx} cy={cy} r="7" className="fill-ember stroke-parchment" strokeWidth="2" />
          ))}
          <Brackets {...b} />
        </g>
      )}
      {kind === "answered" && (
        <g>
          <Brackets {...b} />
          <rect x={b.x} y={b.y - 44} width="76" height="36" className="fill-ink" />
          <path d={`M${b.x + 20} ${b.y - 26}l10 10l24 -20`} fill="none" className="stroke-parchment" strokeWidth="5" strokeLinecap="square" />
        </g>
      )}
    </Scene>
  );
}
