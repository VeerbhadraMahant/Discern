import { useId } from "react";

/** Illustrations are drawn only from SVG primitives, so no dataset photo is redistributed. */

function useSafeId(): string {
  return useId().replace(/[^a-zA-Z0-9]/g, "");
}

/** Perforated stamp with a sunburst, as in the reference. Ember carries only the border marks. */
export function StampSeal({ className = "" }: { className?: string }) {
  const rays = Array.from({ length: 24 }, (_, i) => {
    const a = (i * Math.PI * 2) / 24;
    const inner = i % 2 === 0 ? 17 : 20;
    return { x1: 50 + Math.cos(a) * inner, y1: 50 + Math.sin(a) * inner, x2: 50 + Math.cos(a) * 33, y2: 50 + Math.sin(a) * 33 };
  });
  return (
    <svg viewBox="0 0 100 100" className={className} role="img" aria-label="Stamp seal: Discern, open models">
      <circle cx="50" cy="50" r="46" className="fill-parchment stroke-ember" strokeWidth="6" strokeDasharray="0 7" strokeLinecap="round" />
      <circle cx="50" cy="50" r="41" className="fill-parchment stroke-ember" strokeWidth="2" />
      <circle cx="50" cy="50" r="36" className="fill-none stroke-ink" strokeWidth="1" />
      {rays.map((r, i) => (
        <line key={i} x1={r.x1} y1={r.y1} x2={r.x2} y2={r.y2} className="stroke-ink" strokeWidth="1.2" />
      ))}
      <circle cx="50" cy="50" r="15" className="fill-ink" />
      <text x="50" y="58" textAnchor="middle" className="fill-parchment" style={{ fontFamily: "var(--font-display)", fontSize: 24, letterSpacing: "-0.04em" }}>
        D
      </text>
    </svg>
  );
}

/** One stylised road frame. `clean=false` is the degraded input, `clean=true` is the cleaned frame with boxes. */
export function Frame({ clean }: { clean: boolean }) {
  const id = useSafeId();
  return (
    <svg viewBox="0 0 240 160" className="block h-auto w-full" aria-hidden="true">
      <defs>
        <pattern id={`${id}h`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
          <line x1="0" y1="0" x2="0" y2="6" className="stroke-ink" strokeWidth="1" opacity="0.45" />
        </pattern>
        <pattern id={`${id}d`} width="5" height="5" patternUnits="userSpaceOnUse">
          <circle cx="1.5" cy="1.5" r="0.8" className="fill-ink" opacity="0.5" />
          <circle cx="4" cy="4" r="0.6" className="fill-ink" opacity="0.4" />
        </pattern>
      </defs>
      <rect width="240" height="160" className="fill-parchment" />
      <polygon points="0,78 240,78 240,160 0,160" className="fill-bone" />
      <polygon points="104,78 136,78 214,160 26,160" className="fill-parchment stroke-ink" strokeWidth="1.5" />
      <line x1="0" y1="78" x2="240" y2="78" className="stroke-ink" strokeWidth="1.5" />
      <line x1="120" y1="84" x2="120" y2="96" className="stroke-ink" strokeWidth="2" />
      <line x1="120" y1="106" x2="120" y2="124" className="stroke-ink" strokeWidth="2" />
      <line x1="120" y1="136" x2="120" y2="160" className="stroke-ink" strokeWidth="2" />
      {/* car */}
      <rect x="150" y="98" width="46" height="18" className="fill-ink" />
      <rect x="158" y="90" width="28" height="10" className="fill-ink" />
      <circle cx="160" cy="118" r="5" className="fill-parchment stroke-ink" strokeWidth="2" />
      <circle cx="187" cy="118" r="5" className="fill-parchment stroke-ink" strokeWidth="2" />
      {/* far car */}
      <rect x="62" y="74" width="20" height="8" className="fill-ink" />
      {/* person */}
      <circle cx="46" cy="94" r="4" className="fill-ink" />
      <line x1="46" y1="98" x2="46" y2="116" className="stroke-ink" strokeWidth="3" />
      <line x1="46" y1="116" x2="41" y2="128" className="stroke-ink" strokeWidth="3" />
      <line x1="46" y1="116" x2="52" y2="128" className="stroke-ink" strokeWidth="3" />
      {!clean && (
        <>
          <rect width="240" height="160" fill={`url(#${id}h)`} />
          <rect width="240" height="160" fill={`url(#${id}d)`} />
          <rect x="0" y="40" width="240" height="46" className="fill-parchment" opacity="0.45" />
        </>
      )}
      {clean && (
        <g>
          <rect x="146" y="86" width="54" height="38" fill="none" className="stroke-ink" strokeWidth="2" strokeDasharray="6 3" />
          <rect x="146" y="70" width="34" height="16" className="fill-ink" />
          <text x="150" y="82" className="fill-parchment" style={{ fontSize: 12 }}>
            car
          </text>
          <rect x="34" y="84" width="24" height="48" fill="none" className="stroke-ink" strokeWidth="2" strokeDasharray="6 3" />
          <rect x="30" y="68" width="54" height="16" className="fill-ink" />
          <text x="34" y="80" className="fill-parchment" style={{ fontSize: 12 }}>
            person
          </text>
        </g>
      )}
    </svg>
  );
}

/** Three detectors propose boxes that merge into one. */
export function FuseArt() {
  return (
    <svg viewBox="0 0 240 200" className="block h-auto w-full" role="img" aria-label="Illustration: three detectors draw slightly different boxes around one object and the boxes are merged into one.">
      <rect width="240" height="200" className="fill-bone" />
      <rect x="16" y="26" width="70" height="52" fill="none" className="stroke-ink" strokeWidth="2" strokeDasharray="8 4" />
      <rect x="22" y="34" width="70" height="52" fill="none" className="stroke-ink" strokeWidth="2" strokeDasharray="2 4" />
      <rect x="12" y="38" width="70" height="52" fill="none" className="stroke-ink" strokeWidth="2" strokeDasharray="12 3 2 3" />
      <text x="16" y="108" className="fill-ink" style={{ fontSize: 12 }}>
        three proposals
      </text>
      <path d="M100 62 H140 M132 54 L142 62 L132 70" fill="none" className="stroke-ink" strokeWidth="2" />
      <rect x="156" y="30" width="68" height="56" fill="none" className="stroke-ink" strokeWidth="3" />
      <circle cx="190" cy="58" r="14" className="fill-ink" />
      <text x="156" y="108" className="fill-ink" style={{ fontSize: 12 }}>
        one fused box
      </text>
      <line x1="16" y1="130" x2="224" y2="130" className="stroke-ink" strokeWidth="1" />
      <text x="16" y="154" className="fill-ink" style={{ fontSize: 12 }}>
        grouped by overlap and
      </text>
      <text x="16" y="170" className="fill-ink" style={{ fontSize: 12 }}>
        crop similarity
      </text>
    </svg>
  );
}

/** A timeline with three tracks and a playhead. */
export function TrackArt() {
  const id = useSafeId();
  return (
    <svg viewBox="0 0 240 200" className="block h-auto w-full" role="img" aria-label="Illustration: three object tracks drawn as bars along a video timeline with a playhead.">
      <defs>
        <pattern id={`${id}a`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" className="stroke-ink" strokeWidth="2" />
        </pattern>
        <pattern id={`${id}b`} width="5" height="5" patternUnits="userSpaceOnUse">
          <circle cx="2.5" cy="2.5" r="1.2" className="fill-ink" />
        </pattern>
      </defs>
      <rect width="240" height="200" className="fill-bone" />
      <line x1="16" y1="150" x2="224" y2="150" className="stroke-ink" strokeWidth="1.5" />
      {[16, 68, 120, 172, 224].map((x) => (
        <line key={x} x1={x} y1="146" x2={x} y2="154" className="stroke-ink" strokeWidth="1.5" />
      ))}
      <text x="16" y="170" className="fill-ink" style={{ fontSize: 12 }}>
        start
      </text>
      <text x="224" y="170" textAnchor="end" className="fill-ink" style={{ fontSize: 12 }}>
        end
      </text>
      <rect x="40" y="28" width="96" height="20" fill={`url(#${id}a)`} className="stroke-ink" strokeWidth="1.5" />
      <rect x="84" y="60" width="116" height="20" className="fill-parchment stroke-ink" strokeWidth="1.5" />
      <rect x="140" y="92" width="72" height="20" fill={`url(#${id}b)`} className="stroke-ink" strokeWidth="1.5" />
      <text x="18" y="43" className="fill-ink" style={{ fontSize: 12 }}>
        1
      </text>
      <text x="62" y="75" className="fill-ink" style={{ fontSize: 12 }}>
        2
      </text>
      <text x="118" y="107" className="fill-ink" style={{ fontSize: 12 }}>
        3
      </text>
      <line x1="150" y1="16" x2="150" y2="150" className="stroke-ink" strokeWidth="3" />
      <polygon points="143,12 157,12 150,22" className="fill-ink" />
    </svg>
  );
}
