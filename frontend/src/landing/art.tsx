import { useId } from "react";

/** Illustrations are drawn only from SVG primitives, so no dataset photo is redistributed. */

function useSafeId(): string {
  return useId().replace(/[^a-zA-Z0-9]/g, "");
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
