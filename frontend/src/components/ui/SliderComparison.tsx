import React, { useState, useRef, useCallback } from "react";
import { Sliders, Sparkles, Eye, ShieldAlert } from "lucide-react";

interface Props {
  beforeImage: string;
  afterImage: string;
  beforeLabel?: string;
  afterLabel?: string;
  showOverlayAnnotations?: boolean;
}

export default function SliderComparison({
  beforeImage,
  afterImage,
  beforeLabel = "Raw CCTV Feed (Degraded)",
  afterLabel = "Discern Agentic Restoration + Overlays",
}: Props) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percent);
  }, []);

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      if (isDragging) {
        handleMove(e.touches[0].clientX);
      }
    },
    [isDragging, handleMove]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (isDragging) {
        handleMove(e.clientX);
      }
    },
    [isDragging, handleMove]
  );

  return (
    <div
      ref={containerRef}
      className="relative aspect-[16/9] w-full select-none overflow-hidden rounded-[18px] bg-obsidian border border-slate/20 shadow-feature cursor-ew-resize group"
      onMouseDown={() => setIsDragging(true)}
      onMouseUp={() => setIsDragging(false)}
      onMouseLeave={() => setIsDragging(false)}
      onMouseMove={handleMouseMove}
      onTouchStart={() => setIsDragging(true)}
      onTouchEnd={() => setIsDragging(false)}
      onTouchMove={handleTouchMove}
    >
      {/* Background Image: Restored + Annotated (Right Side) */}
      <img
        src={afterImage}
        alt={afterLabel}
        className="absolute inset-0 h-full w-full object-cover"
        draggable={false}
      />

      {/* Foreground Image: Raw Degraded (Left Side - Clipped) */}
      <div
        className="absolute inset-0 h-full w-full overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
      >
        <img
          src={beforeImage}
          alt={beforeLabel}
          className="absolute inset-0 h-full w-full object-cover"
          draggable={false}
        />
      </div>

      {/* Vertical Divider Line */}
      <div
        className="absolute top-0 bottom-0 z-20 w-0.5 bg-paper shadow-[0_0_10px_rgba(0,0,0,0.5)] transition-all"
        style={{ left: `${sliderPosition}%` }}
      >
        {/* Handle Knob with Surgical Blue dot */}
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex h-9 w-9 items-center justify-center rounded-full bg-obsidian border-2 border-paper text-paper shadow-pill transition-transform group-hover:scale-110">
          <Sliders className="h-4 w-4 text-accent" />
        </div>
      </div>

      {/* Floating Badges */}
      <div className="pointer-events-none absolute top-4 left-4 z-30 flex items-center gap-1.5 rounded-pill bg-obsidian/85 px-3 py-1.5 text-xs font-semibold text-paper backdrop-blur-md border border-paper/10 shadow-subtle">
        <Eye className="h-3.5 w-3.5 text-slate" />
        <span>{beforeLabel}</span>
      </div>

      <div className="pointer-events-none absolute top-4 right-4 z-30 flex items-center gap-1.5 rounded-pill bg-obsidian/85 px-3 py-1.5 text-xs font-semibold text-paper backdrop-blur-md border border-accent/30 shadow-subtle">
        <Sparkles className="h-3.5 w-3.5 text-accent" />
        <span>{afterLabel}</span>
      </div>

      {/* Bottom Hint */}
      <div className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 z-30 rounded-pill bg-obsidian/75 px-3 py-1 text-[11px] font-mono text-paper/80 backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity">
        Drag left / right to compare CV pipeline adaptation
      </div>
    </div>
  );
}
