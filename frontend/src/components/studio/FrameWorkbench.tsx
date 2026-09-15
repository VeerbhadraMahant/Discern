import { useState, useRef, useEffect, useCallback } from "react";
import type { Detection, Violation } from "../../lib/types";
import SliderComparison from "../ui/SliderComparison";
import { 
  Eye, 
  Sparkles, 
  Maximize2, 
  Minimize2, 
  Sliders, 
  ShieldAlert, 
  CheckCircle2, 
  Crosshair,
  Terminal
} from "lucide-react";

interface Props {
  annotatedImage: string;
  rawImage?: string;
  restoredImage?: string;
  /** CSS filter simulating the degraded raw sensor read (preset scenarios only). */
  rawFilter?: string;
  /** CSS filter simulating the agent's restoration pass (preset scenarios only). */
  restoredFilter?: string;
  detections: Detection[];
  violations: Violation[];
  cameraName?: string;
  conditionLabel?: string;
  highlightedDetectionId: string | null;
  onHoverDetection: (id: string | null) => void;
}

type ViewMode = "annotated" | "slider" | "raw" | "restored";

const SEVERITY_BOX_STYLES: Record<string, { border: string; chip: string }> = {
  critical: { border: "border-hermes-critical", chip: "bg-hermes-critical" },
  warning: { border: "border-hermes-warning", chip: "bg-hermes-warning" },
  info: { border: "border-hermes-safe", chip: "bg-hermes-safe" },
};

export default function FrameWorkbench({
  annotatedImage,
  rawImage,
  restoredImage,
  rawFilter = "none",
  restoredFilter = "none",
  detections,
  violations,
  cameraName = "CAM-04_NORTH_ZONE",
  conditionLabel = "Low-Light / Dawn (12.4 Lux)",
  highlightedDetectionId,
  onHoverDetection,
}: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>("annotated");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showHud, setShowHud] = useState(true);
  // Detection boxes come back in the real uploaded image's own pixel space, which
  // varies per photo. The <img> is displayed with object-cover (crops to fill the
  // 16:9 canvas), so overlay boxes must be mapped through the same crop math the
  // browser used, not just naively scaled against the container.
  const canvasRef = useRef<HTMLDivElement>(null);
  const [coverRect, setCoverRect] = useState<{ renderW: number; renderH: number; offsetX: number; offsetY: number } | null>(null);

  const recomputeCoverRect = useCallback((imgW: number, imgH: number) => {
    const el = canvasRef.current;
    if (!el || !imgW || !imgH) return;
    const containerW = el.clientWidth;
    const containerH = el.clientHeight;
    const containerRatio = containerW / containerH;
    const imgRatio = imgW / imgH;
    let renderW: number, renderH: number;
    if (imgRatio > containerRatio) {
      renderH = containerH;
      renderW = containerH * imgRatio;
    } else {
      renderW = containerW;
      renderH = containerW / imgRatio;
    }
    setCoverRect({
      renderW,
      renderH,
      offsetX: (containerW - renderW) / 2,
      offsetY: (containerH - renderH) / 2,
    });
  }, []);

  const naturalDims = useRef<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const onResize = () => {
      if (naturalDims.current) recomputeCoverRect(naturalDims.current.w, naturalDims.current.h);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [recomputeCoverRect]);

  const fallbackRaw = rawImage || annotatedImage;
  const fallbackRestored = restoredImage || annotatedImage;

  const activeFilter =
    viewMode === "raw" ? rawFilter : viewMode === "restored" ? restoredFilter : restoredFilter;

  return (
    <div
      className={`relative overflow-hidden border-2 border-hermes-ink bg-black text-white font-mono shadow-terminal transition-all ${
        isFullscreen ? "fixed inset-4 z-50" : "w-full"
      }`}
    >
      {/* Top Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-black/90 px-4 py-2.5 border-b border-white/20">
        {/* Left: View Mode Tabs */}
        <div className="flex items-center gap-1 border border-white/20 bg-white/5 p-0.5 text-xs uppercase">
          <button
            onClick={() => setViewMode("annotated")}
            className={`px-3 py-1 font-bold transition-all ${
              viewMode === "annotated"
                ? "bg-hermes-blue text-white"
                : "text-white/70 hover:text-white"
            }`}
          >
            :ANNOTATED:
          </button>

          {rawImage && (
            <button
              onClick={() => setViewMode("slider")}
              className={`px-3 py-1 font-bold transition-all ${
                viewMode === "slider"
                  ? "bg-hermes-blue text-white"
                  : "text-white/70 hover:text-white"
              }`}
            >
              :BEFORE_AFTER:
            </button>
          )}

          <button
            onClick={() => setViewMode("raw")}
            className={`px-3 py-1 font-bold transition-all ${
              viewMode === "raw"
                ? "bg-hermes-blue text-white"
                : "text-white/70 hover:text-white"
            }`}
          >
            :RAW_FEED:
          </button>

          {restoredImage && (
            <button
              onClick={() => setViewMode("restored")}
              className={`px-3 py-1 font-bold transition-all ${
                viewMode === "restored"
                  ? "bg-hermes-blue text-white"
                  : "text-white/70 hover:text-white"
              }`}
            >
              :RESTORED_CV:
            </button>
          )}
        </div>

        {/* Right: Quick Toggles */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowHud(!showHud)}
            className={`px-2.5 py-1 text-xs border uppercase ${
              showHud
                ? "border-hermes-accent bg-hermes-blue text-white font-bold"
                : "border-white/20 text-white/60"
            }`}
          >
            HUD: {showHud ? "ACTIVE" : "MUTED"}
          </button>

          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 border border-white/20 text-white/80 hover:text-white hover:border-white transition-colors"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Viewer"}
          >
            {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {/* Main Video/Frame Canvas Area */}
      <div className="relative aspect-[16/9] w-full bg-black overflow-hidden flex items-center justify-center">
        {viewMode === "slider" && rawImage ? (
          <div className="h-full w-full">
            <SliderComparison
              beforeImage={fallbackRaw}
              afterImage={annotatedImage}
              beforeLabel="DEGRADED CAMERA STREAM"
              afterLabel="DISCERN RESTORED + OVERLAYS"
              beforeFilter={rawFilter}
              afterFilter={restoredFilter}
            />
          </div>
        ) : (
          <div ref={canvasRef} className="relative h-full w-full">
            <img
              src={
                viewMode === "raw"
                  ? fallbackRaw
                  : viewMode === "restored"
                  ? fallbackRestored
                  : annotatedImage
              }
              alt="CCTV Frame View"
              className="h-full w-full object-cover transition-[filter] duration-300"
              style={{ filter: activeFilter }}
              onLoad={(e) => {
                const img = e.currentTarget;
                naturalDims.current = { w: img.naturalWidth, h: img.naturalHeight };
                recomputeCoverRect(img.naturalWidth, img.naturalHeight);
              }}
            />

            {/* Interactive Detection Overlays - mapped through the same object-cover
                crop math the browser applied, since detection boxes are in the
                original photo's own pixel space (which varies per image). */}
            {viewMode === "annotated" && detections.length > 0 && coverRect && naturalDims.current && (
              <div className="absolute inset-0 pointer-events-none overflow-hidden">
                {detections.map((det) => {
                  const isHighlighted = highlightedDetectionId === det.id;
                  const violation = violations.find((v) => v.related_detection_ids.includes(det.id));
                  const style = violation
                    ? SEVERITY_BOX_STYLES[violation.severity]
                    : SEVERITY_BOX_STYLES.info;
                  const { w: natW, h: natH } = naturalDims.current!;
                  const { renderW, renderH, offsetX, offsetY } = coverRect;
                  const left = offsetX + (det.box.x / natW) * renderW;
                  const top = offsetY + (det.box.y / natH) * renderH;
                  const width = (det.box.width / natW) * renderW;
                  const height = (det.box.height / natH) * renderH;

                  return (
                    <div
                      key={det.id}
                      style={{ position: "absolute", left, top, width, height }}
                      className={`pointer-events-auto cursor-pointer transition-all border-2 ${style.border} ${
                        isHighlighted ? "ring-4 ring-hermes-blue bg-hermes-blue/25" : ""
                      }`}
                      onMouseEnter={() => onHoverDetection(det.id)}
                      onMouseLeave={() => onHoverDetection(null)}
                    >
                      <span
                        className={`absolute -top-6 left-0 whitespace-nowrap px-1.5 py-0.5 text-[11px] font-bold uppercase text-white ${style.chip}`}
                      >
                        {det.label} [{Math.round(det.confidence * 100)}%]{violation ? ` · ${violation.severity}` : ""}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Live CCTV HUD Overlay */}
        {showHud && viewMode !== "slider" && (
          <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-4 z-10 text-xs">
            {/* Top HUD */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 bg-black/85 px-3 py-1 border border-white/20">
                <span className="flex h-2 w-2 bg-hermes-critical animate-ping" />
                <span className="font-bold text-[11px] uppercase tracking-wider text-white">
                  REC ● {cameraName}
                </span>
              </div>

              <div className="bg-black/85 px-3 py-1 border border-white/20 text-white/90 text-[11px] uppercase">
                <span>ATMOSPHERE: <strong className="text-hermes-warning font-bold">{conditionLabel}</strong></span>
              </div>
            </div>

            {/* Bottom HUD */}
            <div className="flex items-center justify-between">
              <div className="bg-black/85 px-3 py-1 border border-white/20 text-white/70 text-[11px]">
                <span>1080p @ 30fps · H.265 · SENSOR_OK</span>
              </div>

              <div className="flex items-center gap-2 bg-hermes-blue px-3 py-1 text-white border border-white/40 text-[11px] font-bold">
                <Crosshair className="h-3 w-3" />
                <span>DISCERN_AGENT: LIVE</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Frame Status Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-black px-4 py-2.5 text-xs text-white/85 border-t border-white/20">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-hermes-safe font-bold">
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>{detections.length} TARGET(S) TRACKED</span>
          </span>
          <span className="text-white/20">|</span>
          <span className="flex items-center gap-1.5 text-hermes-critical font-bold">
            <ShieldAlert className="h-3.5 w-3.5" />
            <span>{violations.length} VIOLATION(S) DETECTED</span>
          </span>
        </div>

        <div className="text-[11px] text-white/60">
          HOVER TARGET TO TRACE REASONING
        </div>
      </div>
    </div>
  );
}
