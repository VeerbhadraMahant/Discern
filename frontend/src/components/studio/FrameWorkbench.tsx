import { useState } from "react";
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
  detections: Detection[];
  violations: Violation[];
  cameraName?: string;
  conditionLabel?: string;
  highlightedDetectionId: string | null;
  onHoverDetection: (id: string | null) => void;
}

type ViewMode = "annotated" | "slider" | "raw" | "restored";

export default function FrameWorkbench({
  annotatedImage,
  rawImage,
  restoredImage,
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

  const fallbackRaw = rawImage || annotatedImage;
  const fallbackRestored = restoredImage || annotatedImage;

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
            />
          </div>
        ) : (
          <div className="relative h-full w-full">
            <img
              src={
                viewMode === "raw"
                  ? fallbackRaw
                  : viewMode === "restored"
                  ? fallbackRestored
                  : annotatedImage
              }
              alt="CCTV Frame View"
              className="h-full w-full object-cover"
            />

            {/* Interactive Detection Overlays */}
            {viewMode === "annotated" && detections.length > 0 && (
              <div className="absolute inset-0 pointer-events-none">
                {detections.map((det) => {
                  const isHighlighted = highlightedDetectionId === det.id;

                  return (
                    <div
                      key={det.id}
                      style={{
                        position: "absolute",
                        left: `${(det.box.x / 1280) * 100}%`,
                        top: `${(det.box.y / 720) * 100}%`,
                        width: `${(det.box.width / 1280) * 100}%`,
                        height: `${(det.box.height / 720) * 100}%`,
                      }}
                      className={`pointer-events-auto cursor-pointer transition-all ${
                        isHighlighted
                          ? "ring-4 ring-hermes-blue bg-hermes-blue/30 animate-pulse"
                          : ""
                      }`}
                      onMouseEnter={() => onHoverDetection(det.id)}
                      onMouseLeave={() => onHoverDetection(null)}
                    />
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
