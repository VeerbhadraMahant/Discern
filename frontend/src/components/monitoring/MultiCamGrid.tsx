import { useState } from "react";
import { SAMPLE_CAMERAS, SAMPLE_SCENARIOS } from "../../lib/sampleScenarios";
import { Video, ShieldAlert, CheckCircle2, ArrowUpRight } from "lucide-react";

interface Props {
  onSelectCameraPreset: (presetId: string) => void;
}

export default function MultiCamGrid({ onSelectCameraPreset }: Props) {
  const [filter, setFilter] = useState<"all" | "alert" | "active">("all");

  const filteredCameras = SAMPLE_CAMERAS.filter((cam) => {
    if (filter === "alert") return cam.status === "alert";
    if (filter === "active") return cam.status === "active";
    return true;
  });

  return (
    <div className="space-y-6 font-mono text-hermes-ink">
      {/* Top Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-3 border-b border-hermes-ink/15">
        <div>
          <div className="flex items-center gap-2 text-xs">
            <span className="flex h-2 w-2 bg-hermes-safe animate-pulse" />
            <span className="uppercase tracking-widest text-hermes-muted font-bold">
              // SITE_CONTROL_MATRIX
            </span>
          </div>
          <h1 className="mt-1 hermes-title text-2xl sm:text-3xl font-bold uppercase text-hermes-ink">
            Perimeter Video Grid
          </h1>
        </div>

        {/* Filter Tabs */}
        <div className="flex items-center gap-1 border border-hermes-ink/20 p-1 bg-white shadow-lift text-xs uppercase">
          <button
            onClick={() => setFilter("all")}
            className={`px-3 py-1 font-bold transition-all ${
              filter === "all" ? "bg-hermes-blue text-white" : "text-hermes-charcoal hover:text-hermes-blue"
            }`}
          >
            :ALL_FEEDS_4:
          </button>
          <button
            onClick={() => setFilter("alert")}
            className={`px-3 py-1 font-bold transition-all ${
              filter === "alert" ? "bg-hermes-critical text-white" : "text-hermes-critical hover:bg-hermes-critical/10"
            }`}
          >
            :ALERTS_3:
          </button>
          <button
            onClick={() => setFilter("active")}
            className={`px-3 py-1 font-bold transition-all ${
              filter === "active" ? "bg-hermes-safe text-white" : "text-hermes-safe hover:bg-hermes-safe/10"
            }`}
          >
            :COMPLIANT_1:
          </button>
        </div>
      </div>

      {/* 2x2 Camera Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {filteredCameras.map((cam) => {
          const matchingScenario = SAMPLE_SCENARIOS.find((s) => s.id === cam.presetId);
          const previewImg = matchingScenario?.annotatedImageUrl;

          return (
            <div
              key={cam.id}
              className="group border border-hermes-ink/20 bg-white shadow-lift hover:shadow-hermes transition-all flex flex-col justify-between"
            >
              {/* Header */}
              <div className="flex items-center justify-between p-3.5 border-b border-hermes-ink/10 bg-hermes-paper">
                <div className="flex items-center gap-2.5">
                  <div className="flex h-7 w-7 items-center justify-center bg-hermes-blue text-white">
                    <Video className="h-3.5 w-3.5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-xs uppercase text-hermes-ink">
                      {cam.name}
                    </h3>
                    <p className="text-[10px] text-hermes-muted">
                      {cam.location} · {cam.zone}
                    </p>
                  </div>
                </div>

                <span
                  className={`px-2 py-0.5 text-[10px] font-bold uppercase ${
                    cam.status === "alert"
                      ? "bg-hermes-critical text-white"
                      : "bg-hermes-safe text-white"
                  }`}
                >
                  {cam.status === "alert" ? `${cam.violationsCount} HAZARDS` : "NOMINAL"}
                </span>
              </div>

              {/* Video Thumbnail */}
              <div className="relative aspect-[16/9] w-full bg-black overflow-hidden">
                {previewImg && (
                  <img
                    src={previewImg}
                    alt={cam.name}
                    className="h-full w-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                )}

                <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-black/85 px-2 py-0.5 text-[10px] text-white border border-white/20">
                  <span className="h-1.5 w-1.5 bg-hermes-critical animate-ping" />
                  <span>REC ● {cam.resolution} @ {cam.fps}fps</span>
                </div>

                <div className="absolute top-2 right-2 bg-black/85 px-2 py-0.5 text-[10px] text-hermes-warning border border-white/20 uppercase">
                  {cam.weatherCondition}
                </div>

                {/* Hover Launch Button */}
                <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <button
                    onClick={() => onSelectCameraPreset(cam.presetId)}
                    className="hermes-btn-primary text-xs flex items-center gap-2"
                  >
                    <span>Launch Agent Debugger</span>
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              {/* Footer Meta */}
              <div className="p-3.5 flex items-center justify-between text-xs bg-white border-t border-hermes-ink/10">
                <div className="text-[11px] text-hermes-muted">
                  LUX: <strong className="text-hermes-ink font-bold">{cam.lux} lx</strong>
                </div>

                <button
                  onClick={() => onSelectCameraPreset(cam.presetId)}
                  className="font-bold text-hermes-blue hover:underline text-xs uppercase"
                >
                  Inspect Stream &rarr;
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
