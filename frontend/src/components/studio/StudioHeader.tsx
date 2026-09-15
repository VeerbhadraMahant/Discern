import { useState, useEffect } from "react";
import { SAMPLE_SCENARIOS } from "../../lib/sampleScenarios";
import { 
  Camera, 
  Upload, 
  Download, 
  Clock, 
  Sparkles, 
  ChevronDown, 
  Radio, 
  RotateCcw,
  SlidersHorizontal,
  Terminal
} from "lucide-react";

interface Props {
  selectedPresetId: string | null;
  onSelectPreset: (presetId: string) => void;
  onOpenUpload: () => void;
  onOpenExport: () => void;
  canExport?: boolean;
  onReset: () => void;
  customFileName?: string | null;
}

export default function StudioHeader({
  selectedPresetId,
  onSelectPreset,
  onOpenUpload,
  onOpenExport,
  canExport = true,
  onReset,
  customFileName,
}: Props) {
  const [timecode, setTimecode] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimecode(now.toISOString().replace("T", " ").substring(0, 19) + " UTC");
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-4 font-mono">
      {/* Top Banner / Breadcrumb & Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-3 border-b border-hermes-ink/15">
        <div>
          <div className="flex items-center gap-2 text-xs">
            <span className="flex h-2 w-2 bg-hermes-safe animate-pulse" />
            <span className="uppercase tracking-widest text-hermes-muted font-bold">
              // LIVE_AGENT_STUDIO
            </span>
            <span className="text-hermes-ink/30">/</span>
            <span className="font-bold text-hermes-ink">
              {customFileName ? `FILE:${customFileName}` : "RTSP_ACTIVE_STREAM"}
            </span>
          </div>
          <h1 className="mt-1 hermes-title text-2xl sm:text-3xl font-bold uppercase text-hermes-ink">
            Vision Agent Workbench
          </h1>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={onOpenUpload}
            className="hermes-btn-primary bg-hermes-blue text-white hover:bg-hermes-dark text-xs flex items-center gap-2"
          >
            <Upload className="h-3.5 w-3.5" />
            <span>Upload Camera Frame</span>
          </button>

          <button
            onClick={onOpenExport}
            disabled={!canExport}
            title={canExport ? undefined : "Run agent analysis on this frame first"}
            className="hermes-btn-ghost text-hermes-ink border-hermes-ink/30 hover:bg-hermes-ink/10 text-xs flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export Audit (.JSON)</span>
          </button>

          {customFileName && (
            <button
              onClick={onReset}
              className="p-2 border border-hermes-ink/30 text-hermes-ink hover:bg-hermes-ink/10"
              title="Reset to presets"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Preset Scenario Selector Strip Styled in Hermes Terminal Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border border-hermes-ink/20 bg-white p-2.5 shadow-lift">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase text-hermes-muted shrink-0">
            :PRESETS:
          </span>
          <div className="flex items-center gap-1.5 overflow-x-auto">
            {SAMPLE_SCENARIOS.map((preset) => {
              const isSelected = selectedPresetId === preset.id;
              return (
                <button
                  key={preset.id}
                  onClick={() => onSelectPreset(preset.id)}
                  className={`flex shrink-0 items-center gap-1.5 px-3 py-1.5 text-xs uppercase transition-all border ${
                    isSelected
                      ? "bg-hermes-blue text-white font-bold border-hermes-blue"
                      : "bg-hermes-paper text-hermes-ink border-hermes-ink/10 hover:border-hermes-blue"
                  }`}
                >
                  <span>:{preset.tag.toUpperCase()}:</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Live HUD Timestamp */}
        <div className="hidden lg:flex items-center gap-1.5 text-xs text-hermes-muted shrink-0">
          <Clock className="h-3.5 w-3.5 text-hermes-blue" />
          <span>{timecode}</span>
        </div>
      </div>
    </div>
  );
}
