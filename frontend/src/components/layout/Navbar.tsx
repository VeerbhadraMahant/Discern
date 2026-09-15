import { useState, useEffect } from "react";
import type { AppView } from "../../lib/types";
import { checkHealth } from "../../lib/api";
import { Radio, Terminal, ExternalLink, Sliders, Video, FileText, Sparkles } from "lucide-react";

interface Props {
  currentView: AppView;
  onViewChange: (view: AppView) => void;
  onOpenUpload?: () => void;
}

export default function Navbar({ currentView, onViewChange, onOpenUpload }: Props) {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      const ok = await checkHealth();
      if (mounted) setIsOnline(ok);
    };
    poll();
    const interval = setInterval(poll, 12000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="relative z-50 bg-hermes-blue text-white border-b border-white/15">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-8 md:grid md:grid-cols-[1fr_auto_1fr] md:gap-x-6">
        {/* Left: Nav Links */}
        <div className="hidden md:flex items-center gap-x-6 font-mono text-xs uppercase tracking-[0.08em]">
          <button
            onClick={() => onViewChange("landing")}
            className={`transition-opacity hover:opacity-100 ${
              currentView === "landing" ? "opacity-100 font-bold underline underline-offset-4" : "opacity-75"
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => onViewChange("studio")}
            className={`transition-opacity hover:opacity-100 ${
              currentView === "studio" ? "opacity-100 font-bold underline underline-offset-4" : "opacity-75"
            }`}
          >
            Live Studio
          </button>
          <button
            onClick={() => onViewChange("multicam")}
            className={`transition-opacity hover:opacity-100 ${
              currentView === "multicam" ? "opacity-100 font-bold underline underline-offset-4" : "opacity-75"
            }`}
          >
            Multi-Cam
          </button>
          <button
            onClick={() => onViewChange("audit")}
            className={`transition-opacity hover:opacity-100 ${
              currentView === "audit" ? "opacity-100 font-bold underline underline-offset-4" : "opacity-75"
            }`}
          >
            Audit Log
          </button>
        </div>

        {/* Center: Iconic Hermes-Style Brand Stack */}
        <div className="flex flex-col items-center justify-center text-center">
          <button
            onClick={() => onViewChange("landing")}
            className="group flex flex-col items-center leading-none focus:outline-none"
          >
            <div className="flex items-center gap-2 mb-0.5">
              {/* Geometric Wing/Reticle Icon */}
              <div className="flex h-5 w-5 items-center justify-center bg-white text-hermes-blue font-mono text-xs font-bold rotate-45 group-hover:rotate-90 transition-transform">
                <span className="-rotate-45 font-bold">D</span>
              </div>
            </div>
            <span className="font-heading text-2xl sm:text-3xl font-bold tracking-tight uppercase leading-none">
              DISCERN
            </span>
            <span className="font-mono text-[11px] tracking-[0.2em] opacity-80 uppercase mt-0.5">
              AGENT
            </span>
          </button>
        </div>

        {/* Right: Status & Action Button */}
        <div className="flex items-center justify-end gap-4">
          <div className="hidden lg:flex items-center gap-2 font-mono text-[11px] opacity-85">
            <span
              className={`h-2 w-2 ${
                isOnline ? "bg-hermes-safe animate-pulse" : "bg-white/40"
              }`}
            />
            <span>{isOnline ? "EDGE_ONLINE" : "SIMULATOR"}</span>
          </div>

          {currentView === "landing" ? (
            <button
              onClick={() => onViewChange("studio")}
              className="hermes-btn-primary text-xs"
            >
              <span>Launch Studio</span>
            </button>
          ) : (
            <button
              onClick={onOpenUpload}
              className="hermes-btn-primary text-xs flex items-center gap-2"
            >
              <Radio className="h-3 w-3" />
              <span>Analyze Frame</span>
            </button>
          )}
        </div>
      </div>

      {/* Mobile Tab Strip */}
      <div className="flex md:hidden border-t border-white/10 px-4 py-2 gap-2 overflow-x-auto bg-black/20 font-mono text-xs uppercase">
        <button
          onClick={() => onViewChange("landing")}
          className={`px-2.5 py-1 ${currentView === "landing" ? "bg-white text-hermes-blue font-bold" : "text-white/80"}`}
        >
          Overview
        </button>
        <button
          onClick={() => onViewChange("studio")}
          className={`px-2.5 py-1 ${currentView === "studio" ? "bg-white text-hermes-blue font-bold" : "text-white/80"}`}
        >
          Studio
        </button>
        <button
          onClick={() => onViewChange("multicam")}
          className={`px-2.5 py-1 ${currentView === "multicam" ? "bg-white text-hermes-blue font-bold" : "text-white/80"}`}
        >
          Multi-Cam
        </button>
        <button
          onClick={() => onViewChange("audit")}
          className={`px-2.5 py-1 ${currentView === "audit" ? "bg-white text-hermes-blue font-bold" : "text-white/80"}`}
        >
          Audit
        </button>
      </div>
    </header>
  );
}
