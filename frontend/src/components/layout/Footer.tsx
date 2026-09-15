import { ShieldCheck, Cpu, Terminal, ArrowUpRight } from "lucide-react";

export default function Footer({ onLaunch }: { onLaunch: () => void }) {
  return (
    <footer className="bg-hermes-blue text-white border-t border-white/20 py-12 font-mono">
      <div className="mx-auto max-w-7xl px-4 sm:px-8">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8 pb-8 border-b border-white/15">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center bg-white text-hermes-blue font-bold text-xs rotate-45">
                <span className="-rotate-45">D</span>
              </div>
              <span className="font-heading text-lg font-bold uppercase tracking-wider">DISCERN AGENT</span>
              <span className="bg-white/10 px-2 py-0.5 text-[11px] uppercase border border-white/20">
                v2.4-edge
              </span>
            </div>
            <p className="max-w-md text-xs text-white/80 leading-relaxed font-body">
              Open-source adaptive vision agent for harsh industrial sites. Classical OpenCV restoration + YOLOv8n + Gemini Multimodal Vision Reasoning.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={onLaunch}
              className="hermes-btn-primary text-xs"
            >
              <span>Open Monitoring Console</span>
            </button>
          </div>
        </div>

        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-white/70">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-white">
              <Cpu className="h-3.5 w-3.5 text-hermes-accent" />
              OpenCV CLAHE / DCP / Bilateral Filter
            </span>
            <span className="hidden sm:inline opacity-40">|</span>
            <span className="hidden sm:flex items-center gap-1.5 text-white">
              <ShieldCheck className="h-3.5 w-3.5 text-hermes-safe" />
              OSHA 1926 & ISO 45001 Rule Engine
            </span>
          </div>
          <div>
            &copy; 2026 Discern Agent. Built under the MIT License.
          </div>
        </div>
      </div>
    </footer>
  );
}
