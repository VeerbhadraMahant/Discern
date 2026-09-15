import { useState } from "react";
import { ArrowUpRight, Cloud, Copy, Check, Terminal, Play, Radio, Cpu, ShieldAlert } from "lucide-react";
import SliderComparison from "../ui/SliderComparison";
import { SAMPLE_SCENARIOS } from "../../lib/sampleScenarios";

interface Props {
  onLaunch: () => void;
  onSelectPreset: (presetId: string) => void;
}

export default function LandingHero({ onLaunch, onSelectPreset }: Props) {
  const [activeScenarioIndex, setActiveScenarioIndex] = useState(0);
  const [terminalTab, setTerminalTab] = useState<"rtsp" | "docker" | "pip">("rtsp");
  const [copied, setCopied] = useState(false);

  const currentScenario = SAMPLE_SCENARIOS[activeScenarioIndex];

  const terminalCommands = {
    rtsp: "discern-agent --stream rtsp://cctv-cam04:8554/live --auto-restore",
    docker: "docker run -d --gpus all -p 8000:8000 ghcr.io/discern/agent:latest",
    pip: "pip install discern-vision && discern run --camera CAM-04",
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(terminalCommands[terminalTab]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className="relative overflow-hidden bg-[#0000f2] text-white pt-12 pb-20 md:pt-16 md:pb-24 border-b border-white/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-8">
        {/* Symmetrical Hero Copy */}
        <div className="max-w-4xl mx-auto text-center space-y-6">
          {/* Top Monospace Tag */}
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.15em] bg-white/10 text-white px-3.5 py-1 border border-white/30 backdrop-blur-sm">
            <span className="h-2 w-2 bg-[#00e5ff] animate-pulse" />
            <span>Open-Source Adaptive Site Vision</span>
          </div>

          {/* Condensed Tall All-Caps Display Typography (Hermes Style) */}
          <h1 className="hermes-title text-4xl sm:text-6xl lg:text-7xl font-light tracking-[-0.03em] leading-none uppercase text-white">
            <span className="block font-normal text-white">The Vision Agent</span>
            <span className="block font-black text-white mt-1">That Adapts With Site Conditions</span>
          </h1>

          <p className="font-body text-sm sm:text-base text-white/90 max-w-2xl mx-auto leading-relaxed">
            Fixed vision models go blind in dawn low-light, ocean sea fog, particulate mine dust, and industrial glare. 
            <strong className="text-white font-bold"> Discern</strong> inspects each camera frame, classifies atmospheric weather, and dynamically dispatches classical CV restoration before isolating safety violations.
          </p>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
            <button
              onClick={onLaunch}
              className="hermes-btn-primary"
            >
              <span>Launch Studio App</span>
            </button>

            <a
              href="#interactive-demo"
              className="hermes-btn-ghost flex items-center gap-2"
            >
              <Cloud className="h-3.5 w-3.5" />
              <span>Deploy to Site Gateway</span>
            </a>
          </div>

          {/* Terminal Command Box */}
          <div className="mt-8 max-w-xl mx-auto text-left font-mono">
            <div className="flex items-center justify-between pb-1.5 text-[11px] text-white/80 uppercase tracking-wider">
              <span>// STREAM_INGEST_COMMAND</span>
              <span>v2.4-edge</span>
            </div>

            <div className="border border-white/30 bg-[#000088]/80 backdrop-blur-md">
              {/* Terminal Tabs */}
              <div className="flex items-center border-b border-white/20 bg-black/40 text-[11px] uppercase">
                <button
                  onClick={() => setTerminalTab("rtsp")}
                  className={`px-3 py-1.5 transition-colors cursor-pointer ${
                    terminalTab === "rtsp" ? "bg-white text-[#0000f2] font-bold" : "text-white/75 hover:text-white"
                  }`}
                >
                  :RTSP_STREAM:
                </button>
                <button
                  onClick={() => setTerminalTab("docker")}
                  className={`px-3 py-1.5 transition-colors cursor-pointer ${
                    terminalTab === "docker" ? "bg-white text-[#0000f2] font-bold" : "text-white/75 hover:text-white"
                  }`}
                >
                  :DOCKER:
                </button>
                <button
                  onClick={() => setTerminalTab("pip")}
                  className={`px-3 py-1.5 transition-colors cursor-pointer ${
                    terminalTab === "pip" ? "bg-white text-[#0000f2] font-bold" : "text-white/75 hover:text-white"
                  }`}
                >
                  :PYTHON_CLI:
                </button>
              </div>

              {/* Terminal Command Row */}
              <div className="flex items-center justify-between p-3 text-xs bg-black/50">
                <code className="text-white font-mono truncate mr-2">
                  {terminalCommands[terminalTab]}
                </code>
                <button
                  onClick={handleCopy}
                  className="shrink-0 p-1.5 bg-white/15 hover:bg-white/25 transition-colors text-white cursor-pointer"
                  title="Copy command"
                >
                  {copied ? <Check className="h-3.5 w-3.5 text-[#00e676]" /> : <Copy className="h-3.5 w-3.5" />}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Live Interactive Hero Plate */}
        <div className="mt-14 max-w-5xl mx-auto">
          <div className="border-2 border-white/40 bg-[#050518] p-4 sm:p-6 shadow-2xl text-white">
            {/* Header of the Hero Preview */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-white/20">
              <div className="flex items-center gap-3">
                <div className="flex h-7 w-7 items-center justify-center bg-white text-[#0000f2] font-mono font-bold text-xs">
                  REC
                </div>
                <div>
                  <h3 className="text-sm font-mono font-bold text-white uppercase tracking-wider">
                    {currentScenario.site}
                  </h3>
                  <p className="text-xs text-white/70 font-mono">
                    {currentScenario.camera} · Condition: <span className="text-[#ff9900] font-bold">{currentScenario.condition}</span>
                  </p>
                </div>
              </div>

              {/* Scenario Quick Selector Tabs */}
              <div className="flex items-center gap-1 overflow-x-auto font-mono text-xs uppercase">
                {SAMPLE_SCENARIOS.map((sc, idx) => (
                  <button
                    key={sc.id}
                    onClick={() => setActiveScenarioIndex(idx)}
                    className={`shrink-0 px-3 py-1 transition-all border cursor-pointer ${
                      activeScenarioIndex === idx
                        ? "bg-white text-[#0000f2] font-bold border-white"
                        : "bg-white/10 text-white/80 border-white/20 hover:border-white/50"
                    }`}
                  >
                    :{sc.tag}:
                  </button>
                ))}
              </div>
            </div>

            {/* Interactive Before/After Comparison Box */}
            <div className="mt-4 border border-white/20 overflow-hidden bg-black">
              <SliderComparison
                beforeImage={currentScenario.photoUrl}
                afterImage={currentScenario.photoUrl}
                beforeFilter={currentScenario.rawFilter}
                afterFilter={currentScenario.restoredFilter}
                beforeLabel={`RAW FEED (${currentScenario.tag.toUpperCase()})`}
                afterLabel="DISCERN RESTORED + YOLO + REASONING"
              />
            </div>

            {/* Bottom Metrics Bar */}
            <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
              <div className="bg-white/10 p-3 border border-white/15">
                <span className="text-[11px] uppercase text-white/70 block">Dispatched Tool</span>
                <span className="font-bold text-white truncate block mt-0.5">
                  {currentScenario.analysis.plan.restoration_tools.join(" + ") || "Direct Bypass"}
                </span>
              </div>

              <div className="bg-white/10 p-3 border border-white/15">
                <span className="text-[11px] uppercase text-white/70 block">Safety Hazards</span>
                <span className="font-bold text-[#ff3333] truncate block mt-0.5">
                  {currentScenario.analysis.violations.length} Critical Infractions
                </span>
              </div>

              <div className="bg-white/10 p-3 border border-white/15 flex items-center justify-between">
                <div>
                  <span className="text-[11px] uppercase text-white/70 block">Agent Latency</span>
                  <span className="font-bold text-white">
                    {currentScenario.analysis.steps.reduce((a, b) => a + b.duration_ms, 0)}ms
                  </span>
                </div>
                <button
                  onClick={() => {
                    onSelectPreset(currentScenario.id);
                    onLaunch();
                  }}
                  className="hermes-btn-primary py-1 px-3 text-[11px]"
                >
                  Inspect &rarr;
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
