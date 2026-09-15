import { useState } from "react";
import { SAMPLE_SCENARIOS } from "../../lib/sampleScenarios";
import SliderComparison from "../ui/SliderComparison";
import { AlertTriangle, CheckCircle2, ChevronRight, Cpu, Layers, Terminal } from "lucide-react";

interface Props {
  onSelectScenario: (scenarioId: string) => void;
}

export default function InteractiveDemo({ onSelectScenario }: Props) {
  const [activeTab, setActiveTab] = useState(0);
  const scenario = SAMPLE_SCENARIOS[activeTab];

  return (
    <section id="interactive-demo" className="py-20 md:py-28 bg-[#f4f4f7] text-[#050518] border-b border-[#0000f2]/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.15em] bg-[#0000f2]/10 text-[#0000f2] px-3.5 py-1 border border-[#0000f2]/30 font-bold">
            <Terminal className="h-3.5 w-3.5" />
            <span>Interactive Benchmarks</span>
          </div>
          <h2 className="hermes-title text-3xl sm:text-5xl font-light tracking-tight uppercase text-[#050518]">
            <span>Fixed Models Go Blind.</span> <br />
            <span className="font-extrabold text-[#0000f2]">Discern Clears The Lens First.</span>
          </h2>
          <p className="font-body text-sm sm:text-base text-[#3a3a52] leading-relaxed">
            Test adverse site scenarios below to observe how classical CV restoration dynamically adapts prior to YOLO inference and multi-modal reasoning.
          </p>
        </div>

        {/* Condition Tabs */}
        <div className="mt-10 flex justify-center">
          <div className="inline-flex flex-wrap items-center justify-center gap-2 border-2 border-[#050518]/20 p-1.5 bg-white shadow-md font-mono text-xs uppercase">
            {SAMPLE_SCENARIOS.map((s, idx) => (
              <button
                key={s.id}
                onClick={() => setActiveTab(idx)}
                className={`px-4 py-2 transition-all cursor-pointer font-bold ${
                  activeTab === idx
                    ? "bg-[#0000f2] text-white"
                    : "text-[#3a3a52] hover:text-[#0000f2] hover:bg-[#0000f2]/10"
                }`}
              >
                <span>:{s.title.toUpperCase()}:</span>
              </button>
            ))}
          </div>
        </div>

        {/* Main Interactive Showcase Grid */}
        <div className="mt-10 grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left: Frame Comparison Box */}
          <div className="lg:col-span-8 space-y-3">
            <div className="border-2 border-[#050518] shadow-lg overflow-hidden bg-black">
              <SliderComparison
                beforeImage={scenario.photoUrl}
                afterImage={scenario.photoUrl}
                beforeFilter={scenario.rawFilter}
                afterFilter={scenario.restoredFilter}
                beforeLabel={`RAW DEGRADED (${scenario.condition.toUpperCase()})`}
                afterLabel="RESTORED FRAME + YOLOv8 OVERLAYS"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-[#3a3a52] px-1">
              <span className="flex items-center gap-1.5 font-bold">
                <span className="h-2 w-2 bg-[#0000f2]" />
                Stream: 1920x1080 @ 30fps H.265
              </span>
              <span>
                Camera: <strong className="text-[#050518] font-bold">{scenario.camera}</strong>
              </span>
            </div>
          </div>

          {/* Right: Telemetry & Violations Card */}
          <div className="lg:col-span-4 space-y-4 font-mono">
            {/* Scene Understanding */}
            <div className="border-2 border-[#050518]/20 bg-white p-5 shadow-md space-y-3 text-[#050518]">
              <div className="flex items-center justify-between pb-2 border-b border-[#050518]/15">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#5c5c78]">
                  # SCENE_UNDERSTANDING
                </span>
                <span className="text-[11px] bg-[#0000f2] text-white px-2 py-0.5 uppercase font-bold">
                  Gemini Vision
                </span>
              </div>
              <p className="text-xs text-[#050518] leading-snug font-body font-semibold">
                {scenario.analysis.scene.summary}
              </p>

              <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                <div className="bg-[#f4f4f7] p-2 border border-[#050518]/15">
                  <span className="text-[11px] text-[#5c5c78] font-bold block">LIGHTING</span>
                  <span className="font-bold text-[#050518] capitalize">{scenario.analysis.scene.lighting}</span>
                </div>
                <div className="bg-[#f4f4f7] p-2 border border-[#050518]/15">
                  <span className="text-[11px] text-[#5c5c78] font-bold block">WEATHER</span>
                  <span className="font-bold text-[#050518] capitalize">{scenario.analysis.scene.weather}</span>
                </div>
              </div>
            </div>

            {/* Dispatched Strategy */}
            <div className="border-2 border-[#050518]/20 bg-white p-5 shadow-md space-y-3 text-[#050518]">
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#5c5c78] block">
                # DYNAMIC_DISPATCH
              </span>
              <div className="flex flex-wrap gap-1.5">
                {scenario.analysis.plan.restoration_tools.length > 0 ? (
                  scenario.analysis.plan.restoration_tools.map((t) => (
                    <span
                      key={t}
                      className="inline-flex items-center gap-1 bg-[#0000f2] text-white px-2.5 py-1 text-[11px] font-bold"
                    >
                      <Cpu className="h-3 w-3" />
                      {t.toUpperCase()}
                    </span>
                  ))
                ) : (
                  <span className="bg-[#00cc66]/20 text-[#008844] font-bold px-2.5 py-1 text-[11px] border border-[#00cc66]/40">
                    RESTORATION BYPASSED (PRISTINE)
                  </span>
                )}
              </div>
              <p className="text-xs text-[#3a3a52] font-body leading-relaxed font-medium">
                {scenario.analysis.plan.reasoning}
              </p>
            </div>

            {/* Violations Card */}
            <div className="border-2 border-[#050518]/20 bg-white p-5 shadow-md space-y-3 text-[#050518]">
              <div className="flex items-center justify-between pb-2 border-b border-[#050518]/15">
                <span className="text-[11px] font-bold uppercase tracking-wider text-[#5c5c78]">
                  # VIOLATIONS_DETECTED ({scenario.analysis.violations.length})
                </span>
                <span className={`text-[11px] px-2 py-0.5 font-bold uppercase ${
                  scenario.analysis.violations.length > 0 ? "bg-[#ff2222] text-white" : "bg-[#00cc66] text-white"
                }`}>
                  {scenario.analysis.violations.length > 0 ? "CRITICAL" : "COMPLIANT"}
                </span>
              </div>

              <div className="space-y-2">
                {scenario.analysis.violations.map((v) => (
                  <div key={v.id} className="border-2 border-[#ff2222]/30 bg-[#ff2222]/5 p-3 text-xs">
                    <div className="flex justify-between font-bold text-[#ff2222] text-[11px] uppercase">
                      <span>{v.type.replace(/_/g, " ")}</span>
                      <span>{v.oshaCode}</span>
                    </div>
                    <p className="mt-1 text-[#050518] font-body leading-snug font-semibold">{v.description}</p>
                  </div>
                ))}
              </div>

              <button
                onClick={() => onSelectScenario(scenario.id)}
                className="w-full hermes-btn-dark mt-2 text-xs flex items-center justify-center gap-2 cursor-pointer"
              >
                <span>Debug in Live Studio</span>
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
