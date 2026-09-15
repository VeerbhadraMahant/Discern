import { Eye, BrainCircuit, Wand2, UserCheck, ShieldAlert, Cpu } from "lucide-react";

const PIPELINE_STAGES = [
  {
    num: "#1 PERCEIVE",
    title: "Scene Understanding",
    subtitle: "Gemini Vision Classifier",
    description: "Inspects raw frame to classify lighting (dawn/dusk/glare), atmospheric weather (fog/dust/rain), visibility depth, and presence of human workers or heavy machinery.",
    icon: Eye,
    latency: "~380ms",
  },
  {
    num: "#2 DISPATCH",
    title: "Dynamic Tool Selection",
    subtitle: "Agentic Decision Tree",
    description: "Evaluates whether restoration is necessary. If clear, bypasses restoration to save compute latency. If degraded, selects specific OpenCV kernels (CLAHE, Dehaze, Denoise).",
    icon: BrainCircuit,
    latency: "<10ms",
  },
  {
    num: "#3 RESTORE",
    title: "Classical CV Kernels",
    subtitle: "OpenCV Light Recovery",
    description: "Applies dark-channel prior dehazing, CLAHE contrast enhancement, or bilateral spatial denoising. Zero extra heavy model downloads; runs locally at lightning speed.",
    icon: Wand2,
    latency: "~18-35ms",
  },
  {
    num: "#4 LOCALIZE",
    title: "Person Detection",
    subtitle: "Pretrained YOLOv8n",
    description: "Executes ultra-fast bounding box localization on the restored frame to extract pixel coordinates, confidence scores, and crop regions of all ground personnel.",
    icon: UserCheck,
    latency: "~25-35ms",
  },
  {
    num: "#5 REASON",
    title: "Violation Reasoning",
    subtitle: "Gemini Safety Reasoner",
    description: "Synthesizes detected worker crops with spatial proximity to machinery danger zones and OSHA/MSHA PPE mandates (helmets, high-vis vests, exclusion zones).",
    icon: ShieldAlert,
    latency: "~350ms",
  },
];

export default function PipelineShowcase() {
  return (
    <section className="py-20 md:py-28 bg-white text-[#050518] border-b border-[#0000f2]/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.15em] bg-[#0000f2] text-white px-3.5 py-1 font-bold">
            <Cpu className="h-3.5 w-3.5" />
            <span>The 5-Stage Agent Loop</span>
          </div>
          <h2 className="hermes-title text-3xl sm:text-5xl font-light tracking-tight uppercase text-[#050518]">
            <span>How Discern Thinks</span> <br />
            <span className="font-extrabold text-[#0000f2]">Through Every Camera Frame</span>
          </h2>
          <p className="font-body text-sm sm:text-base text-[#3a3a52] leading-relaxed">
            Instead of forcing one rigid model through unpredictable site chaos, Discern orchestrates a dynamic multi-stage agentic loop.
          </p>
        </div>

        {/* Feature Grid with Explicit Dark Text */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {PIPELINE_STAGES.map((stage) => {
            const Icon = stage.icon;
            return (
              <article
                key={stage.num}
                className="group border-2 border-[#050518]/20 bg-[#f4f4f7] p-6 shadow-md transition-all hover:border-[#0000f2] hover:shadow-xl flex flex-col justify-between text-[#050518]"
              >
                <div className="space-y-4">
                  {/* Top Number Tag */}
                  <div className="flex items-center justify-between pb-3 border-b border-[#050518]/15">
                    <p className="font-mono text-xs font-black text-[#0000f2] tracking-wider">
                      {stage.num}
                    </p>
                    <Icon className="h-5 w-5 text-[#0000f2] group-hover:scale-110 transition-transform" />
                  </div>

                  <h3 className="hermes-title text-xl font-bold tracking-normal uppercase leading-tight text-[#050518]">
                    {stage.title}
                  </h3>

                  <p className="font-mono text-[11px] text-[#5c5c78] font-bold uppercase">
                    {stage.subtitle}
                  </p>

                  <p className="font-body text-xs text-[#3a3a52] leading-relaxed font-medium">
                    {stage.description}
                  </p>
                </div>

                {/* Bottom Latency */}
                <div className="mt-6 pt-3 border-t border-[#050518]/15 flex items-center justify-between font-mono text-xs">
                  <span className="text-[#5c5c78] uppercase font-bold text-[11px]">Latency</span>
                  <span className="font-black text-[#0000f2]">{stage.latency}</span>
                </div>
              </article>
            );
          })}
        </div>

        {/* Edge Guarantee Callout */}
        <div className="mt-12 border-2 border-[#0000f2] bg-[#0000f2]/5 p-6 md:p-8 max-w-4xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 text-[#050518]">
          <div className="space-y-2">
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-[#0000f2] block">
              // ZERO_COMPUTE_WASTE_ARCHITECTURE
            </span>
            <h4 className="hermes-title text-xl font-bold text-[#050518] uppercase">
              Classical CV + Selective GPU Inference
            </h4>
            <p className="font-body text-xs sm:text-sm text-[#3a3a52] max-w-xl leading-relaxed font-medium">
              By pairing zero-cost classical CV algorithms (OpenCV CLAHE, Dehaze) with agentic gating, Discern eliminates the need for expensive multi-gigabyte diffusion models on site edge gateways.
            </p>
          </div>
          <div className="shrink-0 bg-[#0000f2] text-white p-4 font-mono text-center min-w-[150px] shadow-md">
            <span className="text-[11px] uppercase opacity-80 block font-bold">Restoration Overhead</span>
            <span className="text-2xl font-black block mt-0.5">&lt; 35 ms</span>
          </div>
        </div>
      </div>
    </section>
  );
}
