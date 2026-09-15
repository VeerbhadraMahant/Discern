import { Check, X, Shield, Cpu, Gauge, Terminal } from "lucide-react";

export default function TechnicalSpecs() {
  return (
    <section className="py-20 md:py-28 bg-white text-[#050518] border-b border-[#0000f2]/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.15em] bg-[#0000f2]/10 text-[#0000f2] px-3.5 py-1 border border-[#0000f2]/30 font-bold">
            <Gauge className="h-3.5 w-3.5" />
            <span>Architecture Matrix</span>
          </div>
          <h2 className="hermes-title text-3xl sm:text-5xl font-light tracking-tight uppercase text-[#050518]">
            <span>Edge Efficiency</span> <br />
            <span className="font-extrabold text-[#0000f2]">Over Server Farm Bloat</span>
          </h2>
          <p className="font-body text-sm sm:text-base text-[#3a3a52] leading-relaxed">
            See how Discern’s hybrid classical CV + vision agent architecture compares to traditional approaches.
          </p>
        </div>

        {/* Comparison Table Card with High-Contrast Dark Text */}
        <div className="mt-16 overflow-hidden border-2 border-[#050518] shadow-lg max-w-5xl mx-auto bg-white text-[#050518]">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b-2 border-[#050518] bg-[#f4f4f7] text-[11px] font-black uppercase tracking-wider text-[#050518]">
                  <th className="py-4 px-6 text-[#050518]">Evaluation Vector</th>
                  <th className="py-4 px-6 text-center text-[#5c5c78]">Fixed Single Model</th>
                  <th className="py-4 px-6 text-center text-[#5c5c78]">Multi-Model Ensemble</th>
                  <th className="py-4 px-6 bg-[#0000f2] text-white text-center font-bold">
                    DISCERN AGENT
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#050518]/15 text-[#050518]">
                <tr>
                  <td className="py-4 px-6 font-bold text-[#050518]">
                    Weather Adaptation
                  </td>
                  <td className="py-4 px-6 text-center text-[#ff2222]">
                    <span className="inline-flex items-center gap-1 font-bold">
                      <X className="h-4 w-4" /> None (Blind in Fog)
                    </span>
                  </td>
                  <td className="py-4 px-6 text-center text-[#3a3a52] font-semibold">
                    Static manual tuning
                  </td>
                  <td className="py-4 px-6 text-center bg-[#0000f2]/10 font-black text-[#0000f2]">
                    <span className="inline-flex items-center gap-1 text-[#008844]">
                      <Check className="h-4 w-4 text-[#008844]" /> Per-frame Dynamic CV
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-4 px-6 font-bold text-[#050518]">
                    Edge Compute Footprint
                  </td>
                  <td className="py-4 px-6 text-center text-[#3a3a52] font-semibold">
                    Lightweight (15MB)
                  </td>
                  <td className="py-4 px-6 text-center text-[#ff2222] font-bold">
                    Heavy (&gt;4.5 GB VRAM)
                  </td>
                  <td className="py-4 px-6 text-center bg-[#0000f2]/10 font-bold text-[#050518]">
                    Minimal (~6MB YOLO + NumPy)
                  </td>
                </tr>

                <tr>
                  <td className="py-4 px-6 font-bold text-[#050518]">
                    Clear Weather Latency
                  </td>
                  <td className="py-4 px-6 text-center text-[#3a3a52] font-semibold">
                    0ms
                  </td>
                  <td className="py-4 px-6 text-center text-[#ff2222] font-bold">
                    Always runs (slows pipeline)
                  </td>
                  <td className="py-4 px-6 text-center bg-[#0000f2]/10 font-black text-[#0000f2]">
                    <span className="inline-flex items-center gap-1 text-[#008844]">
                      <Check className="h-4 w-4 text-[#008844]" /> 0ms (Agent Skips CV)
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-4 px-6 font-bold text-[#050518]">
                    PPE Context & Reasoning
                  </td>
                  <td className="py-4 px-6 text-center text-[#ff2222] font-bold">
                    COCO classes only
                  </td>
                  <td className="py-4 px-6 text-center text-[#3a3a52] font-semibold">
                    Rigid heuristics
                  </td>
                  <td className="py-4 px-6 text-center bg-[#0000f2]/10 font-black text-[#0000f2]">
                    <span className="inline-flex items-center gap-1 text-[#008844]">
                      <Check className="h-4 w-4 text-[#008844]" /> Gemini Vision Reasoner
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-4 px-6 font-bold text-[#050518]">
                    Restoration Method
                  </td>
                  <td className="py-4 px-6 text-center text-[#5c5c78] font-semibold">
                    None
                  </td>
                  <td className="py-4 px-6 text-center text-[#3a3a52] font-semibold">
                    Heavy Diffusion GANs
                  </td>
                  <td className="py-4 px-6 text-center bg-[#0000f2]/10 font-black text-[#0000f2]">
                    OpenCV CLAHE / DCP / Bilateral
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
