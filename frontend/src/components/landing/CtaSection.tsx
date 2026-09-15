import { ArrowUpRight, Camera, Cpu, Sparkles, ShieldCheck, Terminal } from "lucide-react";

export default function CtaSection({ onLaunch }: { onLaunch: () => void }) {
  return (
    <section className="py-20 md:py-28 bg-[#0000f2] text-white">
      <div className="mx-auto max-w-5xl px-4 sm:px-8 text-center">
        <div className="border-2 border-white/40 bg-[#000088]/80 p-8 sm:p-14 backdrop-blur-md space-y-6 shadow-2xl">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.15em] bg-white/10 px-3.5 py-1 border border-white/30 text-white font-bold">
            <Terminal className="h-3.5 w-3.5" />
            <span>Zero Configuration Required</span>
          </div>

          <h2 className="hermes-title text-3xl sm:text-5xl font-light tracking-tight uppercase leading-tight text-white">
            <span className="block font-normal">Ready to Deploy Discern</span>
            <span className="block font-black text-white mt-1">On Your Site Feeds?</span>
          </h2>

          <p className="font-body text-sm sm:text-base text-white/90 max-w-xl mx-auto leading-relaxed">
            Upload any CCTV camera frame or explore our curated industrial scenarios. Watch the agent inspect, enhance, and reason in real time.
          </p>

          <div className="pt-2 flex justify-center">
            <button
              onClick={onLaunch}
              className="hermes-btn-primary text-sm px-8 py-3.5"
            >
              <span>Launch Studio App</span>
            </button>
          </div>

          <div className="pt-6 border-t border-white/20 flex flex-wrap items-center justify-center gap-6 font-mono text-xs text-white/80 uppercase font-bold">
            <span className="flex items-center gap-1.5">
              <Camera className="h-3.5 w-3.5 text-white" /> ANY RTSP / CCTV FEED
            </span>
            <span className="flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-white" /> SUB-50MS LOCAL OPENCV
            </span>
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5 text-white" /> OSHA / MSHA AUDIT TRAIL
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
