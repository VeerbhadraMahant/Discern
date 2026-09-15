import { HardHat, Mountain, Anchor, Factory, ArrowRight } from "lucide-react";

const VERTICALS = [
  {
    num: "SITE_01",
    title: "Construction Sites",
    icon: HardHat,
    tag: "OSHA 1926",
    hazard: "Dawn low-light, scaffolding shadows & heavy excavator swing radius blind spots.",
    solution: "CLAHE contrast stretching isolates worker hard hats in deep shadows; dynamic danger-zone polygon calculation alerts when workers cross excavator rotation perimeters.",
    stats: "94% reduction in unspotted swing-radius intrusions",
  },
  {
    num: "SITE_02",
    title: "Open-Pit Mining",
    icon: Mountain,
    tag: "MSHA 30 CFR",
    hazard: "Dense mineral dust storms & 250-ton haul truck blind spots on steep haul ramps.",
    solution: "Bilateral denoising + fast dehazing cuts through particulate plumes, verifying pedestrian surveyor separation from heavy haul truck transit corridors.",
    stats: "100% haul ramp perimeter compliance tracking",
  },
  {
    num: "SITE_03",
    title: "Maritime & Port Logistics",
    icon: Anchor,
    tag: "OSHA 1917 / IMO",
    hazard: "Coastal sea fog & automated straddle carrier runways during crane hoist operations.",
    solution: "Dark-channel prior dehazing restores visibility past 100 meters, ensuring zero unauthorized personnel enter automated container hoist runways under fog protocols.",
    stats: "Sub-second emergency lockouts on quay breaches",
  },
  {
    num: "SITE_04",
    title: "Industrial & Steel Plants",
    icon: Factory,
    tag: "ISO 45001",
    hazard: "Specular solar glare, molten metal reflections & high thermal contrast in coil storage.",
    solution: "Local adaptive tone curve compression preserves shadow details in coil storage bays without sensor highlight clipping, enforcing hi-vis reflective apparel rules.",
    stats: "3.2x faster incident logging vs manual CCTV review",
  },
];

export default function IndustryVerticals() {
  return (
    <section className="py-20 md:py-28 bg-[#f4f4f7] text-[#050518] border-b border-[#0000f2]/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.15em] bg-[#0000f2]/10 text-[#0000f2] px-3.5 py-1 border border-[#0000f2]/30 font-bold">
            <Factory className="h-3.5 w-3.5" />
            <span>Industrial Deployments</span>
          </div>
          <h2 className="hermes-title text-3xl sm:text-5xl font-light tracking-tight uppercase text-[#050518]">
            <span>Engineered For</span> <br />
            <span className="font-extrabold text-[#0000f2]">High-Consequence Sites</span>
          </h2>
          <p className="font-body text-sm sm:text-base text-[#3a3a52] leading-relaxed">
            From deep open-pit copper mines to coastal maritime terminals, Discern ensures continuous, non-stop safety compliance across all shifts.
          </p>
        </div>

        {/* 4 Cards Grid Styled in Hermes Platform Style */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-6">
          {VERTICALS.map((v) => {
            const Icon = v.icon;
            return (
              <article
                key={v.title}
                className="group border-2 border-[#050518]/20 bg-white p-7 sm:p-8 shadow-md transition-all hover:border-[#0000f2] hover:shadow-xl text-[#050518]"
              >
                <div className="flex items-start justify-between gap-4 font-mono">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center bg-[#0000f2] text-white">
                      <Icon className="h-5 w-5" />
                    </div>
                    <span className="text-xs font-black text-[#5c5c78] uppercase tracking-widest">
                      {v.num}
                    </span>
                  </div>
                  <span className="border border-[#0000f2]/40 bg-[#0000f2]/10 text-[#0000f2] px-3 py-1 text-xs font-black">
                    {v.tag}
                  </span>
                </div>

                <h3 className="hermes-title text-2xl font-bold uppercase text-[#050518] mt-6">
                  {v.title}
                </h3>

                <div className="mt-4 space-y-3 font-mono text-xs">
                  <div className="bg-[#ff2222]/5 border-2 border-[#ff2222]/30 p-3">
                    <span className="text-[11px] font-black uppercase tracking-wider text-[#ff2222] block mb-0.5">
                      // CRITICAL_HAZARD
                    </span>
                    <p className="text-[#050518] font-body font-semibold leading-snug">{v.hazard}</p>
                  </div>

                  <div className="bg-[#0000f2]/5 border-2 border-[#0000f2]/30 p-3">
                    <span className="text-[11px] font-black uppercase tracking-wider text-[#0000f2] block mb-0.5">
                      // AGENTIC_RESPONSE
                    </span>
                    <p className="text-[#050518] font-body font-semibold leading-snug">{v.solution}</p>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-[#050518]/15 flex items-center justify-between font-mono text-xs font-black text-[#008844]">
                  <span>{v.stats}</span>
                  <ArrowRight className="h-4 w-4 text-[#0000f2] group-hover:translate-x-1 transition-transform" />
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}
