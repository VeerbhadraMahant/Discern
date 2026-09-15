import { useState } from "react";
import { Calculator, ShieldCheck, Sparkles, TrendingDown, Terminal } from "lucide-react";

export default function RoiCalculator({ onLaunch }: { onLaunch: () => void }) {
  const [cameraCount, setCameraCount] = useState(24);
  const [shiftHours, setShiftHours] = useState(16);
  const [adverseWeatherDays, setAdverseWeatherDays] = useState(85);

  const blindSpotHoursYearly = Math.round((adverseWeatherDays * shiftHours * 0.65) * (cameraCount / 10));
  const estimatedPreventedHazards = Math.round(blindSpotHoursYearly * 0.12);
  const estimatedInsuranceSavings = (estimatedPreventedHazards * 4200).toLocaleString();
  const computeSavingsPercent = 42;

  return (
    <section className="py-20 md:py-28 bg-[#f4f4f7] text-[#050518] border-b border-[#0000f2]/20">
      <div className="mx-auto max-w-7xl px-4 sm:px-8">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-3">
          <div className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-[0.15em] bg-[#0000f2]/10 text-[#0000f2] px-3.5 py-1 border border-[#0000f2]/30 font-bold">
            <Calculator className="h-3.5 w-3.5" />
            <span>Site Impact Model</span>
          </div>
          <h2 className="hermes-title text-3xl sm:text-5xl font-light tracking-tight uppercase text-[#050518]">
            <span>Calculate Site</span> <br />
            <span className="font-extrabold text-[#0000f2]">Risk Reduction</span>
          </h2>
          <p className="font-body text-sm sm:text-base text-[#3a3a52] leading-relaxed">
            Estimate how many unspotted blind-spot incidents Discern’s dynamic restoration eliminates per operational year.
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 lg:grid-cols-12 gap-8 max-w-5xl mx-auto items-center font-mono">
          {/* Sliders Input Panel */}
          <div className="lg:col-span-6 border-2 border-[#050518]/20 bg-white p-7 shadow-md space-y-6 text-[#050518]">
            <div>
              <div className="flex justify-between items-center text-xs font-black text-[#050518] mb-2">
                <span>ACTIVE CCTV FEEDS:</span>
                <span className="text-sm font-black text-[#0000f2]">{cameraCount} CAMERAS</span>
              </div>
              <input
                type="range"
                min="4"
                max="120"
                step="4"
                value={cameraCount}
                onChange={(e) => setCameraCount(Number(e.target.value))}
                className="w-full h-2 bg-[#050518]/20 appearance-none cursor-pointer accent-[#0000f2]"
              />
              <div className="flex justify-between text-[10px] text-[#707090] font-bold mt-1">
                <span>4 cams</span>
                <span>60 cams</span>
                <span>120 cams</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center text-xs font-black text-[#050518] mb-2">
                <span>OPERATIONAL HOURS / DAY:</span>
                <span className="text-sm font-black text-[#0000f2]">{shiftHours} HOURS</span>
              </div>
              <input
                type="range"
                min="8"
                max="24"
                step="4"
                value={shiftHours}
                onChange={(e) => setShiftHours(Number(e.target.value))}
                className="w-full h-2 bg-[#050518]/20 appearance-none cursor-pointer accent-[#0000f2]"
              />
              <div className="flex justify-between text-[10px] text-[#707090] font-bold mt-1">
                <span>8h (Single Shift)</span>
                <span>16h (Dawn/Dusk)</span>
                <span>24h (Non-stop)</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center text-xs font-black text-[#050518] mb-2">
                <span>ADVERSE WEATHER DAYS / YR:</span>
                <span className="text-sm font-black text-[#0000f2]">{adverseWeatherDays} DAYS</span>
              </div>
              <input
                type="range"
                min="10"
                max="200"
                step="5"
                value={adverseWeatherDays}
                onChange={(e) => setAdverseWeatherDays(Number(e.target.value))}
                className="w-full h-2 bg-[#050518]/20 appearance-none cursor-pointer accent-[#0000f2]"
              />
              <div className="flex justify-between text-[10px] text-[#707090] font-bold mt-1">
                <span>10 days (Mild)</span>
                <span>85 days (Average)</span>
                <span>200 days (Harsh)</span>
              </div>
            </div>
          </div>

          {/* Results Summary Card */}
          <div className="lg:col-span-6 border-2 border-[#0000f2] bg-white p-7 shadow-lg space-y-6 text-[#050518]">
            <span className="text-[11px] font-black uppercase tracking-wider text-[#707090] block">
              // ANNUAL_PROJECTED_PROTECTION
            </span>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-[#f4f4f7] p-4 border border-[#050518]/15">
                <span className="text-[10px] text-[#707090] font-bold uppercase block">Recovered Blind Hours</span>
                <span className="text-2xl sm:text-3xl font-black text-[#050518] mt-1 block">
                  {blindSpotHoursYearly.toLocaleString()} <span className="text-xs font-normal">hrs</span>
                </span>
                <span className="text-[10px] text-[#008844] font-black mt-1 flex items-center gap-1">
                  <TrendingDown className="h-3.5 w-3.5" /> Recovered by CV
                </span>
              </div>

              <div className="bg-[#f4f4f7] p-4 border border-[#050518]/15">
                <span className="text-[10px] text-[#707090] font-bold uppercase block">Hazards Prevented</span>
                <span className="text-2xl sm:text-3xl font-black text-[#ff2222] mt-1 block">
                  ~{estimatedPreventedHazards} <span className="text-xs font-normal">events</span>
                </span>
                <span className="text-[10px] text-[#3a3a52] font-semibold mt-1 block">
                  PPE & Zone Breaches
                </span>
              </div>
            </div>

            <div className="bg-[#0000f2]/10 border-2 border-[#0000f2] p-4 flex items-center justify-between">
              <div>
                <span className="text-[10px] font-bold uppercase text-[#3a3a52] block">Incident Mitigation</span>
                <span className="text-xl sm:text-2xl font-black text-[#0000f2]">
                  ${estimatedInsuranceSavings} <span className="text-xs font-normal text-[#050518]">/ yr</span>
                </span>
              </div>
              <div className="text-right">
                <span className="text-[10px] font-bold uppercase text-[#707090] block">GPU Savings</span>
                <span className="text-sm font-black text-[#050518]">{computeSavingsPercent}% Bypassed</span>
              </div>
            </div>

            <button
              onClick={onLaunch}
              className="w-full hermes-btn-blue text-xs py-3"
            >
              <span>Test Live Feeds &rarr;</span>
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
