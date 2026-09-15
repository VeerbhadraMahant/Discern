import type { ToolPlan } from "../../lib/types";
import { Cpu, Wand2, ShieldCheck, CheckCircle2, Sliders } from "lucide-react";

interface Props {
  plan: ToolPlan;
}

export default function ToolsEngagedCard({ plan }: Props) {
  const restorationCount = plan.restoration_tools.length;
  const detectionCount = plan.detection_tools.length;

  return (
    <div className="border border-hermes-ink/15 bg-white p-5 shadow-lift space-y-4 font-mono">
      <div className="flex items-center justify-between pb-2 border-b border-hermes-ink/10">
        <div className="flex items-center gap-2">
          <Sliders className="h-3.5 w-3.5 text-hermes-blue" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-hermes-ink">
            # DISPATCHED_PIPELINE
          </h2>
        </div>
        <span className="bg-hermes-ink text-white px-2 py-0.5 text-[10px] uppercase font-bold">
          {restorationCount + detectionCount} Tools Active
        </span>
      </div>

      <div className="space-y-3 text-xs">
        {/* Restoration Section */}
        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-hermes-muted block mb-1">
            // OPENCV_RESTORATION
          </span>
          {restorationCount === 0 ? (
            <div className="bg-hermes-safe/10 border border-hermes-safe/30 p-2 text-hermes-safe font-bold text-[11px]">
              RESTORATION BYPASSED (CLEAR CONDITIONS)
            </div>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {plan.restoration_tools.map((t) => (
                <span
                  key={t}
                  className="bg-hermes-blue text-white px-2.5 py-1 text-[11px] font-bold uppercase"
                >
                  :{t.toUpperCase()}:
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Detection Section */}
        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-hermes-muted block mb-1">
            // LOCALIZATION_&_REASONING
          </span>
          <div className="flex flex-wrap gap-1.5">
            {plan.detection_tools.map((t) => (
              <span
                key={t}
                className="bg-hermes-paper border border-hermes-ink/20 text-hermes-ink px-2.5 py-1 text-[11px] font-bold uppercase"
              >
                :{t.toUpperCase()}:
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Rationale */}
      <div className="bg-hermes-paper p-3 border border-hermes-ink/10 text-xs">
        <span className="text-[10px] font-bold uppercase text-hermes-blue block mb-1">
          AGENT_RATIONALE:
        </span>
        <p className="text-hermes-charcoal font-body leading-relaxed">
          {plan.reasoning}
        </p>
      </div>
    </div>
  );
}
