import { useState } from "react";
import type { PipelineStep } from "../../lib/types";
import { CheckCircle2, CircleDashed, Clock, Activity, Cpu } from "lucide-react";

interface Props {
  steps: PipelineStep[];
}

export default function AgentPipelineDAG({ steps }: Props) {
  const totalDuration = steps.reduce((acc, s) => acc + s.duration_ms, 0);

  return (
    <div className="border border-hermes-ink/15 bg-white p-5 shadow-lift space-y-4 font-mono">
      <div className="flex items-center justify-between pb-2 border-b border-hermes-ink/10">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-hermes-blue" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-hermes-ink">
            # EXECUTION_TRACE
          </h2>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-bold text-hermes-blue">
          <Clock className="h-3.5 w-3.5" />
          <span>TOTAL: {totalDuration} MS</span>
        </div>
      </div>

      {/* Step Timeline */}
      <ol className="relative space-y-0">
        {steps.map((step, idx) => {
          const isCompleted = step.status === "completed";

          return (
            <li
              key={`${step.name}-${idx}`}
              className="relative flex gap-3 pb-3.5 last:pb-0 group"
            >
              {idx < steps.length - 1 && (
                <span
                  className="absolute left-[11px] top-6 h-[calc(100%-0.75rem)] w-px bg-hermes-ink/20"
                  aria-hidden="true"
                />
              )}

              {/* Status Indicator */}
              <div
                className={`z-10 mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center font-bold text-xs ${
                  isCompleted
                    ? "bg-hermes-blue text-white"
                    : "bg-hermes-paper text-hermes-muted border border-hermes-ink/20"
                }`}
              >
                {idx + 1}
              </div>

              {/* Step Content */}
              <div
                className={`min-w-0 flex-1 p-3 border text-xs ${
                  isCompleted
                    ? "bg-hermes-paper border-hermes-ink/15"
                    : "bg-hermes-paper/40 border-dashed border-hermes-ink/20 opacity-70"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-bold text-hermes-ink uppercase">
                    :{step.name.toUpperCase()}:
                  </span>

                  {step.duration_ms > 0 && (
                    <span className="font-bold text-hermes-blue">
                      {step.duration_ms}ms
                    </span>
                  )}
                </div>

                <p className="mt-1 text-hermes-charcoal font-body leading-snug">
                  {step.detail}
                </p>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
