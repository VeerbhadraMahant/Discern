import type { PipelineStep } from "../lib/types";
import { CheckCircleIcon, SkipIcon } from "./icons";

export default function PipelineTrace({ steps }: { steps: PipelineStep[] }) {
  return (
    <div className="rounded-card bg-paper p-5 shadow-subtle">
      <h2 className="font-heading text-sm font-semibold uppercase tracking-wide text-charcoal">Agent pipeline</h2>
      <ol className="mt-4 space-y-0">
        {steps.map((step, i) => (
          <li key={step.name} className="relative flex gap-3 pb-5 last:pb-0">
            {i < steps.length - 1 && (
              <span className="absolute left-[11px] top-6 h-[calc(100%-1.25rem)] w-px bg-slate/20" aria-hidden />
            )}
            <span
              className={`z-10 mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full ${
                step.status === "completed" ? "bg-accent text-paper" : "bg-cloud text-slate"
              }`}
            >
              {step.status === "completed" ? <CheckCircleIcon width={14} height={14} /> : <SkipIcon width={14} height={14} />}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <p className="text-sm font-medium text-ink">{step.name}</p>
                {step.status === "completed" && step.duration_ms > 0 && (
                  <span className="shrink-0 text-xs tabular-nums text-slate">{step.duration_ms}ms</span>
                )}
              </div>
              <p className="mt-0.5 text-sm leading-snug text-charcoal">{step.detail}</p>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
