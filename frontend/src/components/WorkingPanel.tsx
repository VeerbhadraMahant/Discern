import { Check } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import { Card, ProgressBar } from "./ui";

export const CLEAN_STEPS = ["Reading the scene", "Choosing a restorer", "Restoring the image", "Comparing original and restored", "Picking the better one"];
export const DETECT_STEPS = ["Choosing detectors", "Finding objects", "Grouping overlapping boxes", "Checking each group"];
export const ASK_STEPS = ["Planning the query", "Searching the video index", "Collecting evidence", "Writing and verifying the answer"];

const STEP_SECONDS = 4;

/** Seconds since mount, ticking once a second. */
function useElapsed(): number {
  const [s, setS] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => setS((v) => v + 1), 1000);
    return () => window.clearInterval(t);
  }, []);
  return s;
}

/**
 * Shows what Discern is doing while a call is in flight: a viewfinder with a scan line, the steps of the stage in
 * order, a timer, and (for video) the real progress reported by the server. Single calls do not report their step,
 * so for those the highlighted step is an estimate from elapsed time, and the panel says so.
 */
export function WorkingPanel({
  title,
  steps,
  message,
  progress,
}: {
  title: string;
  steps: string[];
  /** Real step text from the server (video). When set, the estimate note is not shown. */
  message?: string;
  progress?: number | null;
}) {
  const elapsed = useElapsed();
  const real = message !== undefined;
  const active = real ? -1 : Math.min(steps.length - 1, Math.floor(elapsed / STEP_SECONDS));
  return (
    <Card className="flex flex-col gap-5 md:flex-row md:items-center" role="status" aria-label={title}>
      <div className="viewfinder mx-auto shrink-0" aria-hidden="true">
        <span className="vf-corner vf-tl" />
        <span className="vf-corner vf-tr" />
        <span className="vf-corner vf-bl" />
        <span className="vf-corner vf-br" />
        <span className="vf-scan" />
        <span className="vf-dot" />
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <p className="type-h3">
          {title} <span className="num font-normal">({elapsed}s)</span>
        </p>
        {real && <p className="num font-semibold">{message}</p>}
        {progress !== undefined && <ProgressBar value={progress} label={`${title} progress`} />}
        <ol className="flex flex-col gap-1">
          {steps.map((s, i) => (
            <li key={s} className={`flex items-center gap-2 ${!real && i === active ? "font-semibold" : ""} ${!real && i > active ? "opacity-60" : ""}`}>
              {!real && i < active ? <Check size={16} aria-hidden="true" /> : <span className={`step-dot ${!real && i === active ? "step-active" : ""}`} aria-hidden="true" />}
              <span>{s}</span>
            </li>
          ))}
        </ol>
        {!real && <p className="type-body-s">This call does not report its step, so the highlighted step is an estimate from elapsed time. Cleaning can take a while on a first run while models load.</p>}
      </div>
    </Card>
  );
}
