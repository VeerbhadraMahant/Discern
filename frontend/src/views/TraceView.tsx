import { ArrowClockwise, CaretDown, CaretRight, CheckCircle, WarningCircle } from "@phosphor-icons/react";
import { Fragment, useState } from "react";
import { Banner } from "../components/Layout";
import { Button, EmptyState, ErrorNotice } from "../components/ui";
import { useDiscern } from "../state/DiscernContext";

export function TraceView() {
  const { events, session, refreshTrace, phase, error, clearError } = useDiscern();
  const [open, setOpen] = useState<Set<number>>(new Set());
  const toggle = (i: number) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  return (
    <>
      <Banner title="Trace" kicker="Every decision Discern made for this session, in order." />
      <div className="page flex flex-col gap-6 py-10">
        {error && <ErrorNotice error={error} onDismiss={clearError} />}
        {session && (
          <div>
            <Button variant="link" onClick={() => void refreshTrace()} icon={<ArrowClockwise size={20} aria-hidden="true" />}>
              Refresh from the app
            </Button>
          </div>
        )}
        {events.length === 0 ? (
          <EmptyState title="No events yet">
            <p>Events appear here as you clean, detect and ask.</p>
            {phase === "idle" && !session && (
              <p>
                <a href="#/clean" className="link">
                  Start on the Clean tab
                </a>
              </p>
            )}
          </EmptyState>
        ) : (
          <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Scrollable table">
            <table className="type-body-s w-full min-w-[640px] border-collapse text-left">
              <caption className="sr-only">Pipeline events in order</caption>
              <thead>
                <tr className="border-b border-ink">
                  {["Node", "Decision", "Duration ms", "GPU ms", "Fallback"].map((h) => (
                    <th key={h} scope="col" className={`sticky top-0 bg-parchment px-3 py-2 type-label ${h.endsWith("ms") ? "text-right" : ""}`}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {events.map((e, i) => {
                  const expanded = open.has(i);
                  return (
                    <Fragment key={i}>
                      <tr className="border-b border-ink/40">
                        <td className="px-3 py-1">
                          <button
                            type="button"
                            className="link"
                            aria-expanded={expanded}
                            aria-controls={`ev-${i}`}
                            onClick={() => toggle(i)}
                          >
                            {expanded ? <CaretDown size={16} aria-hidden="true" /> : <CaretRight size={16} aria-hidden="true" />}
                            {e.node}
                          </button>
                        </td>
                        <td className="px-3 py-2 font-normal">{e.decision}</td>
                        <td className="num px-3 py-2 text-right font-normal">{e.duration_ms}</td>
                        <td className="num px-3 py-2 text-right font-normal">{e.gpu_ms ?? "none"}</td>
                        <td className="px-3 py-2 font-normal">
                          {e.fallback_used ? (
                            <span className="type-label inline-flex items-center gap-1">
                              <WarningCircle size={20} className="text-ember" aria-hidden="true" />
                              fallback
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1">
                              <CheckCircle size={20} aria-hidden="true" />
                              none
                            </span>
                          )}
                        </td>
                      </tr>
                      {expanded && (
                        <tr id={`ev-${i}`} className="border-b border-ink/40 bg-bone">
                          <td colSpan={5} className="px-3 py-3">
                            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
                              <dt className="type-label">Input</dt>
                              <dd className="type-code break-words">{e.input_summary}</dd>
                              <dt className="type-label">Rationale</dt>
                              <dd className="font-normal">{e.rationale}</dd>
                              <dt className="type-label">Prompt version</dt>
                              <dd className="type-code ident">{e.prompt_version ?? "none"}</dd>
                            </dl>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
