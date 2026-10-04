import { ArrowClockwise, ChatCircleText, Info, ListChecks, Sparkle, ThumbsUp, WarningCircle, Broom } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import type { Route } from "../lib/route";
import { ROUTES } from "../lib/route";
import { useDiscern } from "../state/DiscernContext";
import { Button, Stamp } from "./ui";

const NAV: Record<Route, { label: string; icon: ReactNode }> = {
  clean: { label: "Clean", icon: <Sparkle size={20} aria-hidden="true" /> },
  ask: { label: "Ask", icon: <ChatCircleText size={20} aria-hidden="true" /> },
  trace: { label: "Trace", icon: <ListChecks size={20} aria-hidden="true" /> },
  feedback: { label: "Feedback", icon: <ThumbsUp size={20} aria-hidden="true" /> },
  about: { label: "About", icon: <Info size={20} aria-hidden="true" /> },
};

export function ConnectionStatus() {
  const { connection, retryConnect } = useDiscern();
  if (connection.status === "connecting") {
    return (
      <p role="status" className="text-caption font-normal">
        Connecting
      </p>
    );
  }
  if (connection.status === "connected") {
    return (
      <p role="status" className="text-caption font-normal">
        Connected to {connection.host}
      </p>
    );
  }
  return (
    <p role="status" className="flex flex-wrap items-center gap-2 text-caption font-semibold">
      <WarningCircle size={20} className="text-ember" aria-hidden="true" />
      Unreachable
      <Button variant="link" onClick={retryConnect} icon={<ArrowClockwise size={16} aria-hidden="true" />}>
        Retry
      </Button>
    </p>
  );
}

export function Header({ route }: { route: Route }) {
  const { info, session, startOver, isMock } = useDiscern();
  return (
    <header className="z-20 md:sticky md:top-0 border-b border-ink bg-parchment">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-2 px-4 py-3 md:grid md:grid-cols-[1fr_auto_1fr] md:items-center md:px-8">
        <div className="order-2 flex items-center justify-between gap-3 md:order-1 md:flex-col md:items-start md:justify-start md:gap-0">
          <p className="text-caption font-normal">{info ? `Profile: ${info.profile}` : "Profile: unknown"}</p>
          <div className="flex items-center gap-3">
            <ConnectionStatus />
            {isMock && <Stamp>Demo data</Stamp>}
          </div>
        </div>
        <p className="order-1 text-center font-display text-subhead leading-none tracking-[-0.05em] md:order-2 md:text-heading-sm">
          Discern
        </p>
        <div className="order-3 flex justify-end">
          {session && (
            <Button variant="link" onClick={() => void startOver()} icon={<Broom size={20} aria-hidden="true" />}>
              Start over
            </Button>
          )}
        </div>
      </div>
      <nav aria-label="Sections" className="mx-auto max-w-[1440px] px-2 md:px-8">
        <ul className="flex justify-between gap-1 md:justify-end md:gap-6">
          {ROUTES.map((r) => {
            const active = r === route;
            return (
              <li key={r} className="flex-1 md:flex-none">
                <a
                  href={`#/${r}`}
                  aria-current={active ? "page" : undefined}
                  className={`flex min-h-11 flex-col items-center justify-center gap-0.5 border-b-2 px-1 py-1 text-caption md:flex-row md:gap-2 md:px-2 md:text-base ${
                    active ? "border-ember font-semibold" : "border-transparent font-normal"
                  }`}
                >
                  {NAV[r].icon}
                  <span>{NAV[r].label}</span>
                </a>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}

export function Banner({ title, kicker }: { title: string; kicker: string }) {
  return (
    <section className="bg-ink text-parchment">
      <div className="mx-auto max-w-[1440px] px-4 pb-6 pt-5 md:px-8">
        <h1 className="font-display text-[clamp(64px,14vw,200px)] leading-[0.8] tracking-[-0.05em]">{title}</h1>
        <p className="mt-4 max-w-[60ch] text-subhead font-normal leading-tight">{kicker}</p>
      </div>
    </section>
  );
}

export function Toast() {
  const { toast } = useDiscern();
  return (
    <div aria-live="polite" role="status" className="pointer-events-none fixed inset-x-0 bottom-4 z-30 flex justify-center px-4">
      {toast && <p className="rise rounded-tag border border-ink bg-parchment px-4 py-3 text-ink">{toast}</p>}
    </div>
  );
}
