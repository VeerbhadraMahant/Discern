import { ArrowClockwise, ChatCircleText, House, Info, ListChecks, Sparkle, ThumbsUp, WarningCircle, Broom } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import type { Route } from "../lib/route";
import { ROUTES } from "../lib/route";

import { useDiscern } from "../state/DiscernContext";
import { Button, Stamp } from "./ui";

const NAV: Record<Route, { label: string; icon: ReactNode }> = {
  home: { label: "Home", icon: <House size={20} aria-hidden="true" /> },
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
      <p role="status" className="type-body-s">
        Connecting
      </p>
    );
  }
  if (connection.status === "connected") {
    return (
      <p role="status" className="type-body-s">
        Connected to {connection.host}
      </p>
    );
  }
  return (
    <p role="status" className="flex flex-wrap items-center gap-2 type-label">
      <WarningCircle size={20} className="text-ember" aria-hidden="true" />
      Unreachable
      <Button variant="link" onClick={retryConnect} icon={<ArrowClockwise size={16} aria-hidden="true" />}>
        Retry
      </Button>
    </p>
  );
}

export function Header({ route, sticky = true }: { route: Route; sticky?: boolean }) {
  const { info, session, startOver, isMock } = useDiscern();
  return (
    <header className={`z-20 border-b border-ink bg-parchment ${sticky ? "md:sticky md:top-0" : ""}`}>
      <div className="mx-auto flex max-w-[1440px] flex-col gap-2 px-4 py-3 md:grid md:grid-cols-[1fr_auto_1fr] md:items-center md:px-8">
        <div className="order-2 flex items-center justify-between gap-3 md:order-1 md:flex-col md:items-start md:justify-start md:gap-0">
          {route === "home" ? (
            <p className="type-body-s">Research prototype</p>
          ) : (
            <>
              <p className="type-body-s">{info ? (<>Profile: <span className="type-code ident">{info.profile}</span></>) : "Profile: unknown"}</p>
              <div className="flex items-center gap-3">
                <ConnectionStatus />
                {isMock && <Stamp>Demo data</Stamp>}
              </div>
            </>
          )}
        </div>
        <p className="order-1 text-center type-wordmark md:order-2">
          <a href="#/" className="inline-flex min-h-11 items-center px-2">
            Discern
          </a>
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
          {(["home", ...ROUTES] as const).map((r) => {
            const active = r === route;
            return (
              <li key={r} className="flex-1 md:flex-none">
                <a
                  href={r === "home" ? "#/" : `#/${r}`}
                  aria-current={active ? "page" : undefined}
                  className={`flex min-h-11 flex-col items-center justify-center gap-0.5 border-b-2 px-1 py-1 type-label md:flex-row md:gap-2 md:px-2 ${
                    active ? "border-ember" : "border-transparent"
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
      <div className="mx-auto max-w-[1440px] px-4 py-6 md:px-8">
        <h1 className="type-h1">{title}</h1>
        <p className="measure mt-2 type-body-l">{kicker}</p>
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
