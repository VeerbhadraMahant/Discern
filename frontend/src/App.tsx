import { useEffect, useRef } from "react";
import { useDiscern } from "./state/DiscernContext";
import { Header, Toast } from "./components/Layout";
import { useHashRoute } from "./lib/route";
import type { Route } from "./lib/route";
import { AboutView } from "./views/AboutView";
import { AskView } from "./views/AskView";
import { CleanView } from "./views/CleanView";
import { FeedbackView } from "./views/FeedbackView";
import { TraceView } from "./views/TraceView";

const TITLES: Record<Route, string> = {
  clean: "Clean",
  ask: "Ask",
  trace: "Trace",
  feedback: "Feedback",
  about: "About",
};

export function Shell() {
  const [route] = useHashRoute();
  const { badSpace } = useDiscern();
  const main = useRef<HTMLElement>(null);
  const first = useRef(true);

  useEffect(() => {
    document.title = `${TITLES[route]} | Discern`;
    if (first.current) {
      first.current = false;
      return;
    }
    main.current?.focus();
  }, [route]);

  return (
    <div className="min-h-dvh">
      <a
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          main.current?.focus();
        }}
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:bg-ink focus:px-4 focus:py-3 focus:text-parchment"
      >
        Skip to main content
      </a>
      <Header route={route} />
      <main id="main" ref={main} tabIndex={-1}>
        {badSpace && (
          <p role="alert" className="mx-auto max-w-[1440px] px-4 py-3 font-semibold md:px-8">
            The space address in the link was ignored: only a hf.space address or a local server is accepted.
          </p>
        )}
        {route === "clean" && <CleanView />}
        {route === "ask" && <AskView />}
        {route === "trace" && <TraceView />}
        {route === "feedback" && <FeedbackView />}
        {route === "about" && <AboutView />}
      </main>
      <Toast />
    </div>
  );
}
