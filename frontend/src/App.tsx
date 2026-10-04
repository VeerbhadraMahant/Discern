import { lazy, Suspense, useEffect, useRef } from "react";
import { useDiscern } from "./state/DiscernContext";
import { Header, Toast } from "./components/Layout";
import { useHashRoute } from "./lib/route";
import type { Route } from "./lib/route";
import { AboutView } from "./views/AboutView";
import { AskView } from "./views/AskView";
import { CleanView } from "./views/CleanView";
import { FeedbackView } from "./views/FeedbackView";
import { TraceView } from "./views/TraceView";

const Landing = lazy(() => import("./landing/Landing"));

const TITLES: Record<Route, string> = {
  home: "Ask a video a question, see the evidence",
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
    document.title = route === "home" ? `Discern | ${TITLES.home}` : `${TITLES[route]} | Discern`;
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
      {route === "home" ? (
        <Suspense fallback={<div className="min-h-dvh" role="status" aria-label="Loading" />}>
          <Landing main={main} />
        </Suspense>
      ) : (
        <>
          <Header route={route} />
          <main id="main" ref={main} tabIndex={-1}>
            {badSpace && (
              <p role="alert" className="page py-3 font-semibold">
                The space address in the link was ignored: only a hf.space address or a local server is accepted.
              </p>
            )}
            {route === "clean" && <CleanView />}
            {route === "ask" && <AskView />}
            {route === "trace" && <TraceView />}
            {route === "feedback" && <FeedbackView />}
            {route === "about" && <AboutView />}
          </main>
        </>
      )}
      <Toast />
    </div>
  );
}
