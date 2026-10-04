import type { RefObject } from "react";
import { Header } from "../components/Layout";
import { goToSection, useReadingProgress, useScrollSpy } from "./hooks";
import { AnnouncementBar, Features, Hero, HowItWorks, StatBand, Strip } from "./sections1";
import { AskDemo, Honesty, QueryTypes, Results } from "./sections2";
import { FaqSection, FinalCta, Footer, Models, Privacy, Roadmap, Stack } from "./sections3";

const NAV: ReadonlyArray<readonly [string, string]> = [
  ["s-features", "Features"],
  ["s-how", "How it works"],
  ["s-queries", "Queries"],
  ["s-example", "Example"],
  ["s-results", "Results"],
  ["s-honesty", "Honesty"],
  ["s-privacy", "Privacy"],
  ["s-models", "Models"],
  ["s-roadmap", "Roadmap"],
  ["s-faq", "FAQ"],
];
const IDS = NAV.map(([id]) => id);

function SectionNav() {
  const active = useScrollSpy(IDS);
  return (
    <nav aria-label="On this page" className="sticky top-0 z-20 hidden border-b border-ink bg-parchment lg:block">
      <ul className="mx-auto flex max-w-[1440px] items-center justify-between gap-2 px-8">
        {NAV.map(([id, label]) => (
          <li key={id}>
            <a
              href={`#${id}`}
              aria-current={active === id ? "location" : undefined}
              onClick={(e) => {
                e.preventDefault();
                goToSection(id);
              }}
              className={`flex min-h-11 items-center border-b-2 px-1 type-label ${active === id ? "border-ember" : "border-transparent"}`}
            >
              {label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/** The landing page: everything between the skip link and the toast, including its own main and footer. */
export default function Landing({ main }: { main: RefObject<HTMLElement | null> }) {
  const bar = useReadingProgress();
  return (
    <>
      <div ref={bar} aria-hidden="true" className="fixed inset-x-0 top-0 z-40 h-[3px] origin-left bg-ink" style={{ transform: "scaleX(0)" }} />
      <AnnouncementBar />
      <Header route="home" sticky={false} />
      <SectionNav />
      <main id="main" ref={main} tabIndex={-1}>
        <Hero />
        <Strip />
        <StatBand />
        <Features />
        <HowItWorks />
        <QueryTypes />
        <AskDemo />
        <Results />
        <Honesty />
        <Privacy />
        <Models />
        <Stack />
        <Roadmap />
        <FaqSection />
        <FinalCta />
      </main>
      <Footer />
    </>
  );
}
