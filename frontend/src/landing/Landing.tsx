import type { RefObject } from "react";
import { Wordmark } from "../components/Wordmark";
import { Honesty, Results } from "./bands3";
import { Bento, Product, Queries } from "./bands2";
import { ContactSheet, Pillars, PromiseBand } from "./bands1";
import { FaqSection, FinalCta, Footer, Methodology, PrivacyModels } from "./bands4";
import { Hero } from "./hero";
import { goToSection, useScrollSpy } from "./hooks";
import { APP_HREF, PrimaryLink } from "./kit";

export const NAV: ReadonlyArray<readonly [string, string]> = [
  ["s-pipeline", "Pipeline"],
  ["s-product", "Product"],
  ["s-queries", "Questions"],
  ["s-results", "Results"],
  ["s-honesty", "Honesty"],
  ["s-method", "Methodology"],
  ["s-models", "Models"],
  ["s-faq", "FAQ"],
];
const IDS = NAV.map(([id]) => id);

/** One slim sticky bar: the wordmark at the left edge of the shared container, the section links, the call to action at the right edge. */
function LandingHeader() {
  const active = useScrollSpy(IDS);
  return (
    <header className="sticky top-0 z-30 border-b border-ink bg-parchment">
      <div className="page flex min-h-14 items-center justify-between gap-4">
        <Wordmark href="#/" />
        <nav aria-label="On this page" className="hidden lg:block">
          <ul className="flex items-center gap-6">
            {NAV.map(([id, label]) => (
              <li key={id}>
                <a
                  href={`#${id}`}
                  aria-current={active === id ? "location" : undefined}
                  onClick={(e) => {
                    e.preventDefault();
                    goToSection(id);
                  }}
                  className={`flex min-h-11 items-center border-b-2 type-label ${active === id ? "border-ember" : "border-transparent"}`}
                >
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
        <PrimaryLink href={APP_HREF}>Open the app</PrimaryLink>
      </div>
    </header>
  );
}

/** The landing page: everything between the skip link and the toast, including its own main and footer. */
export default function Landing({ main }: { main: RefObject<HTMLElement | null> }) {
  return (
    <>
      <LandingHeader />
      <main id="main" ref={main} tabIndex={-1}>
        <Hero />
        <PromiseBand />
        <ContactSheet />
        <Pillars />
        <Product />
        <Bento />
        <Queries />
        <Results />
        <Honesty />
        <Methodology />
        <PrivacyModels />
        <FaqSection />
        <FinalCta />
      </main>
      <Footer />
    </>
  );
}
