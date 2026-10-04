import { ArrowUp, CaretDown, Copy, GithubLogo, Newspaper } from "@phosphor-icons/react";
import { useId, useState } from "react";
import { Wordmark } from "../components/Wordmark";
import { FAQ, METHOD, MODELS, MODELS_SOURCE, NOT_DEPLOYED, PAPER_TITLE, PAPER_URL, PRIVACY, REPO_URL, RUN_LOCALLY, STACK, STATS } from "./facts";
import { APP_HREF, Band, DEMO_HREF, EmberDot, PrimaryLink } from "./kit";

export function PrivacyModels() {
  return (
    <Band id="s-models" tone="bone" title="Your upload is yours, and every model keeps its license">
      <ul className="grid gap-x-8 gap-y-6 md:grid-cols-2 lg:grid-cols-5">
        {PRIVACY.map((p) => (
          <li key={p.id} className="border-t-[3px] border-ink pt-3">
            <h3 className="type-h3">{p.title}</h3>
            <p className="mt-2 type-body">{p.text}</p>
          </li>
        ))}
      </ul>
      <p className="measure mt-4 type-body-s">Only what is implemented is listed here.</p>
      <h3 className="mt-14 type-h2">Open models, with their licenses</h3>
      <p className="measure mt-2 type-body-l">Code is ours; every model keeps its own license. Check the column before any commercial or hosted use.</p>
      <div role="region" aria-label="Models and licenses table, scrollable" tabIndex={0} className="relative mt-6 overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-left">
          <caption className="mb-3 text-left type-body-s">Source: {MODELS_SOURCE}. Hosted-use notes are not legal advice.</caption>
          <thead>
            <tr className="border-b-2 border-ink">
              {["Model", "Role", "License", "Hosted-use note"].map((h) => (
                <th key={h} scope="col" className="py-2 pr-4 font-semibold">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODELS.map((m) => (
              <tr key={m.name} className="border-b border-ink/40 align-top">
                <th scope="row" className="py-2 pr-4 font-semibold">
                  {m.name}
                </th>
                <td className="py-2 pr-4">{m.role}</td>
                <td className="py-2 pr-4">{m.license}</td>
                <td className="py-2 pr-4">{m.hosted}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Band>
  );
}

/** Numbered principles in two staggered columns: the right column starts lower, each with an evidence line (ember dot = evidence is here). */
export function Methodology() {
  return (
    <Band
      id="s-method"
      tone="ink"
      title="How we build and judge it"
      lead="The method matters as much as the model. These are the rules the code and the numbers on this page follow."
    >
      <ol className="grid items-start gap-x-16 gap-y-10 lg:grid-cols-2">
        {METHOD.map((p, i) => (
          <li key={p.id} className={`border-t-[3px] border-parchment pt-5 ${i % 2 === 1 ? "lg:mt-16" : ""}`}>
            <div className="grid grid-cols-[3.5rem_minmax(0,1fr)] gap-x-4 md:grid-cols-[5rem_minmax(0,1fr)]">
              <span className="num type-display-l" aria-hidden="true">
                {i + 1}
              </span>
              <div>
                <h3 className="type-h2">
                  <span className="sr-only">{i + 1}. </span>
                  {p.title}
                </h3>
                <p className="measure mt-3 type-body-l">{p.text}</p>
                <p className="mt-4 flex items-start gap-2 type-body-s">
                  <span className="mt-1">
                    <EmberDot size={10} />
                  </span>
                  <span>
                    Evidence: {p.evidence}. Source: {p.source}.
                  </span>
                </p>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </Band>
  );
}

function FaqItem({ q, a }: { q: string; a: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className="border-b border-ink">
      <h3>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={`${id}-a`}
          onClick={() => setOpen((o) => !o)}
          className="flex min-h-11 w-full items-center justify-between gap-4 py-3 text-left type-body-l"
        >
          <span>{q}</span>
          <CaretDown size={20} aria-hidden="true" className={`shrink-0 ${open ? "rotate-180" : ""}`} />
        </button>
      </h3>
      <div id={`${id}-a`} hidden={!open} className="pb-4">
        <p className="measure">{a}</p>
      </div>
    </div>
  );
}

export function FaqSection() {
  return (
    <Band id="s-faq" split title="Frequently asked">
      <div className="border-t-2 border-ink">
        {FAQ.map((f) => (
          <FaqItem key={f.id} q={f.q} a={f.a} />
        ))}
      </div>
    </Band>
  );
}

export function CodeBlock({ code }: { code: string }) {
  const [state, setState] = useState<"idle" | "ok" | "fail">("idle");
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setState("ok");
    } catch {
      setState("fail");
    }
  };
  return (
    <div className="bg-bone text-ink">
      <div className="flex flex-wrap items-center justify-between gap-x-3 border-b border-ink px-4 py-1">
        <p className="type-label">Terminal</p>
        <div className="flex items-center gap-3">
          <p role="status" className="type-body-s">
            {state === "ok" && "Copied"}
            {state === "fail" && "Copy failed: select the text and copy it by hand"}
          </p>
          <button type="button" onClick={() => void copy()} className="link">
            <Copy size={20} aria-hidden="true" />
            Copy commands
          </button>
        </div>
      </div>
      <pre tabIndex={0} aria-label="Commands to run Discern locally" className="overflow-x-auto whitespace-pre-wrap break-words p-4 type-code">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function FinalCta() {
  return (
    <section id="s-run" tabIndex={-1} aria-labelledby="s-run-h" className="scroll-mt-16 bg-ink text-parchment">
      <div className="page grid grid-cols-[minmax(0,1fr)] gap-10 py-14 md:py-24 lg:grid-cols-12 lg:items-center lg:gap-14">
        <div className="lg:col-span-5">
          <h2 id="s-run-h" className="type-display-l">
            Run it locally
          </h2>
          <p className="mt-5 max-w-[50ch] type-body-l">
            No hosted service exists yet. Three commands start the backend and the frontend. Or open the app with demo data and no backend at all.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3">
            <PrimaryLink href={APP_HREF} onInk>
              Open the app
            </PrimaryLink>
            <a href={DEMO_HREF} className="link">
              Try with demo data
            </a>
            <a href={REPO_URL} target="_blank" rel="noreferrer noopener" className="link">
              Source code on GitHub
            </a>
          </div>
        </div>
        <div className="lg:col-span-7">
          <CodeBlock code={RUN_LOCALLY} />
          <p className="mt-3 type-body-s">Run the third line in a second terminal, once the backend is up.</p>
        </div>
      </div>
    </section>
  );
}

export function Footer() {
  const tests = STATS.find((s) => s.id === "tests")!;
  return (
    <footer className="border-t-2 border-ink">
      <div className="page grid gap-8 py-12 md:grid-cols-12">
        <div className="md:col-span-5">
          <Wordmark href="#/" />
          <p className="measure mt-3 type-body">
            Code is ours; each model keeps its own license (see the models table). A research prototype. {NOT_DEPLOYED}
          </p>
          <p className="measure mt-3 type-body-s">
            {tests.before} {tests.value} {tests.after} Built with {STACK.join(", ")}.
          </p>
        </div>
        <ul className="flex flex-col gap-1 md:col-span-3 md:col-start-7">
          <li>
            <a href={REPO_URL} target="_blank" rel="noreferrer noopener" className="link">
              <GithubLogo size={20} aria-hidden="true" />
              Repository on GitHub
            </a>
          </li>
          <li>
            <a href={PAPER_URL} target="_blank" rel="noreferrer noopener" className="link">
              <Newspaper size={20} aria-hidden="true" />
              Paper: arXiv 2605.31174
            </a>
          </li>
        </ul>
        <div className="flex flex-col items-start gap-2 md:col-span-3">
          <p className="type-body-s">Based on {PAPER_TITLE}.</p>
          <a
            href="#/"
            onClick={(e) => {
              e.preventDefault();
              window.scrollTo({ top: 0 });
              document.getElementById("top")?.focus({ preventScroll: true });
            }}
            className="link"
          >
            <ArrowUp size={20} aria-hidden="true" />
            Back to top
          </a>
        </div>
      </div>
    </footer>
  );
}
