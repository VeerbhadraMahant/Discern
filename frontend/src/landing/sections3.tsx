import { ArrowUp, CaretDown, CheckCircle, CircleHalf, Copy, GithubLogo, Newspaper, WarningCircle } from "@phosphor-icons/react";
import { useId, useState } from "react";
import type { ReactNode } from "react";
import { Card } from "../components/ui";
import { FAQ, MILESTONES, MODELS, MODELS_SOURCE, PAPER_TITLE, PAPER_URL, PRIVACY, REPO_URL, RUN_LOCALLY, STACK } from "./facts";
import type { MilestoneStatus } from "./facts";
import { Section } from "./kit";
import { APP_HREF, DEMO_HREF } from "./sections1";

export function Privacy() {
  return (
    <Section id="s-privacy" title="Your upload is yours" lead="Only what is implemented is listed here.">
      <ul className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {PRIVACY.map((p) => (
          <li key={p.id}>
            <Card as="article" className="h-full">
              <h3 className="type-h3">{p.title}</h3>
              <p className="mt-3">{p.text}</p>
            </Card>
          </li>
        ))}
      </ul>
    </Section>
  );
}

export function Models() {
  return (
    <Section
      id="s-models"
      title="Open models, with their licenses"
      lead="Code is ours; every model keeps its own license. Check the column before any commercial or hosted use."
    >
      <div role="region" aria-label="Models and licenses table, scrollable" tabIndex={0} className="relative overflow-x-auto">
        <table className="w-full min-w-[760px] border-collapse text-left">
          <caption className="mb-3 text-left type-body-s">Source: {MODELS_SOURCE}. Hosted-use notes are not legal advice.</caption>
          <thead>
            <tr className="border-b border-ink">
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
    </Section>
  );
}

export function Stack() {
  return (
    <Section id="s-stack" title="What it is built with">
      <ul className="flex flex-wrap gap-3">
        {STACK.map((s) => (
          <li key={s} className="rounded-tag border border-ink bg-bone px-3 py-2">
            {s}
          </li>
        ))}
      </ul>
    </Section>
  );
}

const STATUS: Record<MilestoneStatus, { label: string; icon: ReactNode }> = {
  done: { label: "Done", icon: <CheckCircle size={20} aria-hidden="true" /> },
  limits: { label: "Done, with limits", icon: <CircleHalf size={20} aria-hidden="true" /> },
  open: { label: "Not done", icon: <WarningCircle size={20} className="text-ember" aria-hidden="true" /> },
};

export function Roadmap() {
  return (
    <Section id="s-roadmap" title="Milestones M0 to M11" lead="One milestone is not done, and most of the rest carry a caveat. Sources: discern-plan.md and the README.">
      <ol className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {MILESTONES.map((m) => (
          <li key={m.id}>
            <Card as="article" className="h-full">
              <div className="flex items-baseline justify-between gap-3">
                <p className="ident type-h2">{m.id}</p>
                <p className="flex items-center gap-1 type-label">
                  {STATUS[m.status].icon}
                  {STATUS[m.status].label}
                </p>
              </div>
              <h3 className="mt-3 font-semibold">{m.title}</h3>
              <p className="mt-1">{m.text}</p>
            </Card>
          </li>
        ))}
      </ol>
    </Section>
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
        <p className="max-w-[65ch]">{a}</p>
      </div>
    </div>
  );
}

export function FaqSection() {
  return (
    <Section id="s-faq" title="Frequently asked">
      <div className="max-w-[900px] border-t border-ink">
        {FAQ.map((f) => (
          <FaqItem key={f.id} q={f.q} a={f.a} />
        ))}
      </div>
    </Section>
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
      <div className="mx-auto grid max-w-[1440px] grid-cols-[minmax(0,1fr)] gap-10 px-4 py-12 md:px-8 md:py-16 lg:grid-cols-[1fr_1fr] lg:items-center">
        <div>
          <h2 id="s-run-h" className="type-display-l">
            Run it locally
          </h2>
          <p className="mt-5 max-w-[50ch] type-body-l">
            No hosted service exists yet. Clone the repository, start the backend, then the frontend. Or open the app with demo data and no backend at all.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3">
            <a href={APP_HREF} className="inline-flex min-h-11 items-center justify-center gap-2 rounded-tag bg-parchment px-5 py-2 text-ink">
              Open the app
            </a>
            <a href={DEMO_HREF} className="link">
              Try with demo data
            </a>
          </div>
        </div>
        <div>
          <CodeBlock code={RUN_LOCALLY} />
        </div>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="border-t-[3px] border-double border-ink">
      <div className="mx-auto grid max-w-[1440px] gap-8 px-4 py-10 md:grid-cols-[1.2fr_1fr_1fr] md:px-8">
        <div>
          <p className="type-wordmark">Discern</p>
          <p className="mt-3 max-w-[40ch]">
            Code is ours; each model keeps its own license (see the models table). A research prototype, not deployed yet.
          </p>
        </div>
        <ul className="flex flex-col gap-1">
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
        <div className="flex flex-col items-start gap-2">
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
