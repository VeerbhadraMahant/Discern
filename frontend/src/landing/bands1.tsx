import { CheckCircle } from "@phosphor-icons/react";
import { PAPER_TITLE, PAPER_URL, PILLARS, SHEET, STATS } from "./facts";
import { Band, Corners, Panel } from "./kit";
import { Still } from "./scene";

const PROMISE_STATS = STATS.filter((s) => ["verifier", "boxf1", "mae", "exact"].includes(s.id));
const VIDEO_SOURCE = PROMISE_STATS[0]!.source;

/** The pull-quote band: the brand's one-line differentiator, with the measured figures beside it, written as sentences. */
export function PromiseBand() {
  return (
    <section id="s-promise" tabIndex={-1} aria-labelledby="s-promise-h" className="scroll-mt-16 bg-bone">
      <div className="page grid gap-12 py-14 md:py-24 lg:grid-cols-12 lg:gap-16">
        <div className="lg:col-span-7">
          <div aria-hidden="true" className="mb-8 h-2 w-24 bg-ink" />
          <h2 id="s-promise-h" className="type-display-l text-balance">
            Other tools answer confidently. Discern answers with its evidence and tells you where it is unsure.
          </h2>
          <p className="measure mt-8 type-body-l">
            It builds on{" "}
            <a href={PAPER_URL} target="_blank" rel="noreferrer noopener" className="underline decoration-1 underline-offset-[3px]">
              {PAPER_TITLE}
            </a>
            : self-adaptive restoration, multi-expertise detection and experience harvesting, extended to video and question answering.
          </p>
        </div>
        <div className="lg:col-span-5 lg:pt-10">
          <h3 className="type-h2">Measured here, not borrowed</h3>
          <ul className="mt-5 flex flex-col gap-5">
            {PROMISE_STATS.map((s) => (
              <li key={s.id} className="border-l-[3px] border-ink pl-4 type-body-l">
                {s.before} <strong className="num font-semibold">{s.value}</strong> {s.after}
              </li>
            ))}
          </ul>
          <p className="mt-5 type-body-s">Source: {VIDEO_SOURCE}. Some of these figures are not flattering, and they are the point.</p>
        </div>
      </div>
    </section>
  );
}

/** Contact sheet: one clip walked through the pipeline, five drawn stills joined by a ruler with timecodes beneath. */
export function ContactSheet() {
  return (
    <Band
      id="s-pipeline"
      tone="ink"
      title="From a degraded frame to a checked answer"
      lead="One clip, five stills, drawn for this page. They are an illustration, not a recorded result."
    >
      <div
        role="region"
        aria-label="Pipeline contact sheet, scrolls sideways on narrow screens"
        tabIndex={0}
        className="-mx-4 overflow-x-auto px-4 pb-2 md:-mx-8 md:px-8 lg:mx-0 lg:overflow-visible lg:px-0 lg:pb-0"
      >
        <ol className="flex snap-x snap-mandatory lg:grid lg:grid-cols-5">
          {SHEET.map((f) => (
            <li key={f.id} className="min-w-[78%] snap-start px-3 sm:min-w-[46%] lg:min-w-0">
              <div className="relative border border-parchment">
                <Still kind={f.id} />
                <Corners tone="parchment" />
              </div>
              <div className="relative mt-6 border-t border-parchment pt-3">
                <span aria-hidden="true" className="absolute -top-2 left-0 h-4 w-0.5 bg-parchment" />
                <p className="type-caption num">{f.time}</p>
                <h3 className="mt-3 type-h3">{f.title}</h3>
                <p className="mt-2 type-body-s">{f.caption}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
      <p className="mt-6 type-body-s lg:hidden">Swipe sideways to see all five frames.</p>
    </Band>
  );
}

type PillarId = (typeof PILLARS)[number]["id"];
const pillar = (id: PillarId) => PILLARS.find((x) => x.id === id)!;

function Proof({ id }: { id: PillarId }) {
  const p = pillar(id);
  return (
    <div className="mt-6 border-l-[3px] border-ink pl-4">
      <p className="type-body">{p.proof}</p>
      <p className="mt-2 type-body-s">Source: {p.source}</p>
    </div>
  );
}

function Token({ children }: { children: string }) {
  return <span className="num border-2 border-parchment px-1.5">{children}</span>;
}

/** Three pillars, composed differently on purpose: a stacked panel, a bordered split panel and a full-width ink panel. */
export function Pillars() {
  const clean = pillar("clean");
  const evidence = pillar("evidence");
  const checked = pillar("checked");
  return (
    <Band id="s-pillars" title="Three promises, each with a measurement">
      <div className="grid gap-8 lg:grid-cols-12">
        <Panel tone="bone" className="lg:col-span-5">
          <div className="grid grid-cols-2 gap-3">
            <figure>
              <div className="border border-ink">
                <Still kind="degraded" />
              </div>
              <figcaption className="mt-2 type-body-s">A hazy shot is restored.</figcaption>
            </figure>
            <figure>
              <div className="border border-ink">
                <Still kind="cleaned" />
              </div>
              <figcaption className="mt-2 type-body-s">A clear shot is left alone.</figcaption>
            </figure>
          </div>
          <h3 className="mt-8 type-h2">{clean.title}</h3>
          <p className="mt-2 type-body-l">{clean.text}</p>
          <Proof id="clean" />
        </Panel>

        <div className="grid gap-6 border-2 border-ink p-6 md:grid-cols-2 md:items-center lg:col-span-7">
          <div>
            <h3 className="type-h2">{evidence.title}</h3>
            <p className="mt-2 type-body-l">{evidence.text}</p>
            <Proof id="evidence" />
          </div>
          <figure>
            <div className="relative border border-ink">
              <Still kind="tracked" />
              <Corners />
            </div>
            <figcaption className="mt-4">
              <ul className="num flex flex-col gap-1 type-body-s">
                <li>Track 2, car, 0:03 to 0:07</li>
                <li>Crop, box and rationale</li>
                <li>Status: accepted</li>
              </ul>
            </figcaption>
          </figure>
        </div>

        <Panel tone="ink" className="grid gap-8 lg:col-span-12 lg:grid-cols-12 lg:items-center lg:p-10">
          <div className="lg:col-span-5">
            <h3 className="type-h2">{checked.title}</h3>
            <p className="mt-2 type-body-l">{checked.text}</p>
            <div className="mt-6 border-l-[3px] border-parchment pl-4">
              <p className="type-body">{checked.proof}</p>
              <p className="mt-2 type-body-s">Source: {checked.source}</p>
            </div>
          </div>
          <figure className="lg:col-span-7">
            <div className="border border-parchment p-5">
              <p className="type-body-l">
                Two tracks match, from <Token>0:03</Token> to <Token>0:07</Token>.
              </p>
              <ul className="num mt-5 flex flex-col gap-2 border-t border-parchment pt-4 type-body">
                <li className="flex items-center gap-2">
                  <CheckCircle size={20} aria-hidden="true" />
                  Computed facts: 2 tracks, matches the answer
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle size={20} aria-hidden="true" />
                  Computed facts: 0:03 to 0:07, matches the answer
                </li>
              </ul>
            </div>
            <figcaption className="mt-3 type-body-s">Illustration of the check, not a recorded result.</figcaption>
          </figure>
        </Panel>
      </div>
    </Band>
  );
}
