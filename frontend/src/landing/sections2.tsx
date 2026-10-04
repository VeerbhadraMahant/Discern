import { CheckCircle, Flask, MapPin, Quotes, WarningCircle } from "@phosphor-icons/react";
import { Card, Stamp } from "../components/ui";
import { DATASETS, DID_NOT_WORK, EXAMPLES, METHODS, QUERY_TYPES, fmt } from "./facts";
import type { DatasetId } from "./facts";
import { Section, Tabs } from "./kit";

export function QueryTypes() {
  return (
    <Section
      id="s-queries"
      title="Six kinds of question"
      lead="Locate and count were run with real models. The other four are implemented and tested with test doubles only."
    >
      <ul className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {QUERY_TYPES.map((q) => (
          <li key={q.id}>
            <Card as="article" className="flex h-full flex-col gap-3">
              <h3 className="type-h3">{q.name}</h3>
              <p className="flex gap-2">
                <Quotes size={20} className="mt-1 shrink-0" aria-hidden="true" />
                <span className="font-semibold">Example question: {q.question}</span>
              </p>
              <p>
                <span className="font-semibold">Evidence: </span>
                {q.evidence}
              </p>
              <p className="mt-auto pt-2">
                {q.verified ? (
                  <Stamp icon={<CheckCircle size={16} aria-hidden="true" />}>Verified with real models</Stamp>
                ) : (
                  <Stamp icon={<Flask size={16} aria-hidden="true" />}>Tested on test doubles only</Stamp>
                )}
              </p>
            </Card>
          </li>
        ))}
      </ul>
    </Section>
  );
}

export function AskDemo() {
  return (
    <Section id="s-example" title="What an exchange looks like" lead="Pick a question. The wording below is illustrative; no clip or answer here was recorded.">
      <Tabs
        label="Example questions"
        tabs={EXAMPLES.map((e) => ({ id: e.id, label: e.tab }))}
        render={(i) => {
          const e = EXAMPLES[i]!;
          return (
            <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
              <div className="flex flex-col gap-4">
                <p>
                  <Stamp icon={<WarningCircle size={16} className="text-ember" aria-hidden="true" />}>Example, not a recorded result</Stamp>
                </p>
                <p className="border-l-2 border-ink pl-4 type-body-l">{e.question}</p>
                <Card>
                  <p className="type-body-l">{e.answer}</p>
                </Card>
                <p className="type-body-s">{e.note}</p>
              </div>
              <Card className="self-start">
                <h3 className="type-h3">Evidence</h3>
                <ul className="mt-3 flex flex-col gap-2">
                  {e.evidence.map((line) => (
                    <li key={line} className="num flex gap-2">
                      <MapPin size={20} className="mt-0.5 shrink-0" aria-hidden="true" />
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            </div>
          );
        }}
      />
    </Section>
  );
}

const PATTERNS = ["bar-solid", "bar-hatch", "bar-dots", "bar-cross", "bar-rows"] as const;

function PatternDefs() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden="true" focusable="false">
      <defs>
        <pattern id="bar-solid" width="4" height="4" patternUnits="userSpaceOnUse">
          <rect width="4" height="4" className="fill-ink" />
        </pattern>
        <pattern id="bar-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="6" height="6" className="fill-parchment" />
          <line x1="0" y1="0" x2="0" y2="6" className="stroke-ink" strokeWidth="3" />
        </pattern>
        <pattern id="bar-dots" width="6" height="6" patternUnits="userSpaceOnUse">
          <rect width="6" height="6" className="fill-parchment" />
          <circle cx="3" cy="3" r="1.6" className="fill-ink" />
        </pattern>
        <pattern id="bar-cross" width="6" height="6" patternUnits="userSpaceOnUse">
          <rect width="6" height="6" className="fill-parchment" />
          <path d="M0 3H6M3 0V6" className="stroke-ink" strokeWidth="1.2" />
        </pattern>
        <pattern id="bar-rows" width="6" height="6" patternUnits="userSpaceOnUse">
          <rect width="6" height="6" className="fill-parchment" />
          <line x1="0" y1="3" x2="6" y2="3" className="stroke-ink" strokeWidth="2.4" />
        </pattern>
      </defs>
    </svg>
  );
}

function bestOf(id: DatasetId): number {
  return Math.max(...METHODS.map((m) => m.scores[id]));
}

export function BarChart({ dataset }: { dataset: DatasetId }) {
  const best = bestOf(dataset);
  const info = DATASETS.find((d) => d.id === dataset)!;
  const summary = `Bar chart of F1 at IoU 0.5 on ${info.label}, scale 0 to 1. ${METHODS.map((m) => `${m.name} ${fmt(m.scores[dataset])}`).join(", ")}. Highest: ${METHODS.filter((m) => m.scores[dataset] === best)
    .map((m) => m.name)
    .join(", ")}. The same numbers are in the table below.`;
  return (
    <figure role="group" aria-label={summary}>
      <figcaption className="mb-4 type-body-s">
        F1 at IoU 0.5, {info.note}. Bars start at 0 and the full width is 1.0. Each method has its own fill pattern and a printed value.
      </figcaption>
      <ul className="flex flex-col gap-4">
        {METHODS.map((m, i) => {
          const v = m.scores[dataset];
          const isBest = v === best && best > 0;
          return (
            <li key={m.id} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 md:grid-cols-[13rem_1fr_6rem]">
              <span className="order-1 font-semibold">{m.name}</span>
              <svg className="order-3 col-span-2 block h-6 w-full md:order-2 md:col-span-1" aria-hidden="true" focusable="false">
                <rect x="0" y="0" width="100%" height="24" className="fill-parchment stroke-ink" strokeWidth="2" />
                {[25, 50, 75].map((t) => (
                  <line key={t} x1={`${t}%`} x2={`${t}%`} y1="0" y2="24" className="stroke-ink" strokeWidth="1" opacity="0.25" />
                ))}
                {v > 0 && <rect x="0" y="0" width={`${v * 100}%`} height="24" fill={`url(#${PATTERNS[i % PATTERNS.length]})`} className="stroke-ink" strokeWidth="2" />}
              </svg>
              <span className="num order-2 flex items-center justify-end gap-1 md:order-3">
                {isBest && <CheckCircle size={16} aria-hidden="true" />}
                <span className={isBest ? "font-semibold" : ""}>{fmt(v)}</span>
                {isBest && <span className="type-body-s">best</span>}
              </span>
            </li>
          );
        })}
      </ul>
      <div className="num mt-3 hidden justify-between type-caption md:ml-[13.75rem] md:mr-[6.75rem] md:flex" aria-hidden="true">
        <span>0</span>
        <span>0.25</span>
        <span>0.5</span>
        <span>0.75</span>
        <span>1.0</span>
      </div>
    </figure>
  );
}

export function ResultsTable() {
  return (
    <div role="region" aria-label="Baseline F1 table, scrollable" tabIndex={0} className="relative overflow-x-auto">
      <table className="num w-full min-w-[640px] border-collapse text-left">
        <caption className="mb-3 text-left type-body-s">
          F1 at IoU 0.5 on fixed 100-image gate subsets. Bold marks the best method in a column. This table has the same numbers as the chart.
        </caption>
        <thead>
          <tr className="border-b border-ink">
            <th scope="col" className="py-2 pr-3 font-semibold">
              Method
            </th>
            {DATASETS.map((d) => (
              <th key={d.id} scope="col" className="py-2 pr-3 text-right font-semibold">
                {d.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METHODS.map((m) => (
            <tr key={m.id} className="border-b border-ink/40">
              <th scope="row" className="py-2 pr-3 font-normal">
                {m.name}
              </th>
              {DATASETS.map((d) => {
                const isBest = m.scores[d.id] === bestOf(d.id) && bestOf(d.id) > 0;
                return (
                  <td key={d.id} className={`py-2 pr-3 text-right ${isBest ? "font-semibold" : ""}`}>
                    {fmt(m.scores[d.id])}
                    {isBest && <span className="sr-only"> (best)</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Results() {
  return (
    <Section
      id="s-results"
      title="Baselines on degraded images"
      lead="No single detector wins everywhere, and DarkFace stays hard for every method."
    >
      <PatternDefs />
      <Card className="p-4 md:p-6">
        <Tabs
          label="Dataset"
          tabs={DATASETS.map((d) => ({ id: d.id, label: d.label }))}
          render={(i) => <BarChart dataset={DATASETS[i]!.id} />}
        />
      </Card>
      <div className="mt-10">
        <ResultsTable />
      </div>
      <p className="mt-6 max-w-[65ch] type-body-s">
        Thresholds were tuned on separate 50-image splits. Datasets come from unofficial Hugging Face mirrors. The COCO subset was scored before the loader started skipping crowd boxes, and the legacy v0 row picks its restorer from the dataset label. Source: repository README, Measured results.
      </p>
    </Section>
  );
}

export function Honesty() {
  return (
    <Section id="s-honesty" title="What did not work" lead="The negative results are part of the result. Here they are plainly.">
      <ul className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {DID_NOT_WORK.map((d) => (
          <li key={d.id}>
            <Card as="article" className="h-full">
              <WarningCircle size={24} className="text-ember" aria-hidden="true" />
              <h3 className="mt-3 type-h3">{d.title}</h3>
              <p className="mt-3">{d.text}</p>
              <p className="mt-3 type-body-s">Source: {d.source}</p>
            </Card>
          </li>
        ))}
      </ul>
    </Section>
  );
}
