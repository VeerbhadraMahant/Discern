import { CheckCircle } from "@phosphor-icons/react";
import { DATASETS, DID_NOT_WORK, METHODS, STATS, fmt } from "./facts";
import type { DatasetId } from "./facts";
import { Band, Panel, Tabs } from "./kit";

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
      <ul className="flex flex-col gap-5">
        {METHODS.map((m, i) => {
          const v = m.scores[dataset];
          const isBest = v === best && best > 0;
          return (
            <li key={m.id} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 md:grid-cols-[13rem_1fr_6rem]">
              <span className="order-1 font-semibold">{m.name}</span>
              <svg className="order-3 col-span-2 block h-8 w-full md:order-2 md:col-span-1" aria-hidden="true" focusable="false">
                <rect x="0" y="0" width="100%" height="32" className="fill-parchment stroke-ink" strokeWidth="2" />
                {[25, 50, 75].map((t) => (
                  <line key={t} x1={`${t}%`} x2={`${t}%`} y1="0" y2="32" className="stroke-ink" strokeWidth="1" opacity="0.25" />
                ))}
                {v > 0 && <rect x="0" y="0" width={`${v * 100}%`} height="32" fill={`url(#${PATTERNS[i % PATTERNS.length]})`} className="stroke-ink" strokeWidth="2" />}
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
          <tr className="border-b border-parchment">
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
            <tr key={m.id} className="border-b border-parchment/40">
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


function stat(id: string): string {
  return STATS.find((s) => s.id === id)!.value;
}

function leader(id: DatasetId) {
  return METHODS.find((m) => m.scores[id] === bestOf(id))!;
}

/** Three findings written as sentences, every number computed from facts.ts. */
function findings(): string[] {
  const coco = leader("coco");
  const haze = leader("hazydet");
  return [
    `No single detector wins everywhere: ${coco.name} leads on COCO at ${fmt(coco.scores.coco)}, and ${haze.name} leads on HazyDet at ${fmt(haze.scores.hazydet)}.`,
    `DarkFace stays hard for every method: the best F1 there is ${fmt(bestOf("darkface"))}.`,
    `On 24 synthetic clips the verifier passed ${stat("verifier")} of answers, but only ${stat("exact")} of counts matched exactly.`,
  ];
}

export function Results() {
  const [lead, ...rest] = findings() as [string, string, string];
  return (
    <Band id="s-results" tone="ink" title="Baselines on degraded images">
      <PatternDefs />
      <p className="type-h1 max-w-[46ch]">{lead}</p>
      <Panel tone="bone" className="mt-8 p-4 md:p-8">
        <Tabs label="Dataset" tabs={DATASETS.map((d) => ({ id: d.id, label: d.label }))} render={(i) => <BarChart dataset={DATASETS[i]!.id} />} />
      </Panel>
      <ul className="mt-10 grid gap-6 md:grid-cols-2">
        {rest.map((f) => (
          <li key={f} className="border-t-[3px] border-parchment pt-4 type-body-l">
            {f}
          </li>
        ))}
      </ul>
      <div className="mt-12">
        <ResultsTable />
      </div>
      <p className="mt-6 max-w-[65ch] type-body-s">
        Thresholds were tuned on separate 50-image splits. Datasets come from unofficial Hugging Face mirrors. The COCO subset was scored before the loader started skipping crowd boxes, and the legacy v0 row picks its restorer from the dataset label. Source: repository README, Measured results.
      </p>
    </Band>
  );
}

/** The brand's differentiator: a bone panel under a thick ink rule, and a plain list. */
export function Honesty() {
  return (
    <Band id="s-honesty" split title="What did not work" lead="The negative results are part of the result. Here they are plainly.">
      <div className="border-t-8 border-ink bg-bone px-6 py-2 md:px-10">
        <ol className="flex flex-col">
          {DID_NOT_WORK.map((d) => (
            <li key={d.id} className="border-b border-ink py-6 last:border-b-0">
              <h3 className="type-h2">{d.title}</h3>
              <p className="measure mt-2 type-body-l">{d.text}</p>
              <p className="mt-2 type-body-s">Source: {d.source}</p>
            </li>
          ))}
        </ol>
      </div>
    </Band>
  );
}
