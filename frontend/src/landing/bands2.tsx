import { ChatsCircle, CheckCircle, Cpu, FilmStrip, Flask, LockKey, MapPin, ThumbsUp, WarningCircle } from "@phosphor-icons/react";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { MockDiscernClient } from "../api/mock";
import type { AskResponse, CleanResponse, Track } from "../api/types";
import { ProfileList, Timeline } from "../components/Decisions";
import { TrackCard } from "../components/TrackCard";
import { Stamp } from "../components/ui";
import { WordmarkMark } from "../components/Wordmark";
import { formatTime } from "../lib/format";
import { FuseArt, TrackArt } from "./art";
import { FEATURES, QUERY_NOTE, QUERY_TYPES, STATS } from "./facts";
import { Band, Corners, Panel, Tabs } from "./kit";
import { Brackets, DEFAULT_PLACEMENT, Scene, boxes } from "./scene";

interface Demo {
  clean: CleanResponse;
  ask: AskResponse;
}

const DEMO_QUESTION = "How many cars are there?";

/** Demo data comes from the app's own mock client, so the mock-up shows exactly what Demo mode shows. */
function useDemo(): Demo | null {
  const [demo, setDemo] = useState<Demo | null>(null);
  useEffect(() => {
    let live = true;
    const client = new MockDiscernClient({ delayMs: 0 });
    void Promise.all([client.clean(), client.ask("demo", DEMO_QUESTION)]).then(([clean, ask]) => {
      if (live) setDemo({ clean, ask });
    });
    return () => {
      live = false;
    };
  }, []);
  return demo;
}

/** The video player is replaced by an SVG frame: the car moves along the road as the time changes. */
function FrameStandIn({ t, tracks }: { t: number; tracks: Track[] }) {
  const active = tracks.find((k) => t >= k.t_start && t <= k.t_end) ?? null;
  const p = { ...DEFAULT_PLACEMENT, carX: Math.round(360 + t * 20) };
  const car = boxes(p).car;
  return (
    <figure>
      <div role="img" aria-label={`Illustration standing in for the video player, at ${formatTime(t)}.${active ? ` Outlined: ${active.label} number ${active.id}.` : ""}`} className="relative border border-ink bg-ink">
        <Scene p={p}>{active && <Brackets {...car} />}</Scene>
        {active && (
          <span
            aria-hidden="true"
            className="absolute bg-ink px-1.5 py-0.5 type-caption font-medium text-parchment"
            style={{ left: `${(car.x / 720) * 100}%`, top: `${(car.y / 460) * 100}%`, transform: "translateY(-100%)" }}
          >
            {active.label} #{active.id}
          </span>
        )}
      </div>
      <figcaption className="num mt-2 type-body-s">Time {formatTime(t)}. Use Jump to time on a track to move it.</figcaption>
    </figure>
  );
}

/** A realistic composed screen from the real components (TrackCard, Timeline, the chat turn styles) with demo data. */
export function Product() {
  const demo = useDemo();
  const [t, setT] = useState(1.2);
  const tracks = demo?.ask.evidence.tracks ?? [];
  return (
    <Band
      id="s-product"
      tone="bone"
      title="The app, shown with demo data"
      lead="This screen is built from the app's own components and filled by its demo client. The video player is replaced by a drawing."
    >
      <div className="relative mx-2 border-2 border-ink bg-parchment shadow-card">
        <Corners />
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-ink px-4 py-2 md:px-6">
          <div className="flex items-center gap-2">
            <WordmarkMark size={20} />
            <span className="type-label">Ask</span>
          </div>
          <Stamp>Demo data</Stamp>
        </div>
        {!demo ? (
          <p role="status" className="flex min-h-[420px] items-center justify-center type-body">
            Loading demo data
          </p>
        ) : (
          <div className="grid gap-8 p-4 md:p-6 lg:grid-cols-3">
            <div className="flex flex-col gap-6">
              <p className="type-h3">Video</p>
              <FrameStandIn t={t} tracks={tracks} />
              <p className="type-h3">Scene profile</p>
              <ProfileList profile={demo.clean.profile} />
            </div>
            <div className="flex flex-col gap-4">
              <p className="type-h3">Conversation</p>
              <p className="border-l-2 border-ink pl-3">{DEMO_QUESTION}</p>
              <div className="flex flex-col gap-3 rounded-card bg-bone p-6 text-ink shadow-card">
                <p>{demo.ask.answer}</p>
                <p className="type-label flex items-center gap-1">
                  <CheckCircle size={20} aria-hidden="true" />
                  Evidence shown
                </p>
              </div>
              <Timeline plan={demo.clean.plan} />
            </div>
            <div className="flex flex-col gap-4">
              <p className="type-h3">Evidence</p>
              <p className="type-body-s">{demo.ask.evidence.summary}</p>
              <ul className="flex flex-col gap-4">
                {tracks.map((k) => (
                  <TrackCard key={k.id} track={k} onJump={setT} />
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </Band>
  );
}

const ICON = { size: 24, "aria-hidden": true } as const;

function Cell({ className = "", tone, children }: { className?: string; tone: "bone" | "ink" | "line"; children: ReactNode }) {
  if (tone === "line") return <div className={`border-2 border-ink p-6 ${className}`}>{children}</div>;
  return (
    <Panel tone={tone} className={className}>
      {children}
    </Panel>
  );
}

function feature(id: string) {
  return FEATURES.find((f) => f.id === id)!;
}

function Words({ id, tone = "bone" }: { id: string; tone?: "bone" | "ink" | "line" }) {
  const f = feature(id);
  void tone;
  return (
    <>
      <h3 className="type-h3">{f.title}</h3>
      <p className="mt-2 type-body">{f.text}</p>
    </>
  );
}

function TraceRows() {
  return (
    <ul aria-hidden="true" className="flex flex-col gap-2">
      {[
        ["perception", "w-1/3"],
        ["restorer", "w-1/4"],
        ["detect", "w-3/4"],
        ["verify", "w-1/2"],
      ].map(([name, w]) => (
        <li key={name} className="grid grid-cols-[5rem_1fr] items-center gap-3 type-body-s">
          <span>{name}</span>
          <span className="block h-3 border border-ink bg-parchment">
            <span className={`block h-full bg-ink ${w}`} />
          </span>
        </li>
      ))}
    </ul>
  );
}

/** The remaining capabilities, as a bento grid of deliberately different cells. */
export function Bento() {
  const gpu = STATS.find((s) => s.id === "gpu")!;
  return (
    <Band id="s-more" title="What else it does, and where it stops">
      <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-12">
        <Cell tone="bone" className="flex flex-col gap-5 md:col-span-2 lg:col-span-5 lg:row-span-2">
          <div className="border border-ink">
            <TrackArt />
          </div>
          <div>
            <Words id="tracking" />
          </div>
        </Cell>
        <Cell tone="ink" className="lg:col-span-3">
          <FilmStrip {...ICON} />
          <div className="mt-4">
            <Words id="index" />
          </div>
        </Cell>
        <Cell tone="line" className="lg:col-span-4">
          <ChatsCircle {...ICON} />
          <div className="mt-4">
            <Words id="followups" />
          </div>
        </Cell>
        <Cell tone="bone" className="grid gap-5 md:col-span-2 md:grid-cols-2 md:items-center lg:col-span-7">
          <div className="border border-ink">
            <FuseArt />
          </div>
          <div>
            <Words id="fusion" />
          </div>
        </Cell>
        <Cell tone="line" className="lg:col-span-3">
          <ThumbsUp {...ICON} />
          <div className="mt-4">
            <Words id="feedback" />
          </div>
        </Cell>
        <Cell tone="bone" className="md:col-span-2 lg:col-span-5">
          <TraceRows />
          <div className="mt-5">
            <Words id="traces" />
          </div>
        </Cell>
        <Cell tone="ink" className="lg:col-span-4">
          <LockKey {...ICON} />
          <div className="mt-4">
            <Words id="privacy" />
          </div>
        </Cell>
        <div className="flex flex-col gap-4 border-t-8 border-ink pt-6 md:col-span-2 md:flex-row md:items-start md:gap-8 lg:col-span-12">
          <Cpu {...ICON} className="mt-1 shrink-0" />
          <div className="max-w-[68ch]">
            <h3 className="type-h2">{feature("local").title}</h3>
            <p className="mt-2 type-body-l">{feature("local").text}</p>
            <p className="mt-2 type-body-l">
              {gpu.before} <strong className="num font-semibold">{gpu.value}</strong> {gpu.after}
            </p>
          </div>
        </div>
      </div>
    </Band>
  );
}

/** Question types and the example exchange, merged: pick a type, read a sample question, answer and the evidence kind. */
export function Queries() {
  return (
    <Band id="s-queries" tone="bone" title="Six kinds of question" lead={QUERY_NOTE}>
      <Tabs
        label="Question types"
        vertical
        tabs={QUERY_TYPES.map((q) => ({ id: q.id, label: q.name }))}
        render={(i) => {
          const q = QUERY_TYPES[i]!;
          return (
            <div className="flex flex-col gap-5">
              <p>
                <Stamp icon={<WarningCircle size={16} className="text-ember" aria-hidden="true" />}>Example, not a recorded result</Stamp>
              </p>
              <p className="border-l-2 border-ink pl-4 type-body-l">{q.question}</p>
              <Panel tone="parchment">
                <p className="type-body-l">{q.answer}</p>
              </Panel>
              <div>
                <h3 className="type-h3">Evidence</h3>
                <p className="mt-1 type-body">{q.evidence}</p>
                <ul className="mt-3 flex flex-col gap-2">
                  {q.lines.map((line) => (
                    <li key={line} className="num flex gap-2">
                      <MapPin size={20} className="mt-0.5 shrink-0" aria-hidden="true" />
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <p>
                {q.verified ? (
                  <Stamp icon={<CheckCircle size={16} aria-hidden="true" />}>Verified with real models</Stamp>
                ) : (
                  <Stamp icon={<Flask size={16} aria-hidden="true" />}>Tested on test doubles only</Stamp>
                )}
              </p>
            </div>
          );
        }}
      />
    </Band>
  );
}
