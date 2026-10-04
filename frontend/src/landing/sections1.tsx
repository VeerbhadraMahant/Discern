import {
  ChatsCircle,
  Crosshair,
  Cpu,
  Database,
  FilmStrip,
  GitMerge,
  ListChecks,
  LockKey,
  PlayCircle,
  SealCheck,
  Sparkle,
  ThumbsUp,
  UsersThree,
} from "@phosphor-icons/react";
import { useState } from "react";
import type { ReactNode } from "react";
import { Card, Stamp } from "../components/ui";
import { Frame, FuseArt, StampSeal, TrackArt } from "./art";
import { ANNOUNCEMENT, DATELINE, FEATURES, NOT_DEPLOYED, PAPER_TITLE, PAPER_URL, STATS, STEPS } from "./facts";
import { prefersReducedMotion } from "./hooks";
import { Section } from "./kit";

export const APP_HREF = "#/clean";
export const DEMO_HREF = "?mock=1#/clean";

export function PrimaryLink({ href, children, onInk = false }: { href: string; children: ReactNode; onInk?: boolean }) {
  return (
    <a
      href={href}
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-tag px-5 py-2 type-label ${onInk ? "bg-parchment text-ink" : "bg-ink text-parchment"}`}
    >
      {children}
    </a>
  );
}

export function AnnouncementBar() {
  return (
    <div className="bg-ink text-parchment">
      <p className="mx-auto flex min-h-11 max-w-[1440px] flex-wrap items-center justify-center gap-x-3 px-4 py-1 text-center type-body-s md:px-8">
        <span className="font-semibold">{ANNOUNCEMENT}</span>
        <span>{NOT_DEPLOYED}</span>
      </p>
    </div>
  );
}

export function Hero() {
  // The headline resolves from a degraded look once. Skipped entirely under reduced motion; the text is readable either way.
  const [resolve] = useState(() => !prefersReducedMotion());
  return (
    <section aria-label="Introduction" id="top" tabIndex={-1}>
      <p className="mx-auto max-w-[1440px] px-4 pt-6 type-body-s md:px-8">{DATELINE}</p>
      <div className="mt-4 bg-ink text-parchment">
        <div className="mx-auto max-w-[1440px] px-4 py-10 md:px-8 md:py-14">
          <div className="relative max-w-[1100px]">
            <h1 className={`type-display-xl ${resolve ? "hero-resolve" : ""}`}>Ask a video a question and see the evidence.</h1>
            {resolve && <span aria-hidden="true" className="hero-grain" />}
          </div>
          <p className="measure mt-6 type-body-l">
            Discern is a degradation-aware, grounded video query agent built on open models. Upload a clip, let it clean each shot, then ask in plain language. Answers point to boxes, tracks and timestamps you can inspect.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3">
            <PrimaryLink href={APP_HREF} onInk>
              Open the app
            </PrimaryLink>
            <a href={DEMO_HREF} className="link">
              <PlayCircle size={20} aria-hidden="true" />
              Try with demo data
            </a>
          </div>
        </div>
      </div>
      <div className="mx-auto max-w-[1440px] px-4 py-12 md:px-8 md:py-16">
        <figure className="max-w-[960px]">
          <div className="relative bg-bone p-4 shadow-card md:p-6">
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <div className="border border-ink">
                  <Frame clean={false} />
                </div>
                <p className="mt-2 type-label">Before: a degraded input frame</p>
              </div>
              <div>
                <div className="border border-ink">
                  <Frame clean />
                </div>
                <p className="mt-2 type-label">After: cleaned, with grounded boxes</p>
              </div>
            </div>
            <StampSeal className="absolute -right-2 -top-8 h-20 w-20 md:-right-6 md:-top-10 md:h-28 md:w-28" />
          </div>
          <figcaption className="mt-3 type-body-s">
            This illustration is drawn in SVG. It is not a dataset image or a recorded result. Demo data in the app is canned and is not a measurement.
          </figcaption>
        </figure>
      </div>
    </section>
  );
}

export function Strip() {
  return (
    <Section id="s-overview" title="Detect, track, then show your work">
      <div className="grid gap-6 lg:grid-cols-[1fr_1.35fr_1fr] lg:items-start">
        <figure>
          <div className="border border-ink">
            <FuseArt />
          </div>
          <figcaption className="mt-2 type-body-s">Fusion: several detectors, one box.</figcaption>
        </figure>
        <div className="measure type-body-l">
          <p>
            Discern is a research prototype. You upload a video or an image, it cleans each shot adaptively, and you ask questions in natural language, follow-ups included. Answers about finding, counting, timing or relating objects point to boxes, tracks and timestamps you can inspect.
          </p>
          <p className="mt-4">
            It builds on{" "}
            <a href={PAPER_URL} target="_blank" rel="noreferrer noopener" className="underline decoration-1 underline-offset-[3px]">
              {PAPER_TITLE}
            </a>
            : self-adaptive restoration, multi-expertise detection and experience harvesting, extended here to video and question answering.
          </p>
        </div>
        <figure>
          <div className="border border-ink">
            <TrackArt />
          </div>
          <figcaption className="mt-2 type-body-s">Tracking: boxes over time become tracks.</figcaption>
        </figure>
      </div>
    </Section>
  );
}

export function StatBand() {
  return (
    <Section id="s-numbers" title="Measured here, not borrowed" tone="ink" lead="Every figure comes from this repository's own runs. Some of them are not flattering.">
      <ul className="grid grid-cols-1 gap-x-10 gap-y-8 md:grid-cols-2 lg:grid-cols-3">
        {STATS.map((s) => (
          <li key={s.id} className="border-l border-parchment pl-5">
            <p className="type-body-l">
              {s.before} <strong className="num font-semibold">{s.value}</strong> {s.after}
            </p>
            <p className="mt-2 type-body-s">Source: {s.source}</p>
          </li>
        ))}
      </ul>
    </Section>
  );
}

const FEATURE_ICONS: Record<string, ReactNode> = {
  clean: <Sparkle size={24} aria-hidden="true" />,
  fusion: <GitMerge size={24} aria-hidden="true" />,
  evidence: <Crosshair size={24} aria-hidden="true" />,
  verified: <SealCheck size={24} aria-hidden="true" />,
  followups: <ChatsCircle size={24} aria-hidden="true" />,
  index: <FilmStrip size={24} aria-hidden="true" />,
  tracking: <UsersThree size={24} aria-hidden="true" />,
  memory: <Database size={24} aria-hidden="true" />,
  feedback: <ThumbsUp size={24} aria-hidden="true" />,
  privacy: <LockKey size={24} aria-hidden="true" />,
  local: <Cpu size={24} aria-hidden="true" />,
  traces: <ListChecks size={24} aria-hidden="true" />,
};

export function Features() {
  return (
    <Section id="s-features" title="What it does, and where it stops" lead="Each card says one thing the code does, with its limit where we measured one.">
      <ul className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map((f) => (
          <li key={f.id}>
            <Card as="article" className="h-full">
              <span className="flex h-11 w-11 items-center justify-center border border-ink bg-parchment">{FEATURE_ICONS[f.id]}</span>
              <h3 className="mt-4 type-h3">{f.title}</h3>
              <p className="mt-2 type-body">{f.text}</p>
            </Card>
          </li>
        ))}
      </ul>
    </Section>
  );
}

export function HowItWorks() {
  return (
    <Section id="s-how" title="Five steps from upload to answer">
      <ol className="grid gap-8 lg:grid-cols-5 lg:gap-6">
        {STEPS.map((s, i) => (
          <li key={s.id} className="relative pl-16 lg:pl-0">
            <span className="absolute left-0 top-0 flex h-11 w-11 items-center justify-center border border-ink bg-bone type-h2 lg:static lg:mb-4">
              <span className="num">{i + 1}</span>
              <span className="sr-only">Step </span>
            </span>
            {i < STEPS.length - 1 && (
              <>
                <span aria-hidden="true" className="absolute bottom-[-32px] left-[21px] top-12 w-px bg-ink lg:hidden" />
                <span aria-hidden="true" className="absolute left-12 right-[-24px] top-[21px] hidden h-px bg-ink lg:block" />
              </>
            )}
            <h3 className="type-h3">{s.title}</h3>
            <p className="mt-2 type-body">{s.text}</p>
            {s.paper && (
              <p className="mt-3 flex flex-wrap gap-2">
                {s.paper.map((p) => (
                  <Stamp key={p}>{p}</Stamp>
                ))}
              </p>
            )}
          </li>
        ))}
      </ol>
      <p className="measure mt-10 type-body-s">
        Stamps name the DetAS paper term behind a step: SAIR is self-adaptive restoration, MED is multi-expertise detection and SEEH is self-evolving experience harvesting.
      </p>
    </Section>
  );
}
