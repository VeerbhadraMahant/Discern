import { Crosshair, Play, Sparkle } from "@phosphor-icons/react";
import { useId, useState } from "react";
import type { FormEvent } from "react";
import { BeforeAfter } from "../components/BeforeAfter";
import { ProfileList, Timeline } from "../components/Decisions";
import { DropZone } from "../components/DropZone";
import { Banner } from "../components/Layout";
import { Button, Card, ErrorNotice, Field, ProgressBar, Skeleton, inputClass } from "../components/ui";
import { formatBytes, formatRange } from "../lib/format";
import { useDiscern } from "../state/DiscernContext";

function DetectPanel() {
  const { detect, runDetect, phase } = useDiscern();
  const id = useId();
  const [targets, setTargets] = useState("");
  const busy = phase === "detecting";
  const submit = (e: FormEvent) => {
    e.preventDefault();
    void runDetect(targets.trim());
  };
  return (
    <Card as="section" aria-labelledby={`${id}-h`} className="flex flex-col gap-4">
      <h2 id={`${id}-h`} className="font-display text-heading-sm leading-[0.95] tracking-[-0.04em]">
        Detect objects
      </h2>
      <form onSubmit={submit} className="flex flex-col gap-4">
        <Field id={id} label="Targets (optional)" helper="Names separated by commas, for example: car, person. Leave empty to let Discern choose.">
          <input id={id} value={targets} onChange={(e) => setTargets(e.target.value)} className={inputClass} aria-describedby={`${id}-help`} />
        </Field>
        <div>
          <Button type="submit" loading={busy} loadingLabel="Detecting" icon={<Crosshair size={20} aria-hidden="true" />}>
            Detect
          </Button>
        </div>
      </form>
      {busy && <Skeleton className="h-48 w-full" />}
      {detect && !busy && (
        <div className="flex flex-col gap-4">
          <img src={detect.annotated_url} alt={`Image with ${detect.detections.length} detection boxes drawn`} className="block w-full bg-ink" />
          {detect.detections.length === 0 ? (
            <p>No objects were found for those targets.</p>
          ) : (
            <ul className="flex flex-col divide-y divide-ink/40">
              {detect.detections.map((d, i) => (
                <li key={i} className="flex flex-wrap justify-between gap-2 py-2">
                  <span className="font-semibold">{d.label}</span>
                  <span className="num font-normal">
                    score {d.score.toFixed(2)}, {d.detector}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Card>
  );
}

function ShotCards() {
  const { ingest } = useDiscern();
  const done = ingest?.done;
  if (!done) return null;
  return (
    <section aria-labelledby="shots-h" className="flex flex-col gap-4">
      <h2 id="shots-h" className="font-display text-heading-sm leading-[0.95] tracking-[-0.04em]">
        Shots
      </h2>
      <p className="num font-normal">
        {done.shots.length} {done.shots.length === 1 ? "shot" : "shots"}, {done.duration.toFixed(1)} s, {done.width} by {done.height} at {done.fps} fps
      </p>
      <ul className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {done.shots.map((s) => (
          <Card as="li" key={s.id} className="flex flex-col gap-4">
            <h3 className="font-display text-subhead leading-none tracking-[-0.04em]">
              Shot {Number(s.id) + 1 || String(s.id)} <span className="num font-body text-base font-normal">{formatRange(s.t_start, s.t_end)}</span>
            </h3>
            <BeforeAfter
              beforeUrl={s.before_url}
              afterUrl={s.after_url}
              beforeAlt={`Shot ${s.id} before cleaning, ${s.profile.scene_label}`}
              afterAlt={`Shot ${s.id} after cleaning`}
            />
            <ProfileList profile={s.profile} />
            <Timeline plan={s.plan} label="Plan decisions" />
          </Card>
        ))}
      </ul>
    </section>
  );
}

export function CleanView() {
  const { session, info, uploadFile, phase, error, clearError, clean, ingest, runClean, runIngest } = useDiscern();
  const working = phase === "uploading" || phase === "cleaning" || phase === "ingesting";
  const uploading = phase === "uploading";

  return (
    <>
      <Banner title="CLEAN" kicker="Upload an image or a video and see how Discern cleans it, step by step." />
      <div className="mx-auto flex max-w-[1440px] flex-col gap-10 px-4 py-10 md:px-8">
        <Card as="section" aria-labelledby="up-h" className="flex flex-col gap-4">
          <h2 id="up-h" className="font-display text-heading-sm leading-[0.95] tracking-[-0.04em]">
            Your file
          </h2>
          <DropZone limits={info?.limits ?? null} disabled={working || !info} onFile={(f) => void uploadFile(f)} />
          {uploading && (
            <div className="flex flex-col gap-2" role="status">
              <p>Uploading and checking the file</p>
              <ProgressBar value={null} label="Upload in progress" />
            </div>
          )}
          {session && !uploading && (
            <p className="font-normal">
              Session file: <span className="font-semibold">{session.name || "unnamed"}</span> ({session.kind}
              {session.sizeBytes ? `, ${formatBytes(session.sizeBytes)}` : ""})
            </p>
          )}
          {error && <ErrorNotice error={error} onDismiss={clearError} onRetry={session ? (session.kind === "image" ? () => void runClean() : () => void runIngest()) : undefined} />}
        </Card>

        {session?.kind === "image" && (
          <>
            {phase === "cleaning" && (
              <Card className="flex flex-col gap-4" role="status">
                <p>Cleaning the image</p>
                <Skeleton className="h-56 w-full" />
              </Card>
            )}
            {clean && phase !== "cleaning" && (
              <Card as="section" aria-labelledby="clean-h" className="flex flex-col gap-6">
                <h2 id="clean-h" className="font-display text-heading-sm leading-[0.95] tracking-[-0.04em]">
                  Before and after
                </h2>
                <BeforeAfter beforeUrl={clean.original_url} afterUrl={clean.cleaned_url} beforeAlt="Original image before cleaning" afterAlt="Image after cleaning" />
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  <div className="flex flex-col gap-2">
                    <p className="font-semibold">Scene profile</p>
                    <ProfileList profile={clean.profile} />
                  </div>
                  <Timeline plan={clean.plan} />
                </div>
              </Card>
            )}
            {!clean && !working && (
              <div>
                <Button onClick={() => void runClean()} icon={<Sparkle size={20} aria-hidden="true" />}>
                  Clean this image
                </Button>
              </div>
            )}
            {clean && <DetectPanel />}
          </>
        )}

        {session?.kind === "video" && (
          <>
            {ingest && !ingest.done && phase === "ingesting" && (
              <Card as="section" className="flex flex-col gap-3" aria-label="Video progress">
                <p role="status" className="num font-semibold">
                  Step {Math.max(1, ingest.step)}: {ingest.message} ({Math.round(ingest.progress * 100)}%)
                </p>
                <ProgressBar value={ingest.progress} label="Video processing progress" />
                <Skeleton className="h-40 w-full" />
              </Card>
            )}
            {!ingest?.done && phase !== "ingesting" && !uploading && (
              <div>
                <Button onClick={() => void runIngest()} icon={<Play size={20} aria-hidden="true" />}>
                  Process this video
                </Button>
              </div>
            )}
            <ShotCards />
          </>
        )}
      </div>
    </>
  );
}
