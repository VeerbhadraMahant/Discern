import { Crosshair, Play, Sparkle } from "@phosphor-icons/react";
import { useId, useState } from "react";
import type { FormEvent } from "react";
import { BeforeAfter } from "../components/BeforeAfter";
import { ProfileList, Timeline } from "../components/Decisions";
import { DropZone } from "../components/DropZone";
import { Banner } from "../components/Layout";
import { Button, Card, ErrorNotice, Field, inputClass } from "../components/ui";
import { CLEAN_STEPS, DETECT_STEPS, WorkingPanel } from "../components/WorkingPanel";
import { formatBytes, formatRange, formatTime } from "../lib/format";
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
      <h2 id={`${id}-h`} className="type-h2">
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
      {busy && <WorkingPanel title="Detecting objects" steps={DETECT_STEPS} />}
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
                    confidence {Math.round(d.score * 100)}%, {d.detector}
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
      <h2 id="shots-h" className="type-h2">
        Shots
      </h2>
      <p className="num font-normal">
        {done.shots.length} {done.shots.length === 1 ? "shot" : "shots"}, {formatTime(done.duration)} long, {done.width} by {done.height} at {done.fps} fps
      </p>
      <ul className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {done.shots.map((s) => (
          <Card as="li" key={s.id} className="flex flex-col gap-4">
            <h3 className="type-h3">
              Shot {Number(s.id) + 1 || String(s.id)} <span className="num ident">{formatRange(s.t_start, s.t_end)}</span>
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
      <Banner title="Clean" kicker="Upload an image or a video and see how Discern cleans it, step by step." />
      <div className="page flex flex-col gap-10 py-10">
        <Card as="section" aria-labelledby="up-h" className="flex flex-col gap-4">
          <h2 id="up-h" className="type-h2">
            Your file
          </h2>
          <DropZone limits={info?.limits ?? null} disabled={working || !info} onFile={(f) => void uploadFile(f)} />
          {uploading && (
            <WorkingPanel title="Uploading and checking the file" steps={["Sending the file", "Checking type, size and length"]} />
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
              <WorkingPanel title="Cleaning the image" steps={CLEAN_STEPS} />
            )}
            {clean && phase !== "cleaning" && (
              <Card as="section" aria-labelledby="clean-h" className="flex flex-col gap-6">
                <h2 id="clean-h" className="type-h2">
                  Before and after
                </h2>
                <BeforeAfter beforeUrl={clean.original_url} afterUrl={clean.view_url} beforeAlt="Original image before cleaning" afterAlt="Clear view of the image after cleaning" />
                <p className="type-body-s">
                  Detection runs on the {clean.detection_image} image{clean.detection_image === "original" ? ", which scored better for finding objects" : ""}. The clear view is for you to look at.
                </p>
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
              <WorkingPanel
                title="Processing the video"
                steps={["Splitting into shots", "Cleaning each shot", "Detecting and tracking", "Building the index"]}
                message={`Step ${Math.max(1, ingest.step)}: ${ingest.message} (${Math.round(ingest.progress * 100)}%)`}
                progress={ingest.progress}
              />
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
