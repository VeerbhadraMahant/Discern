import { CheckCircle } from "@phosphor-icons/react";
import { useId, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { toDiscernError } from "../api/errors";
import type { DiscernError } from "../api/errors";
import type { Verdict } from "../api/types";
import { Banner } from "../components/Layout";
import { Button, Card, EmptyState, ErrorNotice, Field, inputClass } from "../components/ui";
import { formatTtl } from "../lib/format";
import { useDiscern } from "../state/DiscernContext";

const VERDICTS: Array<{ value: Verdict; label: string; help: string }> = [
  { value: "correct", label: "Correct", help: "The answer or track is right." },
  { value: "wrong", label: "Wrong", help: "The answer or track is wrong." },
  { value: "missing", label: "Missing", help: "Something that should be there is not." },
];

export function FeedbackView() {
  const { client, session, chat, info } = useDiscern();
  const uid = useId();
  const sets = useMemo(
    () =>
      chat.flatMap((t, i) => {
        const r = t.response;
        if (t.role !== "assistant" || !r || !r.result_set_id) return [];
        const q = chat.slice(0, i).reverse().find((x) => x.role === "user")?.text ?? "question";
        return [{ id: r.result_set_id, question: q, tracks: r.evidence.tracks }];
      }),
    [chat],
  );
  const [setId, setSetId] = useState<string>("");
  const [trackId, setTrackId] = useState<string>("");
  const [verdict, setVerdict] = useState<Verdict | "">("");
  const [note, setNote] = useState("");
  const [retain, setRetain] = useState(false);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<DiscernError | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const activeSetId = setId || sets[sets.length - 1]?.id || "";
  const active = sets.find((s) => s.id === activeSetId);

  if (!session || sets.length === 0) {
    return (
      <>
        <Banner title="Feedback" kicker="Tell us when an answer was right, wrong or missing something." />
        <div className="mx-auto max-w-[1440px] px-4 py-10 md:px-8">
          <EmptyState title="No results to rate yet">
            <p>Ask a question on the Ask tab. A grounded answer gives you a result set to give feedback on.</p>
            <p>
              <a href="#/ask" className="link">
                Go to Ask
              </a>
            </p>
          </EmptyState>
        </div>
      </>
    );
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!client || !verdict) {
      setFormError("Choose a verdict: correct, wrong or missing.");
      return;
    }
    setFormError(null);
    setError(null);
    setDone(false);
    setBusy(true);
    try {
      await client.feedback({
        session_id: session.id,
        result_set_id: activeSetId,
        track_id: trackId === "" ? null : Number(trackId),
        verdict,
        note: note.trim(),
        retain_media: retain,
      });
      setDone(true);
      setNote("");
    } catch (err) {
      setError(toDiscernError(err));
    } finally {
      setBusy(false);
    }
  };

  const ttl = info ? formatTtl(info.limits.ttl_seconds) : null;

  return (
    <>
      <Banner title="Feedback" kicker="Tell us when an answer was right, wrong or missing something." />
      <div className="mx-auto max-w-[1440px] px-4 py-10 md:px-8">
        <Card as="section" aria-labelledby="fb-h" className="max-w-2xl">
          <form onSubmit={(e) => void submit(e)} className="flex flex-col gap-6" noValidate>
            <h2 id="fb-h" className="type-h2">
              Your feedback
            </h2>
            <Field id={`${uid}-set`} label="Result set">
              <select id={`${uid}-set`} value={activeSetId} onChange={(e) => { setSetId(e.target.value); setTrackId(""); }} className={inputClass}>
                {sets.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.question}
                  </option>
                ))}
              </select>
            </Field>
            <Field id={`${uid}-track`} label="Track (optional)" helper="Leave on whole answer to rate everything, or pick one track.">
              <select id={`${uid}-track`} value={trackId} onChange={(e) => setTrackId(e.target.value)} className={inputClass} aria-describedby={`${uid}-track-help`}>
                <option value="">Whole answer</option>
                {active?.tracks.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label} #{t.id} ({t.status})
                  </option>
                ))}
              </select>
            </Field>
            <fieldset className="flex flex-col gap-1" aria-describedby={formError ? `${uid}-verr` : undefined}>
              <legend className="font-semibold">Verdict</legend>
              {VERDICTS.map((v) => (
                <label key={v.value} className="flex min-h-11 items-center gap-3 font-normal">
                  <input type="radio" name={`${uid}-verdict`} value={v.value} checked={verdict === v.value} onChange={() => setVerdict(v.value)} className="size-5 shrink-0 accent-ink" />
                  <span>
                    <span className="font-semibold">{v.label}</span>: {v.help}
                  </span>
                </label>
              ))}
              {formError && (
                <p id={`${uid}-verr`} role="alert" className="font-semibold">
                  {formError}
                </p>
              )}
            </fieldset>
            <Field id={`${uid}-note`} label="Note (optional)">
              <textarea id={`${uid}-note`} rows={3} value={note} onChange={(e) => setNote(e.target.value)} className={inputClass} />
            </Field>
            <div className="flex flex-col gap-1">
              <label className="flex items-start gap-3 font-semibold">
                <input
                  type="checkbox"
                  checked={retain}
                  onChange={(e) => setRetain(e.target.checked)}
                  aria-describedby={`${uid}-retain-help`}
                  className="mt-1 size-5 shrink-0 accent-ink"
                />
                Allow Discern to keep my media for improving the system
              </label>
              <p id={`${uid}-retain-help`} className="pl-8 type-body-s">
                {ttl ? `Without this, your files and results are deleted automatically after ${ttl}, or sooner when you press Start over.` : "Without this, your files and results are deleted automatically when your session expires."}{" "}
                With it, your uploaded media and this feedback are kept after that time and may be used to improve Discern.
              </p>
            </div>
            <div className="flex flex-col gap-3">
              <div>
                <Button type="submit" loading={busy} loadingLabel="Sending">
                  Send feedback
                </Button>
              </div>
              {done && (
                <p role="status" className="flex items-center gap-2 font-semibold">
                  <CheckCircle size={24} aria-hidden="true" />
                  Thank you. Your feedback was sent.
                </p>
              )}
              {error && <ErrorNotice error={error} onRetry={() => void submit({ preventDefault() {} } as FormEvent)} onDismiss={() => setError(null)} />}
            </div>
          </form>
        </Card>
      </div>
    </>
  );
}
