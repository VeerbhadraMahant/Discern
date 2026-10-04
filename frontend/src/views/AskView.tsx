import { PaperPlaneRight, Question, WarningCircle } from "@phosphor-icons/react";
import { useId, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { toDiscernError } from "../api/errors";
import type { DiscernError } from "../api/errors";
import { Banner } from "../components/Layout";
import { TrackCard } from "../components/TrackCard";
import { VideoPlayer } from "../components/VideoPlayer";
import type { PlayerHandle } from "../components/VideoPlayer";
import { Button, Card, EmptyState, ErrorNotice, inputClass } from "../components/ui";
import { ASK_STEPS, WorkingPanel } from "../components/WorkingPanel";
import { useDiscern } from "../state/DiscernContext";
import type { ChatTurn } from "../state/DiscernContext";

const STARTERS = ["How many cars are there?", "Is there a person in the video?", "When does something first appear?"];

function AssistantTurn({ turn, selected, onSelect }: { turn: ChatTurn; selected: boolean; onSelect: () => void }) {
  const r = turn.response;
  const clarification = r?.clarification ?? false;
  const ungrounded = r ? !r.grounded && !clarification : false;
  const hasEvidence = !!r && r.evidence.tracks.length > 0;
  return (
    <Card className={`rise flex flex-col gap-3 ${selected ? "outline outline-2 outline-ink" : ""}`}>
      {clarification && (
        <p className="type-label flex items-center gap-2">
          <Question size={20} aria-hidden="true" />
          Discern needs one more detail
        </p>
      )}
      <p className={clarification ? "type-body-l" : ""}>{turn.text}</p>
      <div className="flex flex-wrap items-center gap-4">
        {ungrounded && (
          <p className="type-label flex items-center gap-1">
            <WarningCircle size={20} className="text-ember" aria-hidden="true" />
            Not grounded in detections
          </p>
        )}
        {hasEvidence && (
          <Button variant="link" onClick={onSelect} aria-pressed={selected}>
            {selected ? "Evidence shown" : "Show evidence"}
          </Button>
        )}
      </div>
    </Card>
  );
}

export function AskView() {
  const { session, ingest, chat, ask, phase, error, clearError, info, client } = useDiscern();
  const [draft, setDraft] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [annotated, setAnnotated] = useState<Record<string, string>>({});
  const [annotatedLoading, setAnnotatedLoading] = useState(false);
  const [annotatedError, setAnnotatedError] = useState<DiscernError | null>(null);
  const player = useRef<PlayerHandle>(null);
  const inputId = useId();
  const asking = phase === "asking";
  const busy = phase !== "idle";

  const assistants = chat.filter((t) => t.role === "assistant");
  const withEvidence = assistants.filter((t) => t.response && t.response.evidence.tracks.length > 0);
  const shown = withEvidence.find((t) => t.id === selectedId) ?? withEvidence[withEvidence.length - 1] ?? null;
  const latestSet = [...assistants].reverse().find((t) => t.response?.result_set_id)?.response?.result_set_id ?? null;
  const asked = chat.filter((t) => t.role === "user").length;
  const maxQueries = info?.limits.max_queries ?? null;
  const limitReached = maxQueries !== null && asked >= maxQueries;
  const lastAnswer = assistants[assistants.length - 1]?.text ?? "";
  const videoUrl = ingest?.done?.video_url ?? null;

  const send = async (e?: FormEvent) => {
    e?.preventDefault();
    const q = draft.trim();
    if (!q || busy || limitReached) return;
    setDraft("");
    const ok = await ask(q);
    if (!ok) setDraft(q);
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send();
    }
  };

  const requestAnnotated = async () => {
    if (!client || !session || !latestSet) return;
    setAnnotatedLoading(true);
    setAnnotatedError(null);
    try {
      const res = await client.annotatedVideo(session.id, latestSet);
      setAnnotated((prev) => ({ ...prev, [latestSet]: res.video_url }));
    } catch (e) {
      setAnnotatedError(toDiscernError(e));
    } finally {
      setAnnotatedLoading(false);
    }
  };

  return (
    <>
      <Banner title="Ask" kicker="Ask a question about the video. Every answer shows the evidence it rests on." />
      <div className="page py-10">
        {!session ? (
          <EmptyState title="Nothing to ask about yet">
            <p>Upload a video on the Clean tab first.</p>
            <p>
              <a href="#/clean" className="link">
                Go to Clean
              </a>
            </p>
          </EmptyState>
        ) : (
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-3">
            <section aria-labelledby="media-h" className="flex flex-col gap-4">
              <h2 id="media-h" className="type-h2">
                Video
              </h2>
              {videoUrl ? (
                <VideoPlayer
                  ref={player}
                  originalUrl={videoUrl}
                  annotatedUrl={latestSet ? (annotated[latestSet] ?? null) : null}
                  canAnnotate={!!latestSet}
                  loadingAnnotated={annotatedLoading}
                  onRequestAnnotated={() => void requestAnnotated()}
                />
              ) : (
                <Card>
                  <p>
                    {session.kind === "image"
                      ? "This session holds an image, so there is no video to play."
                      : "The video is not loaded in this page. Process it again on the Clean tab to play it here."}
                  </p>
                </Card>
              )}
              {annotatedError && <ErrorNotice error={annotatedError} onRetry={() => void requestAnnotated()} onDismiss={() => setAnnotatedError(null)} />}
            </section>

            <section aria-labelledby="chat-h" className="flex flex-col gap-4">
              <h2 id="chat-h" className="type-h2">
                Conversation
              </h2>
              <div aria-live="polite" className="sr-only">
                {lastAnswer}
              </div>
              <ol className="flex flex-col gap-4">
                {chat.map((t) => (
                  <li key={t.id}>
                    {t.role === "user" ? (
                      <p className="border-l-2 border-ink pl-3">{t.text}</p>
                    ) : (
                      <AssistantTurn turn={t} selected={shown?.id === t.id} onSelect={() => setSelectedId(t.id)} />
                    )}
                  </li>
                ))}
                {asking && (
                  <li>
                    <WorkingPanel title="Discern is answering" steps={ASK_STEPS} />
                  </li>
                )}
              </ol>
              {chat.length === 0 && (
                <div className="flex flex-col gap-2">
                  <p className="font-semibold">Try a starter question, then edit it</p>
                  <ul className="flex flex-col items-start gap-1">
                    {STARTERS.map((s) => (
                      <li key={s}>
                        <Button variant="link" onClick={() => setDraft(s)}>
                          {s}
                        </Button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {error && phase === "idle" && <ErrorNotice error={error} onDismiss={clearError} />}
              <form onSubmit={(e) => void send(e)} className="flex flex-col gap-2">
                <label htmlFor={inputId} className="font-semibold">
                  Your question
                </label>
                <textarea
                  id={inputId}
                  rows={3}
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={onKey}
                  className={inputClass}
                  aria-describedby={`${inputId}-help`}
                />
                <p id={`${inputId}-help`} className="type-body-s">
                  Enter sends, Shift+Enter adds a new line.
                  {maxQueries !== null && ` Questions used: ${asked} of ${maxQueries}.`}
                </p>
                <div>
                  <Button
                    type="submit"
                    disabled={!draft.trim() || limitReached || busy}
                    loading={asking}
                    loadingLabel="Working"
                    icon={<PaperPlaneRight size={20} aria-hidden="true" />}
                  >
                    Send
                  </Button>
                </div>
                {limitReached && <p role="alert">You reached the question limit for this session. Start over to ask more.</p>}
              </form>
            </section>

            <section aria-labelledby="ev-h" className="flex flex-col gap-4">
              <h2 id="ev-h" className="type-h2">
                Evidence
              </h2>
              {shown?.response ? (
                <>
                  {shown.response.evidence.summary && <p>{shown.response.evidence.summary}</p>}
                  <ul className="flex flex-col gap-6">
                    {shown.response.evidence.tracks.map((tr) => (
                      <TrackCard key={tr.id} track={tr} onJump={(t) => player.current?.seek(t)} />
                    ))}
                  </ul>
                </>
              ) : (
                <p>Evidence tracks appear here after a grounded answer.</p>
              )}
            </section>
          </div>
        )}
      </div>
    </>
  );
}
