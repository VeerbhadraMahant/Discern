import { describe, expect, it } from "vitest";
import { MockDiscernClient } from "./mock";
import type {
  AskResponse,
  CleanResponse,
  DiscernClient,
  IngestDone,
  IngestEvent,
  InfoResponse,
  Plan,
  SceneProfile,
  TraceEvent,
} from "./types";

function keys(o: object): string[] {
  return Object.keys(o).sort();
}

const EVENT_KEYS = ["decision", "duration_ms", "fallback_used", "gpu_ms", "input_summary", "node", "prompt_version", "rationale", "started_at"];
const PROFILE_KEYS = ["confidence", "illumination", "object_density", "object_scale", "scene_label", "visibility"];
const PLAN_KEYS = ["decisions", "restorer", "sr_factor", "use_restored"];

function checkEvents(events: TraceEvent[]) {
  expect(events.length).toBeGreaterThan(0);
  for (const e of events) expect(keys(e)).toEqual(EVENT_KEYS);
}
function checkProfile(p: SceneProfile) {
  expect(keys(p)).toEqual(PROFILE_KEYS);
}
function checkPlan(p: Plan) {
  expect(keys(p)).toEqual(PLAN_KEYS);
}

describe("MockDiscernClient", () => {
  const mock: DiscernClient = new MockDiscernClient({ delayMs: 0 });

  it("is marked as a mock", () => {
    expect(mock.isMock).toBe(true);
  });

  it("info matches the contract and reports unmeasured values as null", async () => {
    const info: InfoResponse = await mock.info();
    expect(keys(info)).toEqual(["limits", "measured_gpu_seconds", "memory_version", "models", "ok", "profile"]);
    expect(keys(info.limits)).toEqual(["max_pixels", "max_queries", "max_upload_mb", "max_video_seconds", "ttl_seconds"]);
    expect(info.measured_gpu_seconds).toBeNull();
    expect(info.memory_version).toBeNull();
    for (const m of info.models) expect(keys(m)).toEqual(["license", "name", "role"]);
  });

  it("upload, clean and detect match the contract", async () => {
    const up = await mock.upload(new File(["x"], "a.png", { type: "image/png" }));
    expect(keys(up)).toEqual(["kind", "name", "ok", "session_id", "size_bytes"]);
    expect(up.kind).toBe("image");
    const clean: CleanResponse = await mock.clean(up.session_id);
    expect(keys(clean)).toEqual(["cleaned_url", "events", "ok", "original_url", "plan", "profile"]);
    checkProfile(clean.profile);
    checkPlan(clean.plan);
    checkEvents(clean.events);
    const det = await mock.detect(up.session_id, "car, person");
    expect(keys(det)).toEqual(["annotated_url", "detections", "events", "ok"]);
    expect(det.detections).toHaveLength(2);
    for (const d of det.detections) expect(keys(d)).toEqual(["box", "detector", "label", "score"]);
  });

  it("ingest yields progress then one done", async () => {
    const up = await mock.upload(new File(["x"], "a.mp4", { type: "video/mp4" }));
    expect(up.kind).toBe("video");
    const seen: IngestEvent[] = [];
    for await (const ev of mock.ingest(up.session_id)) seen.push(ev);
    const done = seen[seen.length - 1] as IngestDone;
    expect(seen.slice(0, -1).every((e) => e.stage === "progress")).toBe(true);
    expect(done.stage).toBe("done");
    expect(keys(done)).toEqual(["duration", "events", "fps", "height", "ok", "shots", "stage", "video_url", "width"]);
    for (const s of done.shots) {
      expect(keys(s)).toEqual(["after_url", "before_url", "id", "plan", "profile", "t_end", "t_start"]);
      checkProfile(s.profile);
      checkPlan(s.plan);
    }
  });

  it("ask, annotated video, trace, feedback and cleanup match the contract", async () => {
    const ask: AskResponse = await mock.ask("s", "How many cars are there?");
    expect(keys(ask)).toEqual(["answer", "clarification", "events", "evidence", "grounded", "ok", "result_set_id"]);
    expect(ask.grounded).toBe(true);
    for (const t of ask.evidence.tracks) {
      expect(keys(t)).toEqual(["crop_url", "id", "label", "mean_score", "n_frames", "rationale", "status", "t_end", "t_start"]);
    }
    const vague = await mock.ask("s", "cars");
    expect(vague.clarification).toBe(true);
    const ungrounded = await mock.ask("s", "why is it dark");
    expect(ungrounded.grounded).toBe(false);
    expect(keys(await mock.annotatedVideo("s", ask.result_set_id ?? "x"))).toEqual(["ok", "video_url"]);
    const tr = await mock.trace("s");
    expect(tr.ok).toBe(true);
    checkEvents(tr.events);
    expect(await mock.feedback({ session_id: "s", result_set_id: "r", track_id: null, verdict: "correct", note: "", retain_media: false })).toEqual({ ok: true });
    expect(await mock.cleanup("s")).toEqual({ ok: true });
  });
});
