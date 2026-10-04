import { backendError } from "./errors";
import type {
  AnnotatedVideoResponse,
  AskResponse,
  CleanResponse,
  DetectResponse,
  DiscernClient,
  IngestEvent,
  InfoResponse,
  MediaKind,
  OkResponse,
  Plan,
  SceneProfile,
  Shot,
  TraceEvent,
  TraceResponse,
  UploadResponse,
} from "./types";

/**
 * Demo backend with canned responses. Everything it returns is placeholder data and the UI must
 * mark it as such (see the "Demo data" stamp). Never present these values as measurements.
 */

function placeholder(label: string, dark: boolean): string {
  const bg = dark ? "#3a3835" : "#cdc6be";
  const fg = dark ? "#cdc6be" : "#1d1d1b";
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">` +
    `<rect width="640" height="360" fill="${bg}"/>` +
    `<rect x="60" y="190" width="150" height="80" fill="none" stroke="${fg}" stroke-width="3"/>` +
    `<rect x="300" y="150" width="110" height="60" fill="none" stroke="${fg}" stroke-width="3"/>` +
    `<rect x="470" y="210" width="80" height="50" fill="none" stroke="${fg}" stroke-width="3"/>` +
    `<text x="320" y="60" font-family="Georgia, serif" font-size="28" fill="${fg}" text-anchor="middle">${label}</text>` +
    `<text x="320" y="95" font-family="Georgia, serif" font-size="16" fill="${fg}" text-anchor="middle">Demo placeholder</text>` +
    `</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

const PROFILE: SceneProfile = {
  scene_label: "street at dusk",
  illumination: "low",
  visibility: "hazy",
  object_scale: "small",
  object_density: "medium",
  confidence: 0.8,
};

const PLAN: Plan = {
  restorer: "demo restorer",
  use_restored: true,
  sr_factor: 2,
  decisions: [
    "Perception: low light and haze noticed, so restoration is considered.",
    "Restorer: demo restorer chosen because visibility is hazy.",
    "Image selection: the restored frame is kept over the original.",
    "Super-resolution: 2x because objects are small.",
  ],
};

function event(node: string, decision: string, rationale: string, fallback = false): TraceEvent {
  return {
    node,
    started_at: Date.now() / 1000,
    duration_ms: 120,
    gpu_ms: null,
    prompt_version: null,
    input_summary: "demo input",
    decision,
    rationale,
    fallback_used: fallback,
  };
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export interface MockOptions {
  /** Artificial latency per call in ms. Tests pass 0. */
  delayMs?: number;
}

export class MockDiscernClient implements DiscernClient {
  readonly isMock = true;
  readonly host = "demo data";
  private readonly delayMs: number;
  private readonly kinds = new Map<string, MediaKind>();
  private events: TraceEvent[] = [];
  private asks = 0;

  constructor(opts: MockOptions = {}) {
    this.delayMs = opts.delayMs ?? 500;
  }

  private async wait(factor = 1): Promise<void> {
    if (this.delayMs > 0) await sleep(this.delayMs * factor);
  }

  private record(...ev: TraceEvent[]): TraceEvent[] {
    this.events.push(...ev);
    return ev;
  }

  async info(): Promise<InfoResponse> {
    await this.wait(0.5);
    return {
      ok: true,
      profile: "demo",
      limits: { max_upload_mb: 50, max_video_seconds: 60, max_queries: 20, ttl_seconds: 3600, max_pixels: 8_000_000 },
      models: [
        { role: "detector", name: "Demo detector", license: "demo license" },
        { role: "restorer", name: "Demo restorer", license: "demo license (non-commercial)" },
      ],
      measured_gpu_seconds: null,
      memory_version: null,
    };
  }

  async upload(file: File): Promise<UploadResponse> {
    await this.wait();
    const kind: MediaKind = file.type.startsWith("video/") || /\.(mp4|mov|webm|mkv)$/i.test(file.name) ? "video" : "image";
    const session_id = `demo-${Math.random().toString(36).slice(2, 10)}`;
    this.kinds.set(session_id, kind);
    return { ok: true, session_id, kind, name: file.name, size_bytes: file.size };
  }

  async clean(): Promise<CleanResponse> {
    await this.wait(1.5);
    return {
      ok: true,
      original_url: placeholder("Before", true),
      cleaned_url: placeholder("After", false),
      view_url: placeholder("Clear view", false),
      detection_image: "original",
      profile: PROFILE,
      plan: PLAN,
      events: this.record(
        event("perception", "profile built", "Low light and haze found."),
        event("restorer_select", "demo restorer", "Visibility is hazy."),
        event("sr_select", "2x", "Objects are small.", true),
      ),
    };
  }

  async detect(sid: string, targets: string): Promise<DetectResponse> {
    await this.wait();
    if (!sid) throw backendError("Unknown session.");
    const labels = targets
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    const use = labels.length ? labels : ["car"];
    return {
      ok: true,
      annotated_url: placeholder("Annotated", false),
      detections: use.map((label, i) => ({
        label,
        score: 0.9 - i * 0.1,
        box: [60 + i * 40, 190, 210 + i * 40, 270],
        detector: "demo detector",
      })),
      events: this.record(event("detector_select", "demo detector", "Targets are common objects.")),
    };
  }

  async *ingest(sid: string): AsyncGenerator<IngestEvent, void, void> {
    const steps = ["Reading video", "Finding shots", "Profiling shots", "Cleaning frames"];
    for (let i = 0; i < steps.length; i++) {
      await this.wait(0.8);
      yield { ok: true, stage: "progress", progress: (i + 1) / (steps.length + 1), message: steps[i] ?? "Working" };
    }
    await this.wait(0.8);
    const shots: Shot[] = [
      { id: 0, t_start: 0, t_end: 5.5, profile: PROFILE, plan: PLAN, before_url: placeholder("Shot 1 before", true), after_url: placeholder("Shot 1 after", false) },
      { id: 1, t_start: 5.5, t_end: 12, profile: { ...PROFILE, illumination: "normal", visibility: "clear" }, plan: { ...PLAN, use_restored: false, sr_factor: null, decisions: ["Perception: scene is already clear.", "Restorer: skipped, the original is kept."] }, before_url: placeholder("Shot 2 before", true), after_url: placeholder("Shot 2 after", false) },
    ];
    yield {
      ok: true,
      stage: "done",
      duration: 12,
      fps: 15,
      width: 480,
      height: 270,
      shots,
      video_url: `${import.meta.env.BASE_URL}demo/demo.mp4`,
      events: this.record(event("ingest", `${shots.length} shots`, `Demo ingest for ${sid}.`)),
    };
  }

  async ask(sid: string, question: string): Promise<AskResponse> {
    await this.wait(1.5);
    this.asks += 1;
    if (!sid) throw backendError("Unknown session.");
    const vague = question.trim().split(/\s+/).length < 2;
    if (vague) {
      return {
        ok: true,
        answer: "Which object do you mean, and in which part of the video?",
        grounded: false,
        clarification: true,
        result_set_id: null,
        evidence: { tracks: [], summary: "" },
        events: this.record(event("ask", "clarify", "The question is too short to ground.")),
      };
    }
    const ungrounded = /why|feel|think/i.test(question);
    const id = `rs-demo-${this.asks}`;
    return {
      ok: true,
      answer: ungrounded
        ? "This is a demo answer that is not backed by tracked evidence."
        : "In this demo clip, two tracked objects match your question.",
      grounded: !ungrounded,
      clarification: false,
      result_set_id: ungrounded ? null : id,
      evidence: ungrounded
        ? { tracks: [], summary: "No tracks matched." }
        : {
            summary: "Two demo tracks were kept after verification.",
            tracks: [
              { id: 1, label: "car", t_start: 1.2, t_end: 4.8, n_frames: 54, mean_score: 0.91, status: "accepted", rationale: "Stable box over 54 frames, score above the gate.", crop_url: placeholder("car", false) },
              { id: 2, label: "car", t_start: 6.0, t_end: 7.4, n_frames: 21, mean_score: 0.42, status: "rejected", rationale: "Short and low score, likely a reflection.", crop_url: null },
            ],
          },
      events: this.record(event("ask", ungrounded ? "ungrounded" : "grounded", "Demo routing."), event("verify", "2 tracks", "Demo verification.", true)),
    };
  }

  async annotatedVideo(sid: string, resultSetId: string): Promise<AnnotatedVideoResponse> {
    await this.wait();
    if (!sid || !resultSetId) throw backendError("Unknown result set.");
    return { ok: true, video_url: `${import.meta.env.BASE_URL}demo/demo.mp4` };
  }

  async trace(): Promise<TraceResponse> {
    await this.wait(0.5);
    return { ok: true, events: [...this.events] };
  }

  async feedback(): Promise<OkResponse> {
    await this.wait(0.5);
    return { ok: true };
  }

  async cleanup(sid: string): Promise<OkResponse> {
    await this.wait(0.3);
    this.kinds.delete(sid);
    this.events = [];
    return { ok: true };
  }
}
