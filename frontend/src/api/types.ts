/** TypeScript mirror of frontend/API.md. */

export interface TraceEvent {
  node: string;
  started_at: number | string;
  duration_ms: number;
  gpu_ms: number | null;
  prompt_version: string | null;
  input_summary: string;
  decision: string;
  rationale: string;
  fallback_used: boolean;
}

export interface Limits {
  max_upload_mb: number;
  max_video_seconds: number;
  max_queries: number;
  ttl_seconds: number;
  max_pixels: number;
}

export interface ModelInfo {
  role: string;
  name: string;
  license: string;
}

export interface InfoResponse {
  ok: true;
  profile: string;
  limits: Limits;
  models: ModelInfo[];
  measured_gpu_seconds: number | null;
  memory_version: string | null;
}

export type MediaKind = "image" | "video";

export interface UploadResponse {
  ok: true;
  session_id: string;
  kind: MediaKind;
  name: string;
  size_bytes: number;
}

export interface SceneProfile {
  scene_label: string;
  illumination: string;
  visibility: string;
  object_scale: string;
  object_density: string;
  confidence: number;
}

export interface Plan {
  restorer: string;
  use_restored: boolean;
  sr_factor: number | null;
  decisions: string[];
}

export interface CleanResponse {
  ok: true;
  original_url: string;
  cleaned_url: string;
  /** Haze-free picture for display; the same as cleaned_url when there is no haze. */
  view_url: string;
  /** Which image detection runs on. */
  detection_image: "original" | "restored";
  profile: SceneProfile;
  plan: Plan;
  events: TraceEvent[];
}

export interface Detection {
  label: string;
  score: number;
  box: [number, number, number, number];
  detector: string;
}

export interface DetectResponse {
  ok: true;
  annotated_url: string;
  detections: Detection[];
  events: TraceEvent[];
}

export interface Shot {
  id: number | string;
  t_start: number;
  t_end: number;
  profile: SceneProfile;
  plan: Plan;
  before_url: string;
  after_url: string;
}

export interface IngestProgress {
  ok: true;
  stage: "progress";
  progress: number;
  message: string;
}

export interface IngestDone {
  ok: true;
  stage: "done";
  duration: number;
  fps: number;
  width: number;
  height: number;
  shots: Shot[];
  video_url: string;
  events: TraceEvent[];
}

export type IngestEvent = IngestProgress | IngestDone;

export type TrackStatus = "accepted" | "rejected";

export interface Track {
  id: number;
  label: string;
  t_start: number;
  t_end: number;
  n_frames: number;
  mean_score: number;
  status: TrackStatus;
  rationale: string;
  crop_url: string | null;
}

export interface Evidence {
  tracks: Track[];
  summary: string;
}

export interface AskResponse {
  ok: true;
  answer: string;
  grounded: boolean;
  clarification: boolean;
  result_set_id: string | null;
  evidence: Evidence;
  events: TraceEvent[];
}

export interface AnnotatedVideoResponse {
  ok: true;
  video_url: string;
}

export interface TraceResponse {
  ok: true;
  events: TraceEvent[];
}

export type Verdict = "correct" | "wrong" | "missing";

export interface FeedbackRequest {
  session_id: string;
  result_set_id: string;
  track_id: number | null;
  verdict: Verdict;
  note: string;
  retain_media: boolean;
}

export interface OkResponse {
  ok: true;
}

export interface ErrorResponse {
  ok: false;
  error: string;
}

/** The contract every backend (Gradio or mock) implements. */
export interface DiscernClient {
  /** True only for the built-in demo client. */
  readonly isMock: boolean;
  /** Human-readable host, for the connection status. */
  readonly host: string;
  info(): Promise<InfoResponse>;
  upload(file: File): Promise<UploadResponse>;
  clean(sid: string): Promise<CleanResponse>;
  detect(sid: string, targets: string): Promise<DetectResponse>;
  ingest(sid: string): AsyncGenerator<IngestEvent, void, void>;
  ask(sid: string, question: string): Promise<AskResponse>;
  annotatedVideo(sid: string, resultSetId: string): Promise<AnnotatedVideoResponse>;
  trace(sid: string): Promise<TraceResponse>;
  feedback(req: FeedbackRequest): Promise<OkResponse>;
  cleanup(sid: string): Promise<OkResponse>;
}
