import { Client, handle_file } from "@gradio/client";
import { backendError, invalidError, networkError, toDiscernError } from "./errors";
import type {
  AnnotatedVideoResponse,
  AskResponse,
  CleanResponse,
  DetectResponse,
  DiscernClient,
  FeedbackRequest,
  IngestEvent,
  InfoResponse,
  OkResponse,
  TraceResponse,
  UploadResponse,
} from "./types";

/** The slice of the Gradio client this wrapper needs; faked in tests. */
export interface GradioLike {
  predict(endpoint: string, payload?: Record<string, unknown>): Promise<{ data: unknown }>;
  submit(
    endpoint: string,
    payload?: Record<string, unknown>,
  ): AsyncIterable<{ type?: string; data?: unknown }>;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Unwrap a Gradio data array into the endpoint's JSON object, mapping `{ok:false}` to a typed error. */
export function unwrap<T>(data: unknown, endpoint: string): T {
  const first = Array.isArray(data) ? data[0] : data;
  if (!isRecord(first) || typeof first.ok !== "boolean") throw invalidError(endpoint);
  if (!first.ok) throw backendError(typeof first.error === "string" ? first.error : "");
  return first as T;
}

/**
 * The backend returns media as root-relative Gradio file URLs ("/gradio_api/file=..."). Resolve every
 * `*_url` string against the Gradio server so the browser fetches them from there, not from this app's origin.
 */
export function absolutizeMedia<T>(value: T, root: string): T {
  if (Array.isArray(value)) return value.map((v) => absolutizeMedia(v, root)) as unknown as T;
  if (!isRecord(value)) return value;
  const out: Record<string, unknown> = {};
  for (const [key, v] of Object.entries(value)) {
    out[key] =
      key.endsWith("_url") && typeof v === "string" && v.startsWith("/")
        ? new URL(v, root).toString()
        : absolutizeMedia(v, root);
  }
  return out as T;
}

export class GradioDiscernClient implements DiscernClient {
  readonly isMock = false;
  readonly host: string;
  private readonly app: GradioLike;
  private readonly root: string;

  constructor(app: GradioLike, host: string, root = "http://localhost/") {
    this.app = app;
    this.host = host;
    this.root = root;
  }

  static async connect(url: string): Promise<GradioDiscernClient> {
    try {
      const app = await Client.connect(url);
      return new GradioDiscernClient(app as unknown as GradioLike, new URL(url).host, new URL(url).origin);
    } catch (e) {
      throw networkError(e);
    }
  }

  private async call<T>(endpoint: string, payload: Record<string, unknown> = {}): Promise<T> {
    let res: { data: unknown };
    try {
      res = await this.app.predict(endpoint, payload);
    } catch (e) {
      throw toDiscernError(e);
    }
    return absolutizeMedia(unwrap<T>(res.data, endpoint), this.root);
  }

  info() {
    return this.call<InfoResponse>("/discern_info");
  }
  upload(file: File) {
    return this.call<UploadResponse>("/discern_upload", { file: handle_file(file) });
  }
  clean(sid: string) {
    return this.call<CleanResponse>("/discern_clean", { session_id: sid });
  }
  detect(sid: string, targets: string) {
    return this.call<DetectResponse>("/discern_detect", { session_id: sid, targets });
  }
  async *ingest(sid: string): AsyncGenerator<IngestEvent, void, void> {
    try {
      for await (const msg of this.app.submit("/discern_ingest", { session_id: sid })) {
        if (msg.type !== undefined && msg.type !== "data") continue;
        yield absolutizeMedia(unwrap<IngestEvent>(msg.data, "/discern_ingest"), this.root);
      }
    } catch (e) {
      throw toDiscernError(e);
    }
  }
  ask(sid: string, question: string) {
    return this.call<AskResponse>("/discern_ask", { session_id: sid, question });
  }
  annotatedVideo(sid: string, resultSetId: string) {
    return this.call<AnnotatedVideoResponse>("/discern_annotated_video", {
      session_id: sid,
      result_set_id: resultSetId,
    });
  }
  trace(sid: string) {
    return this.call<TraceResponse>("/discern_trace", { session_id: sid });
  }
  feedback(req: FeedbackRequest) {
    return this.call<OkResponse>("/discern_feedback", { ...req });
  }
  cleanup(sid: string) {
    return this.call<OkResponse>("/discern_cleanup", { session_id: sid });
  }
}
