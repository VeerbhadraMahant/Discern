import { describe, expect, it, vi } from "vitest";
import { GradioDiscernClient } from "./client";
import type { GradioLike } from "./client";
import { DiscernError } from "./errors";
import type { IngestEvent } from "./types";

function fake(overrides: Partial<GradioLike> = {}): GradioLike {
  return {
    predict: vi.fn(async () => ({ data: [{ ok: true }] })),
    submit: vi.fn(() => (async function* () {})()),
    ...overrides,
  };
}

describe("GradioDiscernClient", () => {
  it("maps a success response to the endpoint object and sends named parameters", async () => {
    const predict = vi.fn(async () => ({
      data: [{ ok: true, session_id: "s1", kind: "image", name: "a.png", size_bytes: 3 }],
    }));
    const client = new GradioDiscernClient(fake({ predict }), "host");
    const res = await client.clean("s1");
    expect(res).toMatchObject({ ok: true, session_id: "s1" });
    expect(predict).toHaveBeenCalledWith("/discern_clean", { session_id: "s1" });
    await client.detect("s1", "car, person");
    expect(predict).toHaveBeenLastCalledWith("/discern_detect", { session_id: "s1", targets: "car, person" });
  });

  it("sends feedback fields including retain_media", async () => {
    const predict = vi.fn(async () => ({ data: [{ ok: true }] }));
    const client = new GradioDiscernClient(fake({ predict }), "host");
    await client.feedback({ session_id: "s", result_set_id: "r", track_id: null, verdict: "wrong", note: "n", retain_media: false });
    expect(predict).toHaveBeenCalledWith("/discern_feedback", {
      session_id: "s",
      result_set_id: "r",
      track_id: null,
      verdict: "wrong",
      note: "n",
      retain_media: false,
    });
  });

  it("maps {ok:false} to a typed backend error with a message and retry hint", async () => {
    const predict = vi.fn(async () => ({ data: [{ ok: false, error: "Video too long." }] }));
    const client = new GradioDiscernClient(fake({ predict }), "host");
    const err = await client.ask("s", "q").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(DiscernError);
    expect(err).toMatchObject({ kind: "backend", message: "Video too long.", retryable: true });
    expect((err as DiscernError).hint).not.toBe("");
  });

  it("maps network failures to a typed network error", async () => {
    const predict = vi.fn(async () => {
      throw new Error("Failed to fetch");
    });
    const client = new GradioDiscernClient(fake({ predict }), "host");
    const err = await client.info().catch((e: unknown) => e);
    expect(err).toMatchObject({ kind: "network", retryable: true });
    expect((err as DiscernError).message).toContain("Could not reach");
  });

  it("rejects a malformed reply as an invalid error", async () => {
    const predict = vi.fn(async () => ({ data: ["nope"] }));
    const client = new GradioDiscernClient(fake({ predict }), "host");
    await expect(client.info()).rejects.toMatchObject({ kind: "invalid", retryable: false });
  });

  it("yields ingest progress in order and then done, skipping status messages", async () => {
    const submit = vi.fn(() =>
      (async function* () {
        yield { type: "status", data: undefined };
        yield { type: "data", data: [{ ok: true, stage: "progress", progress: 0.2, message: "a" }] };
        yield { type: "data", data: [{ ok: true, stage: "progress", progress: 0.6, message: "b" }] };
        yield {
          type: "data",
          data: [{ ok: true, stage: "done", duration: 1, fps: 1, width: 1, height: 1, shots: [], video_url: "u", events: [] }],
        };
      })(),
    );
    const client = new GradioDiscernClient(fake({ submit }), "host");
    const seen: IngestEvent[] = [];
    for await (const ev of client.ingest("s")) seen.push(ev);
    expect(seen.map((e) => e.stage)).toEqual(["progress", "progress", "done"]);
    expect(seen.map((e) => (e.stage === "progress" ? e.progress : 1))).toEqual([0.2, 0.6, 1]);
    expect(submit).toHaveBeenCalledWith("/discern_ingest", { session_id: "s" });
  });

  it("raises a backend error from inside the ingest stream", async () => {
    const submit = vi.fn(() =>
      (async function* () {
        yield { type: "data", data: [{ ok: false, error: "bad video" }] };
      })(),
    );
    const client = new GradioDiscernClient(fake({ submit }), "host");
    const run = async () => {
      for await (const ev of client.ingest("s")) void ev;
    };
    await expect(run()).rejects.toMatchObject({ kind: "backend", message: "bad video" });
  });
});
