import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import type { Limits, Track } from "../api/types";
import { DropZone, validateFile } from "./DropZone";
import { TrackCard } from "./TrackCard";
import { VideoPlayer } from "./VideoPlayer";
import type { PlayerHandle } from "./VideoPlayer";

const LIMITS: Limits = { max_upload_mb: 1, max_video_seconds: 30, max_queries: 5, ttl_seconds: 600, max_pixels: 1000 };

describe("validateFile", () => {
  it("accepts images and videos under the limit", () => {
    expect(validateFile(new File(["x"], "a.png", { type: "image/png" }), LIMITS)).toBeNull();
    expect(validateFile(new File(["x"], "a.mp4", { type: "video/mp4" }), LIMITS)).toBeNull();
  });
  it("rejects other types and oversize files with a helpful message", () => {
    expect(validateFile(new File(["x"], "a.pdf", { type: "application/pdf" }), LIMITS)).toMatch(/not an image or a video/);
    const big = new File([new Uint8Array(2 * 1024 * 1024)], "big.png", { type: "image/png" });
    expect(validateFile(big, LIMITS)).toMatch(/over the 1 MB limit/);
  });
});

describe("DropZone", () => {
  it("shows limits, passes valid files and blocks invalid ones with an alert", async () => {
    const onFile = vi.fn();
    render(<DropZone limits={LIMITS} onFile={onFile} />);
    expect(screen.getByText(/Drop an image or video, or choose a file/)).toBeInTheDocument();
    expect(screen.getByText(/up to 1 MB/)).toBeInTheDocument();
    const input = screen.getByLabelText(/Media file/);
    const user = userEvent.setup({ applyAccept: false });

    await user.upload(input, new File(["x"], "notes.txt", { type: "text/plain" }));
    expect(onFile).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent(/notes.txt/);

    const good = new File(["x"], "ok.png", { type: "image/png" });
    await user.upload(input, good);
    expect(onFile).toHaveBeenCalledWith(good);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

const TRACK: Track = {
  id: 7,
  label: "car",
  t_start: 2.5,
  t_end: 4,
  n_frames: 10,
  mean_score: 0.8,
  status: "accepted",
  rationale: "Stable box.",
  crop_url: null,
};

describe("TrackCard", () => {
  it("shows label, range, score and status and jumps to the start time", async () => {
    const onJump = vi.fn();
    render(
      <ul>
        <TrackCard track={TRACK} onJump={onJump} />
      </ul>,
    );
    expect(screen.getByText(/00:02.5 to 00:04.0/)).toBeInTheDocument();
    expect(screen.getByText(/mean score 0.80/)).toBeInTheDocument();
    expect(screen.getByText("accepted")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Jump to time/ }));
    expect(onJump).toHaveBeenCalledWith(2.5);
  });
});

describe("VideoPlayer", () => {
  it("seeks the video element through its handle", () => {
    const ref = createRef<PlayerHandle>();
    render(
      <VideoPlayer ref={ref} originalUrl="/a.mp4" annotatedUrl={null} canAnnotate={false} loadingAnnotated={false} onRequestAnnotated={() => {}} />,
    );
    const el = document.querySelector("video") as HTMLVideoElement;
    let t = 0;
    Object.defineProperty(el, "currentTime", { get: () => t, set: (v: number) => (t = v), configurable: true });
    ref.current?.seek(3.2);
    expect(t).toBe(3.2);
  });

  it("requests the annotated video and switches to it", async () => {
    const onRequest = vi.fn();
    const { rerender } = render(
      <VideoPlayer originalUrl="/a.mp4" annotatedUrl={null} canAnnotate loadingAnnotated={false} onRequestAnnotated={onRequest} />,
    );
    await userEvent.click(screen.getByRole("button", { name: "Show annotated video" }));
    expect(onRequest).toHaveBeenCalled();
    rerender(<VideoPlayer originalUrl="/a.mp4" annotatedUrl="/b.mp4" canAnnotate loadingAnnotated={false} onRequestAnnotated={onRequest} />);
    expect(screen.getByLabelText("Annotated video")).toHaveAttribute("src", "/b.mp4");
  });
});
