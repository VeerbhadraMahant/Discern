import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MockDiscernClient } from "../api/mock";
import type { InfoResponse } from "../api/types";
import { renderApp } from "../test/helpers";
import { AboutView } from "./AboutView";
import { AskView } from "./AskView";
import { CleanView } from "./CleanView";
import { FeedbackView } from "./FeedbackView";
import { TraceView } from "./TraceView";

async function ask(text: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Your question"), text);
  await user.click(screen.getByRole("button", { name: "Send" }));
  return user;
}

describe("AskView chat flow", () => {
  it("fills the box from a starter, then shows the grounded answer and evidence", async () => {
    renderApp(<AskView />);
    await userEvent.click(screen.getByRole("button", { name: "How many cars are there?" }));
    expect(screen.getByLabelText("Your question")).toHaveValue("How many cars are there?");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect((await screen.findAllByText(/two tracked objects match/)).length).toBeGreaterThan(0);
    expect(screen.getByText("accepted")).toBeInTheDocument();
    expect(screen.getByText("rejected")).toBeInTheDocument();
    expect(screen.queryByText("Not grounded in detections")).not.toBeInTheDocument();
    // The live region carries the newest answer.
    const live = document.querySelector('[aria-live="polite"]');
    expect(live).toHaveTextContent(/two tracked objects match/);
    expect(screen.getByLabelText("Your question")).toHaveValue("");
  });

  it("disables Send while waiting and shows a typing state", async () => {
    const client = new MockDiscernClient({ delayMs: 40 });
    renderApp(<AskView />, { client });
    await ask("How many cars are there?");
    const send = screen.getByRole("button", { name: /Working/ });
    expect(send).toBeDisabled();
    expect(screen.getByRole("status", { name: "Discern is answering" })).toBeInTheDocument();
    await screen.findAllByText(/two tracked objects match/);
  });

  it("labels ungrounded answers in text", async () => {
    renderApp(<AskView />);
    await ask("why is it dark");
    expect(await screen.findByText("Not grounded in detections")).toBeInTheDocument();
  });

  it("styles a clarification as a question to the user", async () => {
    renderApp(<AskView />);
    await ask("cars");
    expect(await screen.findByText("Discern needs one more detail")).toBeInTheDocument();
    expect(screen.queryByText("Not grounded in detections")).not.toBeInTheDocument();
  });

  it("uploads a video, then Jump to time on an evidence card seeks the player", async () => {
    let t = 0;
    Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
      configurable: true,
      get: () => t,
      set: (v: number) => {
        t = v;
      },
    });
    try {
      renderApp(
        <>
          <CleanView />
          <AskView />
        </>,
        { session: null },
      );
      const input = screen.getByLabelText(/Media file/);
      await waitFor(() => expect(input).toBeEnabled());
      await userEvent.setup({ applyAccept: false }).upload(input, new File(["x"], "clip.mp4", { type: "video/mp4" }));
      await screen.findByLabelText("Original video");
      await ask("How many cars are there?");
      const jump = (await screen.findAllByRole("button", { name: /Jump to time/ }))[0]!;
      await userEvent.click(jump);
      expect(t).toBe(1.2);
    } finally {
      Reflect.deleteProperty(HTMLMediaElement.prototype, "currentTime");
    }
  });

  it("shows an empty state without a session", () => {
    renderApp(<AskView />, { session: null });
    expect(screen.getByText("Nothing to ask about yet")).toBeInTheDocument();
  });
});

describe("TraceView", () => {
  it("shows an empty state, then events with expandable rows and fallback icon plus word", async () => {
    renderApp(
      <>
        <AskView />
        <TraceView />
      </>,
    );
    expect(screen.getByText("No events yet")).toBeInTheDocument();
    await ask("How many cars are there?");
    await screen.findAllByText(/two tracked objects match/);
    const table = await screen.findByRole("table");
    expect(within(table).getByText("fallback")).toBeInTheDocument();
    const first = within(table).getAllByRole("button")[0]!;
    expect(first).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(first);
    expect(first).toHaveAttribute("aria-expanded", "true");
    expect(within(table).getByText("Rationale")).toBeInTheDocument();
  });
});

describe("FeedbackView", () => {
  async function withResult(client = new MockDiscernClient({ delayMs: 0 })) {
    renderApp(
      <>
        <AskView />
        <FeedbackView />
      </>,
      { client },
    );
    await ask("How many cars are there?");
    await screen.findAllByText(/two tracked objects match/);
    return client;
  }

  it("asks for a result first", () => {
    renderApp(<FeedbackView />, { session: null });
    expect(screen.getByText("No results to rate yet")).toBeInTheDocument();
  });

  it("keeps the retain checkbox unchecked and sends retain_media false by default", async () => {
    const client = new MockDiscernClient({ delayMs: 0 });
    const spy = vi.spyOn(client, "feedback");
    await withResult(client);
    const box = screen.getByLabelText(/Allow Discern to keep my media/);
    expect(box).not.toBeChecked();
    expect(screen.getByText(/deleted automatically after 1 hour/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("radio", { name: /Correct/ }));
    await userEvent.click(screen.getByRole("button", { name: "Send feedback" }));
    expect(await screen.findByText(/Your feedback was sent/)).toBeInTheDocument();
    expect(spy).toHaveBeenCalledWith(expect.objectContaining({ verdict: "correct", retain_media: false, track_id: null }));
  });

  it("sends retain_media true only after the box is ticked, and a chosen track id", async () => {
    const client = new MockDiscernClient({ delayMs: 0 });
    const spy = vi.spyOn(client, "feedback");
    await withResult(client);
    await userEvent.click(screen.getByLabelText(/Allow Discern to keep my media/));
    await userEvent.selectOptions(screen.getByLabelText(/Track \(optional\)/), "2");
    await userEvent.click(screen.getByRole("radio", { name: /Wrong/ }));
    await userEvent.click(screen.getByRole("button", { name: "Send feedback" }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy).toHaveBeenCalledWith(expect.objectContaining({ verdict: "wrong", retain_media: true, track_id: 2 }));
  });

  it("requires a verdict and shows backend errors inline", async () => {
    const client = new MockDiscernClient({ delayMs: 0 });
    await withResult(client);
    await userEvent.click(screen.getByRole("button", { name: "Send feedback" }));
    expect(await screen.findByText(/Choose a verdict/)).toBeInTheDocument();
    vi.spyOn(client, "feedback").mockRejectedValueOnce(new Error("offline"));
    await userEvent.click(screen.getByRole("radio", { name: /Correct/ }));
    await userEvent.click(screen.getByRole("button", { name: "Send feedback" }));
    expect(await screen.findByText(/Could not reach the Discern app/)).toBeInTheDocument();
  });
});

describe("AboutView", () => {
  it("shows not measured yet and none for null values, and licenses as given", async () => {
    renderApp(<AboutView />);
    expect(await screen.findByText("not measured yet")).toBeInTheDocument();
    expect(screen.getByText("none")).toBeInTheDocument();
    expect(screen.getByText("demo license (non-commercial)")).toBeInTheDocument();
    expect(screen.getByText(/kept for 1 hour/)).toBeInTheDocument();
  });

  it("shows measured values when the API returns them", async () => {
    const client = new MockDiscernClient({ delayMs: 0 });
    const base: InfoResponse = await client.info();
    vi.spyOn(client, "info").mockResolvedValue({ ...base, measured_gpu_seconds: 12.5, memory_version: "v3" });
    renderApp(<AboutView />, { client });
    expect(await screen.findByText("12.5")).toBeInTheDocument();
    expect(screen.getByText("v3")).toBeInTheDocument();
    expect(screen.queryByText("not measured yet")).not.toBeInTheDocument();
  });
});
