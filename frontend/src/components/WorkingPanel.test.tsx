import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CLEAN_STEPS, WorkingPanel } from "./WorkingPanel";

describe("WorkingPanel", () => {
  it("lists every step and says the highlighted step is an estimate", () => {
    render(<WorkingPanel title="Cleaning the image" steps={CLEAN_STEPS} />);
    expect(screen.getByRole("status", { name: "Cleaning the image" })).toBeTruthy();
    for (const s of CLEAN_STEPS) expect(screen.getByText(s)).toBeTruthy();
    expect(screen.getByText(/estimate from elapsed time/)).toBeTruthy();
  });

  it("shows the server's real message and progress for video, without the estimate note", () => {
    render(<WorkingPanel title="Processing the video" steps={["a", "b"]} message="Step 2: Detecting (40%)" progress={0.4} />);
    expect(screen.getByText("Step 2: Detecting (40%)")).toBeTruthy();
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("40");
    expect(screen.queryByText(/estimate from elapsed time/)).toBeNull();
  });
});
