import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { Shell } from "./App";
import { renderApp } from "./test/helpers";

function go(hash: string) {
  act(() => {
    location.hash = hash;
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
}

describe("hash routing", () => {
  beforeEach(() => {
    location.hash = "";
  });

  it("opens the tab named by the hash and updates on hashchange", async () => {
    location.hash = "#/trace";
    renderApp(<Shell />, { session: null });
    expect(screen.getByRole("heading", { level: 1, name: "TRACE" })).toBeInTheDocument();
    go("#/about");
    expect(await screen.findByRole("heading", { level: 1, name: "ABOUT" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /About/ })).toHaveAttribute("aria-current", "page");
  });

  it("navigates with the tab links and defaults to Clean", async () => {
    renderApp(<Shell />, { session: null });
    expect(screen.getByRole("heading", { level: 1, name: "CLEAN" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("link", { name: /Ask/ }));
    expect(await screen.findByRole("heading", { level: 1, name: "ASK" })).toBeInTheDocument();
    expect(location.hash).toBe("#/ask");
  });

  it("shows a skip link, the connection status and the demo stamp for the mock client", async () => {
    renderApp(<Shell />, { session: null });
    expect(screen.getByRole("link", { name: "Skip to main content" })).toBeInTheDocument();
    expect(await screen.findByText(/Connected to demo data/)).toBeInTheDocument();
    expect(screen.getByText("Demo data")).toBeInTheDocument();
  });

  it("offers Start over once a session exists", async () => {
    renderApp(<Shell />);
    await userEvent.click(screen.getByRole("button", { name: /Start over/ }));
    expect(await screen.findByText("Your session was deleted.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Start over/ })).not.toBeInTheDocument();
  });
});
