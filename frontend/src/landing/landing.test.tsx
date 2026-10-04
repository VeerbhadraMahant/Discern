import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Shell } from "../App";
import { renderApp } from "../test/helpers";
import { DATASETS, FAQ, METHODS, STATS, fmt } from "./facts";
import { AskDemo, Results, ResultsTable } from "./sections2";
import { CodeBlock, FaqSection } from "./sections3";
import { Hero } from "./sections1";

function go(hash: string) {
  act(() => {
    location.hash = hash;
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  });
}

describe("landing page routing and links", () => {
  beforeEach(() => {
    location.hash = "";
  });

  it("is the default route and has exactly one h1", async () => {
    renderApp(<Shell />, { session: null });
    expect(await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ })).toBeInTheDocument();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(document.title).toMatch(/^Discern \|/);
  });

  it("keeps headings sequential and landmarks present", async () => {
    renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    const levels = screen.getAllByRole("heading").map((h) => Number(h.tagName.slice(1)));
    levels.forEach((l, i) => {
      if (i > 0) expect(l - levels[i - 1]!).toBeLessThanOrEqual(1);
    });
    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "On this page" })).toBeInTheDocument();
  });

  it("still opens each app tab by hash, and returns home from the brand link", async () => {
    location.hash = "#/ask";
    renderApp(<Shell />, { session: null });
    expect(screen.getByRole("heading", { level: 1, name: "Ask" })).toBeInTheDocument();
    go("#/about");
    expect(await screen.findByRole("heading", { level: 1, name: "About" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute("href", "#/");
    expect(screen.getByRole("link", { name: "Discern" })).toHaveAttribute("href", "#/");
    go("#/");
    expect(await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ })).toBeInTheDocument();
  });

  it("points the call-to-action links at the app and at demo mode", async () => {
    renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    for (const link of screen.getAllByRole("link", { name: "Open the app" })) expect(link).toHaveAttribute("href", "#/clean");
    for (const link of screen.getAllByRole("link", { name: "Try with demo data" })) expect(link).toHaveAttribute("href", "?mock=1#/clean");
    expect(screen.getByRole("link", { name: /Repository on GitHub/ })).toHaveAttribute("href", "https://github.com/VeerbhadraMahant/Discern");
  });
});

describe("landing content comes from facts.ts", () => {
  it("renders every baseline number in the table exactly as facts.ts formats it", () => {
    render(<ResultsTable />);
    const table = screen.getByRole("table");
    for (const m of METHODS) {
      const row = within(table).getByRole("row", { name: new RegExp(m.name) });
      const cells = within(row).getAllByRole("cell");
      DATASETS.forEach((d, i) => expect(cells[i]!.textContent).toContain(fmt(m.scores[d.id])));
    }
  });

  it("renders every stat from facts.ts", async () => {
    renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    for (const s of STATS) expect(screen.getAllByText(s.value).length).toBeGreaterThan(0);
  });

  it("the bar chart has a table alternative, an aria summary and printed values", async () => {
    render(<Results />);
    const table = screen.getByRole("table");
    expect(table).toBeInTheDocument();
    const chart = screen.getByRole("group", { name: /Bar chart of F1 at IoU 0.5 on COCO/ });
    expect(chart.getAttribute("aria-label")).toContain(fmt(METHODS[0]!.scores.coco));
    expect(within(chart).getAllByText(fmt(0.66)).length).toBeGreaterThan(0);
    await userEvent.click(screen.getByRole("tab", { name: "DarkFace" }));
    expect(screen.getByRole("group", { name: /on DarkFace/ })).toBeInTheDocument();
  });

  it("labels every example with the not-a-recorded-result stamp", async () => {
    render(<AskDemo />);
    expect(screen.getByText("Example, not a recorded result")).toBeInTheDocument();
    const tabs = screen.getAllByRole("tab");
    await userEvent.click(tabs[0]!);
    tabs[0]!.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Count" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Example, not a recorded result")).toBeInTheDocument();
  });
});

describe("interactions", () => {
  it("toggles an FAQ answer with the keyboard", async () => {
    render(<FaqSection />);
    const first = screen.getByRole("button", { name: FAQ[0]!.q });
    expect(first).toHaveAttribute("aria-expanded", "false");
    first.focus();
    await userEvent.keyboard("{Enter}");
    expect(first).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(FAQ[0]!.a)).toBeVisible();
    await userEvent.keyboard(" ");
    expect(first).toHaveAttribute("aria-expanded", "false");
  });

  it("copies the commands, and says so plainly when the clipboard is unavailable", async () => {
    const writeText = vi.fn().mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error("denied"));
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    render(<CodeBlock code="npm ci" />);
    await userEvent.click(screen.getByRole("button", { name: /Copy commands/ }));
    expect(await screen.findByText("Copied")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Copy commands/ }));
    expect(await screen.findByText(/Copy failed/)).toBeInTheDocument();
  });
});

describe("motion", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });
  function stubReduced(reduced: boolean) {
    vi.stubGlobal("matchMedia", (q: string) => ({ matches: reduced && q.includes("reduce"), media: q, addEventListener() {}, removeEventListener() {} }));
  }

  it("resolves the hero headline once when motion is allowed, with real selectable text", () => {
    stubReduced(false);
    const { container } = render(<Hero />);
    const h1 = screen.getByRole("heading", { level: 1 });
    expect(h1).toHaveTextContent("Ask a video a question and see the evidence.");
    expect(h1.className).toContain("hero-resolve");
    expect(container.querySelector(".hero-grain")).not.toBeNull();
  });

  it("renders the hero sharp, with no animation classes, under reduced motion", () => {
    stubReduced(true);
    const { container } = render(<Hero />);
    expect(screen.getByRole("heading", { level: 1 }).className).not.toContain("hero-resolve");
    expect(container.querySelector(".hero-grain")).toBeNull();
  });

  it("has no scroll-reveal or entrance animation classes anywhere else on the page", async () => {
    stubReduced(false);
    const { container } = renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    expect(container.querySelector(".reveal, .reveal-in, .rise")).toBeNull();
  });
});
