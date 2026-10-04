import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Shell } from "../App";
import { renderApp } from "../test/helpers";
import { DATASETS, FAQ, METHOD, METHODS, PILLARS, QUERY_TYPES, RUN_LOCALLY, SHEET, STATS, fmt } from "./facts";
import { ContactSheet } from "./bands1";
import { Product, Queries } from "./bands2";
import { Results, ResultsTable } from "./bands3";
import { CodeBlock, FaqSection } from "./bands4";
import { FocusPull, Hero, describeFocus } from "./hero";

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

  it("keeps headings sequential and landmarks present, with the wordmark in the header and the footer", async () => {
    renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    const levels = screen.getAllByRole("heading").map((h) => Number(h.tagName.slice(1)));
    levels.forEach((l, i) => {
      if (i > 0) expect(l - levels[i - 1]!).toBeLessThanOrEqual(1);
    });
    const banner = screen.getByRole("banner");
    expect(within(banner).getByRole("link", { name: "Discern" })).toHaveAttribute("href", "#/");
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(within(screen.getByRole("contentinfo")).getByRole("link", { name: "Discern" })).toBeInTheDocument();
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

  it("renders every stat from facts.ts somewhere on the page", async () => {
    renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    for (const s of STATS) expect(screen.getAllByText(s.value, { exact: false }).length).toBeGreaterThan(0);
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

  it("states three findings above the table, with numbers computed from the data", () => {
    render(<Results />);
    expect(screen.getByText(/No single detector wins everywhere: YOLO-World v2 leads on COCO at 0.660, and OWLv2 base leads on HazyDet at 0.664/)).toBeInTheDocument();
    expect(screen.getByText(/DarkFace stays hard for every method: the best F1 there is 0.155/)).toBeInTheDocument();
    expect(screen.getByText(/the verifier passed 0.962 of answers, but only 0.136 of counts matched exactly/)).toBeInTheDocument();
  });

  it("merges question types and examples: pick a type, see question, answer, evidence kind and status", async () => {
    render(<Queries />);
    expect(screen.getByText("Example, not a recorded result")).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(QUERY_TYPES.length);
    expect(screen.getByText("Verified with real models")).toBeInTheDocument();
    screen.getAllByRole("tab")[0]!.focus();
    await userEvent.keyboard("{ArrowDown}{ArrowDown}");
    expect(screen.getByRole("tab", { name: "Temporal" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(QUERY_TYPES[2]!.question)).toBeInTheDocument();
    expect(screen.getByText("Tested on test doubles only")).toBeInTheDocument();
    expect(screen.getByText("Example, not a recorded result")).toBeInTheDocument();
  });

  it("writes the contact-sheet captions from the one data module, in order, with timecodes", () => {
    render(<ContactSheet />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(SHEET.length);
    SHEET.forEach((f, i) => {
      expect(items[i]).toHaveTextContent(f.caption);
      expect(items[i]).toHaveTextContent(f.time);
      expect(within(items[i]!).getByRole("heading", { level: 3, name: f.title })).toBeInTheDocument();
    });
  });

  it("has three pillars, each with its measured proof sentence", async () => {
    renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    for (const p of PILLARS) {
      expect(screen.getByRole("heading", { level: 3, name: p.title })).toBeInTheDocument();
      expect(screen.getByText(p.proof)).toBeInTheDocument();
    }
  });

  it("shows the product mock-up with the Demo data stamp, filled by the demo client", async () => {
    render(<Product />);
    expect(screen.getByText("Demo data")).toBeInTheDocument();
    expect(await screen.findByText(/two tracked objects match your question/)).toBeInTheDocument();
    expect(screen.getAllByText(/mean confidence/)).toHaveLength(2);
    await userEvent.click(screen.getAllByRole("button", { name: /Jump to time/ })[1]!);
    expect(screen.getByText(/Time 0:06/)).toBeInTheDocument();
  });
});

describe("methodology replaces the roadmap", () => {
  it("lists every principle from facts.ts with an evidence line and a source, and has no milestone wording", async () => {
    location.hash = "";
    renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    expect(screen.getByRole("heading", { level: 2, name: "How we build and judge it" })).toBeInTheDocument();
    expect(METHOD).toHaveLength(8);
    for (const p of METHOD) {
      expect(screen.getByRole("heading", { level: 3, name: new RegExp(p.title) })).toBeInTheDocument();
      expect(screen.getByText(p.text)).toBeInTheDocument();
      expect(screen.getByText((t) => t.includes(`Evidence: ${p.evidence}`))).toBeInTheDocument();
      expect(p.source.length).toBeGreaterThan(5);
    }
    expect(screen.getByRole("link", { name: "Methodology" })).toHaveAttribute("href", "#s-method");
    expect(screen.queryByRole("link", { name: /Roadmap/i })).toBeNull();
    expect(document.body.textContent).not.toMatch(/milestone|roadmap/i);
  });
});

describe("hero focus pull", () => {
  it("is a slider with aria values and descriptive text, labelled as an illustration", () => {
    render(<FocusPull />);
    const s = screen.getByRole("slider");
    expect(s).toHaveAttribute("aria-valuemin", "0");
    expect(s).toHaveAttribute("aria-valuemax", "100");
    expect(s).toHaveAttribute("aria-valuenow", "45");
    expect(s).toHaveAttribute("aria-valuetext", describeFocus(45));
    expect(s.getAttribute("aria-valuetext")).toMatch(/car, car/);
    expect(s.getAttribute("aria-valuetext")).not.toMatch(/person/);
    expect(screen.getByText("Illustration, not a recorded result")).toBeInTheDocument();
  });

  it("moves with the arrow keys, Shift, Home and End, clamps, and reveals the person's evidence", async () => {
    render(<FocusPull />);
    const s = screen.getByRole("slider");
    s.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(s).toHaveAttribute("aria-valuenow", "50");
    await userEvent.keyboard("{Shift>}{ArrowLeft}{/Shift}");
    expect(s).toHaveAttribute("aria-valuenow", "40");
    await userEvent.keyboard("{Home}");
    expect(s).toHaveAttribute("aria-valuenow", "0");
    await userEvent.keyboard("{ArrowLeft}");
    expect(s).toHaveAttribute("aria-valuenow", "0");
    expect(s.getAttribute("aria-valuetext")).toMatch(/person/);
    await userEvent.keyboard("{End}");
    expect(s).toHaveAttribute("aria-valuenow", "100");
    expect(s.getAttribute("aria-valuetext")).toContain("nothing yet");
  });
});

describe("one shared container", () => {
  it("uses the same page container in the landing header, every band and the footer", async () => {
    location.hash = "";
    const { container } = renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    expect(screen.getByRole("banner").firstElementChild).toHaveClass("page");
    const bands = [...container.querySelectorAll("main > section")];
    expect(bands.length).toBeGreaterThanOrEqual(11);
    for (const b of bands) expect(b.firstElementChild, b.id).toHaveClass("page");
    expect(screen.getByRole("contentinfo").firstElementChild).toHaveClass("page");
  });

  it("uses it in the app header, its nav and the page banner", () => {
    location.hash = "#/ask";
    renderApp(<Shell />, { session: null });
    const header = screen.getByRole("banner");
    expect(header.firstElementChild).toHaveClass("page");
    expect(within(header).getByRole("navigation", { name: "Sections" })).toHaveClass("page");
    expect(screen.getByRole("heading", { level: 1, name: "Ask" }).parentElement).toHaveClass("page");
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

  it("offers exactly three commands and copies them, saying so plainly when the clipboard is unavailable", async () => {
    expect(RUN_LOCALLY.split("\n")).toHaveLength(3);
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
    location.hash = "";
    const { container } = renderApp(<Shell />, { session: null });
    await screen.findByRole("heading", { level: 1, name: /Ask a video a question/ });
    expect(container.querySelector(".reveal, .reveal-in, .rise")).toBeNull();
  });
});
