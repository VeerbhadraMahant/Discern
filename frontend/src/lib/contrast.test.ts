import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { contrastRatio } from "./contrast";

const css = readFileSync(resolve(__dirname, "../index.css"), "utf8");
function token(name: string): string {
  const m = new RegExp(`--color-${name}:\\s*(#[0-9a-fA-F]{6})`).exec(css);
  if (!m || !m[1]) throw new Error(`token ${name} missing`);
  return m[1];
}

/** Text and background pairs actually used by the components. */
const TEXT_PAIRS: Array<[string, string, string]> = [
  ["ink", "parchment", "body text on the page"],
  ["ink", "bone", "text inside cards, inputs and expanded rows"],
  ["parchment", "ink", "banner and primary button"],
  // Landing page pairs: stat band, announcement bar and run-it-locally band use parchment on ink;
  // cards, code block, chart card and stack chips use ink on bone; body, tables, chart values and stamps use ink on parchment.
  ["parchment", "ink", "landing stat band, announcement bar and run-it-locally band"],
  ["ink", "bone", "landing feature cards, code block, chart card and stack chips"],
  ["ink", "parchment", "landing body text, tables, chart values and stamps"],
  // Redesigned landing: ink bands (contact sheet, results, final call to action) carry parchment text and the results table;
  // the pull-quote, product, questions, models and FAQ bands sit on bone; panels alternate bone, parchment and ink.
  ["parchment", "ink", "contact sheet captions, results findings and results table on the ink bands"],
  ["bone", "ink", "secondary text drawn in bone on an ink band"],
  ["ink", "bone", "pull-quote, product, questions, models and FAQ bands, and the honesty panel"],
  ["ink", "parchment", "panels on bone bands (mock-up window, question answer, privacy rules)"],
  ["parchment", "ink", "ink panels on parchment bands (pillar checked numbers, bento cells)"],
];

describe("palette contrast", () => {
  it("is computed from the real tokens", () => {
    expect(contrastRatio(token("ink"), token("parchment"))).toBeGreaterThan(12);
  });

  it.each(TEXT_PAIRS)("%s on %s reaches 4.5:1 (%s)", (fg, bg) => {
    expect(contrastRatio(token(fg), token(bg))).toBeGreaterThanOrEqual(4.5);
  });

  it("documents the pairs that must not carry normal-size text", () => {
    // These fail 4.5:1, so the UI keeps them to borders, icons and rules.
    expect(contrastRatio(token("charcoal"), token("parchment"))).toBeLessThan(4.5);
    expect(contrastRatio(token("charcoal"), token("bone"))).toBeLessThan(4.5);
    expect(contrastRatio(token("ember"), token("parchment"))).toBeLessThan(4.5);
    expect(contrastRatio(token("parchment"), token("ember"))).toBeLessThan(4.5);
  });

  it("keeps chart and illustration strokes at 3:1 on every landing surface", () => {
    expect(contrastRatio(token("ink"), token("parchment"))).toBeGreaterThanOrEqual(3);
    expect(contrastRatio(token("ink"), token("bone"))).toBeGreaterThanOrEqual(3);
    expect(contrastRatio(token("parchment"), token("ink"))).toBeGreaterThanOrEqual(3);
  });

  it("keeps ember at 3:1 or better for borders, icons and focus rings", () => {
    expect(contrastRatio(token("ember"), token("parchment"))).toBeGreaterThanOrEqual(3);
    expect(contrastRatio(token("ember"), token("ink"))).toBeGreaterThanOrEqual(3);
  });

  it("keeps the ember evidence mark visible (3:1) on every surface it is drawn on, and never as text", () => {
    for (const bg of ["parchment", "bone", "ink"]) expect(contrastRatio(token("ember"), token(bg))).toBeGreaterThanOrEqual(3);
  });

  it("keeps the dividers and hatch strokes (ink and parchment) at 3:1 on the new band surfaces", () => {
    expect(contrastRatio(token("ink"), token("bone"))).toBeGreaterThanOrEqual(3);
    expect(contrastRatio(token("parchment"), token("ink"))).toBeGreaterThanOrEqual(3);
    expect(contrastRatio(token("bone"), token("ink"))).toBeGreaterThanOrEqual(3);
  });
});
