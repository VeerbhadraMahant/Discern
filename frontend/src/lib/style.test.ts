import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve, sep } from "node:path";
import { describe, expect, it } from "vitest";

const SRC = resolve(__dirname, "..");

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

const files = walk(SRC).filter((f) => /\.(tsx?|css)$/.test(f) && !/\.test\./.test(f));

function scan(re: RegExp): string[] {
  return files.filter((f) => re.test(readFileSync(f, "utf8"))).map((f) => f.replace(SRC, "src"));
}

function classLists(): string[] {
  return files.flatMap((f) =>
    [...readFileSync(f, "utf8").matchAll(/className=(?:"([^"]*)"|\{`([^`]*)`\})/g)].map((m) => m[1] ?? m[2] ?? ""),
  );
}

describe("design guardrails (MASTER.md)", () => {
  it("scans real files", () => {
    expect(files.length).toBeGreaterThan(10);
  });
  it("has no gradients", () => {
    expect(scan(/gradient/i)).toEqual([]);
  });
  it("has no blur or backdrop filters outside the hero resolve keyframes in index.css", () => {
    expect(scan(/backdrop-/i)).toEqual([]);
    expect(scan(/blur/i)).toEqual(["src/index.css"].map((f) => f.replace(/\//g, sep)));
    const css = readFileSync(resolve(SRC, "index.css"), "utf8");
    const withBlur = css.split(String.fromCharCode(10)).filter((l) => /blur/i.test(l)).map((l) => l.trimEnd());
    expect(withBlur).toEqual(["    from { filter: blur(6px); }"]);
  });
  it("includes the landing sources in the scan", () => {
    expect(files.some((f) => /landing[\\/]facts\.ts$/.test(f))).toBe(true);
    expect(files.some((f) => /landing[\\/]sections3\.tsx$/.test(f))).toBe(true);
  });
  it("does not use the retired typefaces", () => {
    expect(scan(/Bodoni|Source Serif|["']Inter["']|font-body/)).toEqual([]);
  });
  it("names font families only in the index.css tokens, never in components", () => {
    const tsx = files.filter((f) => /\.tsx?$/.test(f));
    const bad = tsx.filter((f) => /IBM Plex|Bricolage|font-sans|sans-serif|font-\[/.test(readFileSync(f, "utf8")));
    expect(bad.map((f) => f.replace(SRC, "src"))).toEqual([]);
  });
  it("has no all-caps text transforms or all-caps headings", () => {
    expect(scan(/\buppercase\b|text-transform:\s*uppercase|\bcapitalize\b/)).toEqual([]);
    expect(scan(/title="[A-Z]{3,}"/)).toEqual([]);
  });
  it("uses type tokens, with no raw pixel text sizes and nothing below 12px or lighter than 400", () => {
    expect(scan(/text-\[\d/)).toEqual([]);
    expect(scan(/\btext-(xs|sm|base|lg|xl|[2-9]xl)\b/)).toEqual([]);
    const sizes = files.flatMap((f) => {
      const src = readFileSync(f, "utf8");
      const px = [...src.matchAll(/font-size:\s*(\d+(?:\.\d+)?)px/g)].map((m) => Number(m[1]));
      const rem = [...src.matchAll(/font-size:\s*(\d+(?:\.\d+)?)rem/g)].map((m) => Number(m[1]) * 16);
      const svg = [...src.matchAll(/fontSize:\s*(\d+(?:\.\d+)?)/g)].map((m) => Number(m[1]));
      const tokens = [...src.matchAll(/--text-[a-z-]+:\s*(\d+(?:\.\d+)?)rem/g)].map((m) => Number(m[1]) * 16);
      return [...px, ...rem, ...svg, ...tokens];
    });
    expect(sizes.length).toBeGreaterThan(5);
    expect(Math.min(...sizes)).toBeGreaterThanOrEqual(12);
    expect(scan(/\bfont-(thin|extralight|light)\b|font-weight:\s*[123]00\s*;/)).toEqual([]);
  });
  it("keeps the display face for the landing page and the wordmark only", () => {
    const users = files.filter((f) => /type-display-|type-wordmark|font-display/.test(readFileSync(f, "utf8")) && !/index\.css$/.test(f));
    expect(users.map((f) => f.replace(SRC, "src").split(sep).join("/")).sort()).toEqual(
      ["src/components/Layout.tsx", "src/landing/art.tsx", "src/landing/kit.tsx", "src/landing/sections1.tsx", "src/landing/sections3.tsx"].sort(),
    );
  });
  it("has no radius above 12px", () => {
    expect(scan(/rounded-(md|lg|xl|2xl|3xl|full)\b/)).toEqual([]);
    const px = files.flatMap((f) => [...readFileSync(f, "utf8").matchAll(/border-radius:\s*(\d+(?:\.\d+)?)px/g)].map((m) => Number(m[1])));
    expect(px.every((n) => n <= 12)).toBe(true);
  });
  it("has no emoji", () => {
    expect(scan(/\p{Extended_Pictographic}/u)).toEqual([]);
  });
  it("has no middle dots or arrows in the UI copy", () => {
    expect(scan(/[·•→←➜]/)).toEqual([]);
  });
  it("has no em dashes", () => {
    expect(scan(new RegExp(String.fromCharCode(0x2014)))).toEqual([]);
  });
  it("never combines charcoal text with a bone surface in one class list", () => {
    expect(classLists().filter((c) => /bg-bone/.test(c) && /text-charcoal/.test(c))).toEqual([]);
  });
  it("never uses ember as a fill, because light text on it is below 4.5:1", () => {
    expect(classLists().filter((c) => /bg-ember/.test(c))).toEqual([]);
  });
});
