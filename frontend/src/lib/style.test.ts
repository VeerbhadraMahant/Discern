import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
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
  it("has no sans-serif or font-sans", () => {
    expect(scan(/sans-serif|font-sans/i)).toEqual([]);
  });
  it("has no radius above 12px", () => {
    expect(scan(/rounded-(md|lg|xl|2xl|3xl|full)\b/)).toEqual([]);
    const px = files.flatMap((f) => [...readFileSync(f, "utf8").matchAll(/border-radius:\s*(\d+(?:\.\d+)?)px/g)].map((m) => Number(m[1])));
    expect(px.every((n) => n <= 12)).toBe(true);
  });
  it("has no emoji", () => {
    expect(scan(/\p{Extended_Pictographic}/u)).toEqual([]);
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
