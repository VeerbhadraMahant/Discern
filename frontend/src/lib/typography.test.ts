import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(__dirname, "../index.css"), "utf8");
const pub = resolve(__dirname, "../../public/fonts");

function decl(name: string): string {
  const m = new RegExp(`${name}:\\s*([^;]+);`).exec(css);
  if (!m || !m[1]) throw new Error(`token ${name} missing`);
  return m[1].trim();
}

describe("docs/typography.md tokens", () => {
  it("defines the three font-family tokens with the exact fallback stacks", () => {
    expect(decl("--font-display")).toBe('"Bricolage Grotesque", "Arial Narrow", "Roboto Condensed", system-ui, sans-serif');
    expect(decl("--font-sans")).toBe('"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif');
    expect(decl("--font-mono")).toBe('"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace');
  });

  it("defines the type scale: size, line height, tracking and weight per token", () => {
    const scale: Record<string, [string, string, string, string]> = {
      "display-xl": ["clamp(2.5rem, 1.796rem + 3.005vw, 4.5rem)", "1", "-0.01em", "700"],
      "display-l": ["clamp(2.25rem, 1.986rem + 1.127vw, 3rem)", "1.05", "0", "700"],
      h1: ["1.5rem", "1.25", "0", "600"],
      h2: ["1.3125rem", "1.3", "0", "600"],
      h3: ["1.125rem", "1.35", "0", "600"],
      "body-l": ["1.125rem", "1.6", "0", "400"],
      body: ["1rem", "1.55", "0", "400"],
      "body-s": ["0.875rem", "1.45", "0", "400"],
      label: ["0.875rem", "1.3", "0", "500"],
      caption: ["0.75rem", "1.4", "0.01em", "400"],
      "data-narrow": ["0.875rem", "1.3", "0", "400"],
      code: ["0.875rem", "1.5", "0", "400"],
    };
    for (const [name, [size, lh, ls, fw]] of Object.entries(scale)) {
      expect(decl(`--text-${name}`)).toBe(size);
      expect(decl(`--text-${name}--line-height`)).toBe(lh);
      expect(decl(`--text-${name}--letter-spacing`)).toBe(ls);
      expect(decl(`--text-${name}--font-weight`)).toBe(fw);
    }
  });

  it("maps each token to the right family", () => {
    const family = (token: string) => new RegExp(`@utility type-${token} \\{\\s*font-family: var\\(--font-(\\w+)\\)`).exec(css)?.[1];
    expect(family("display-xl")).toBe("display");
    expect(family("display-l")).toBe("display");
    expect(family("h1")).toBe("sans");
    expect(family("body")).toBe("sans");
    expect(family("data-narrow")).toBe("sans");
    expect(family("code")).toBe("mono");
  });

  it("self-hosts the font files and keeps the landing payload under 100 KB", () => {
    const files = {
      display: "bricolage-grotesque-82-700-latin.woff2",
      sans: "ibm-plex-sans-latin-wdth-normal.woff2",
      mono: "ibm-plex-mono-latin-400-normal.woff2",
    };
    for (const f of Object.values(files)) expect(existsSync(resolve(pub, f))).toBe(true);
    expect(existsSync(resolve(pub, "LICENSES.md"))).toBe(true);
    const size = (f: string) => statSync(resolve(pub, f)).size;
    expect(size(files.display) + size(files.sans)).toBeLessThan(100 * 1024);
    expect(css).not.toMatch(/fonts\.googleapis|fonts\.gstatic/);
    expect(readFileSync(resolve(__dirname, "../../index.html"), "utf8")).not.toMatch(/fonts\.googleapis|fonts\.gstatic/);
  });

  it("declares font-display swap on every face and a metric-adjusted display fallback", () => {
    const faces = css.match(/@font-face \{[^}]*\}/g) ?? [];
    expect(faces.length).toBe(4);
    const real = faces.filter((f) => /src:\s*url/.test(f));
    expect(real.length).toBe(3);
    for (const f of real) expect(f).toMatch(/font-display:\s*swap/);
    const fallback = faces.find((f) => /font-family:\s*"Arial Narrow"/.test(f)) ?? "";
    expect(fallback).toMatch(/size-adjust/);
    expect(fallback).toMatch(/ascent-override/);
    expect(fallback).toMatch(/descent-override/);
  });
});
