import { describe, expect, it } from "vitest";
import { DEFAULT_URL, resolveConfig } from "./config";

describe("resolveConfig", () => {
  it("defaults to the local Gradio URL", () => {
    expect(resolveConfig("", {})).toEqual({ url: DEFAULT_URL, mock: false, badSpace: null });
  });
  it("uses the env URL and mock flag", () => {
    const c = resolveConfig("", { VITE_DISCERN_URL: "https://x.hf.space", VITE_USE_MOCK: "1" });
    expect(c).toMatchObject({ url: "https://x.hf.space", mock: true });
  });
  it("lets ?space= override the env URL when it is http(s)", () => {
    expect(resolveConfig("?space=https://a.hf.space", { VITE_DISCERN_URL: "http://c.d" }).url).toBe("https://a.hf.space");
  });
  it("rejects a non-http ?space= value", () => {
    const c = resolveConfig("?space=javascript:alert(1)", {});
    expect(c.url).toBe(DEFAULT_URL);
    expect(c.badSpace).toBe("javascript:alert(1)");
  });
  it("ignores a ?space= host that is not a Space or local, so uploads cannot be redirected", () => {
    for (const bad of [
      "https://evil.example",
      "https://evilhf.space",
      "https://hf.space.evil.example",
      "https://hf.space",
      "http://a.hf.space",
      "https://user:pw@a.hf.space",
      "http://localhost.evil.example",
    ]) {
      const c = resolveConfig(`?space=${encodeURIComponent(bad)}`, {});
      expect(c.url, bad).toBe(DEFAULT_URL);
      expect(c.badSpace, bad).toBe(bad);
    }
  });
  it("accepts a local server in ?space=", () => {
    expect(resolveConfig("?space=http://localhost:7861", {}).url).toBe("http://localhost:7861");
  });
  it("lets ?mock=0 turn the env mock off and ?mock=1 turn it on", () => {
    expect(resolveConfig("?mock=0", { VITE_USE_MOCK: "1" }).mock).toBe(false);
    expect(resolveConfig("?mock=1", {}).mock).toBe(true);
  });
});
