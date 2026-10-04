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
    expect(resolveConfig("?space=https://a.b", { VITE_DISCERN_URL: "http://c.d" }).url).toBe("https://a.b");
  });
  it("rejects a non-http ?space= value", () => {
    const c = resolveConfig("?space=javascript:alert(1)", {});
    expect(c.url).toBe(DEFAULT_URL);
    expect(c.badSpace).toBe("javascript:alert(1)");
  });
  it("lets ?mock=0 turn the env mock off and ?mock=1 turn it on", () => {
    expect(resolveConfig("?mock=0", { VITE_USE_MOCK: "1" }).mock).toBe(false);
    expect(resolveConfig("?mock=1", {}).mock).toBe(true);
  });
});
