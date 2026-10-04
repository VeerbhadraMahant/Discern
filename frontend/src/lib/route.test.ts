import { describe, expect, it } from "vitest";
import { parseHash } from "./route";

describe("parseHash", () => {
  it("maps each tab hash to a route", () => {
    expect(parseHash("#/ask")).toBe("ask");
    expect(parseHash("#/trace")).toBe("trace");
    expect(parseHash("#feedback")).toBe("feedback");
    expect(parseHash("#/about")).toBe("about");
  });
  it("falls back to clean for empty or unknown hashes", () => {
    expect(parseHash("")).toBe("clean");
    expect(parseHash("#/nope")).toBe("clean");
  });
});
