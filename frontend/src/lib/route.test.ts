import { describe, expect, it } from "vitest";
import { parseHash } from "./route";

describe("parseHash", () => {
  it("maps each tab hash to a route", () => {
    expect(parseHash("#/ask")).toBe("ask");
    expect(parseHash("#/trace")).toBe("trace");
    expect(parseHash("#feedback")).toBe("feedback");
    expect(parseHash("#/about")).toBe("about");
  });
  it("opens the landing page for an empty hash, the root hash or an in-page anchor", () => {
    expect(parseHash("")).toBe("home");
    expect(parseHash("#")).toBe("home");
    expect(parseHash("#/")).toBe("home");
    expect(parseHash("#s-features")).toBe("home");
    expect(parseHash("#/nope")).toBe("home");
  });
  it("ignores a query string after the tab name", () => {
    expect(parseHash("#/clean?mock=1")).toBe("clean");
  });
});
