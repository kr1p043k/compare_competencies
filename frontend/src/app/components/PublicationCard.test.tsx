import { describe, expect, it } from "vitest";

import { doiHref, errorText, isAbort } from "./PublicationCard";

describe("PublicationCard helpers", () => {
  it("doiHref prefixes bare DOIs and keeps URLs", () => {
    expect(doiHref("10.1000/xyz")).toBe("https://doi.org/10.1000/xyz");
    expect(doiHref("https://doi.org/10.1000/xyz")).toBe("https://doi.org/10.1000/xyz");
  });

  it("isAbort detects AbortError only", () => {
    expect(isAbort(new DOMException("x", "AbortError"))).toBe(true);
    expect(isAbort(new Error("boom"))).toBe(false);
    expect(isAbort("string")).toBe(false);
  });

  it("errorText maps rate-limit and network failures to Russian text", () => {
    expect(errorText(new Error("Too many requests"))).toContain("Слишком много");
    expect(errorText(new Error("Failed to fetch"))).toContain("связи с сервером");
  });

  it("errorText unwraps a JSON detail", () => {
    const err = new Error(JSON.stringify({ detail: "понятная причина" }));
    expect(errorText(err)).toBe("понятная причина");
  });

  it("errorText passes through unknown messages", () => {
    expect(errorText(new Error("что-то своё"))).toBe("что-то своё");
  });
});
