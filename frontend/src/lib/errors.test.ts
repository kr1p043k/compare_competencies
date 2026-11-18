import { describe, expect, it, vi } from "vitest";

import { ApiError } from "./api";
import { errMessage, onError, reportError } from "./errors";

describe("reportError", () => {
  it("returns the message and logs it", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    const message = reportError("test", new Error("boom"));

    expect(message).toBe("boom");
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("uses ApiError message", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    expect(reportError("test", new ApiError("нет прав", 403, null, "/"))).toBe("нет прав");
  });

  it("notifies listeners only when user-visible", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    const listener = vi.fn();
    const off = onError(listener);

    reportError("test", new Error("silent"));
    expect(listener).not.toHaveBeenCalled();

    reportError("test", new Error("показать"), true);
    expect(listener).toHaveBeenCalledWith("показать");

    off();
    reportError("test", new Error("после off"), true);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("keeps working when a listener throws", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    const good = vi.fn();
    const offBad = onError(() => {
      throw new Error("listener broken");
    });
    const offGood = onError(good);

    expect(() => reportError("test", new Error("x"), true)).not.toThrow();
    expect(good).toHaveBeenCalledWith("x");

    offBad();
    offGood();
  });

  it("stringifies values that are not Error instances", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    expect(reportError("test", 42)).toBe("42");
  });

  it("survives a failing backend log", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));

    expect(() => reportError("test", new Error("x"))).not.toThrow();
    await Promise.resolve();
  });
});

describe("errMessage", () => {
  it("reads a message from Error and message-like objects", () => {
    expect(errMessage(new Error("boom"))).toBe("boom");
    expect(errMessage({ message: "obj" })).toBe("obj");
  });

  it("returns undefined for values without a string message", () => {
    expect(errMessage({ message: 42 })).toBeUndefined();
    expect(errMessage({})).toBeUndefined();
    expect(errMessage(null)).toBeUndefined();
    expect(errMessage("str")).toBeUndefined();
  });
});