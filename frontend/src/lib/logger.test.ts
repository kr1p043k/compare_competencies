import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const { handleUnauthorized } = vi.hoisted(() => ({ handleUnauthorized: vi.fn() }));

vi.mock("./auth", () => ({ handleUnauthorized }));

import { initApiLogger } from "./logger";

const fetchMock = vi.fn();
const originalFetch = globalThis.fetch;

beforeAll(() => {
  vi.stubGlobal("fetch", fetchMock);
  initApiLogger();
});

afterAll(() => {
  globalThis.fetch = originalFetch;
  vi.unstubAllGlobals();
});

beforeEach(() => {
  fetchMock.mockReset();
  handleUnauthorized.mockReset();
  vi.spyOn(Date, "now").mockReturnValue(1_000_000);
});

const logCalls = () => fetchMock.mock.calls.filter(([url]) => url === "/api/log");

const loggedBody = (call: unknown[]) => JSON.parse((call[1] as RequestInit).body as string);

describe("initApiLogger", () => {
  it("forwards the request and preserves explicit headers and credentials", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    const res = await globalThis.fetch("/api/x", {
      method: "POST",
      headers: { "X-A": "1" },
      credentials: "include",
    });

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/x",
      expect.objectContaining({ method: "POST", headers: { "X-A": "1" }, credentials: "include" }),
    );
  });

  it("copies Headers instances and defaults credentials to same-origin", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    await globalThis.fetch("/api/headers", { headers: new Headers({ "X-H": "2" }) });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.headers).toEqual({ "x-h": "2" });
    expect(init.credentials).toBe("same-origin");
  });

  it("copies array headers and keeps the method", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    await globalThis.fetch("/api/array", { method: "DELETE", headers: [["X-K", "3"]] });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.headers).toEqual({ "X-K": "3" });
    expect(init.method).toBe("DELETE");
  });

  it("resolves URL and Request inputs for the log payload", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 500 }));

    await globalThis.fetch(new URL("http://localhost/api/url"));
    await globalThis.fetch(new Request("http://localhost/api/request"));

    const logs = logCalls();
    expect(logs).toHaveLength(2);
    expect(loggedBody(logs[0]).data.url).toBe("http://localhost/api/url");
    expect(loggedBody(logs[1]).data.url).toBe("http://localhost/api/request");
  });

  it("handles 401 on protected endpoints only", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 401 }));

    await globalThis.fetch("/api/protected");
    expect(handleUnauthorized).toHaveBeenCalledTimes(1);

    await globalThis.fetch("/api/auth/login", { method: "POST" });
    expect(handleUnauthorized).toHaveBeenCalledTimes(1);
  });

  it("logs slow requests as warnings", async () => {
    let tick = 0;
    vi.spyOn(performance, "now").mockImplementation(() => {
      tick += 1;
      return tick * 1500;
    });
    fetchMock.mockResolvedValue(new Response("{}", { status: 200 }));

    await globalThis.fetch("/api/slow");

    const [log] = logCalls();
    expect(log).toBeDefined();
    const body = loggedBody(log);
    expect(body).toMatchObject({ level: "warning", message: "API slow" });
    expect(body.data.elapsed_ms).toBeGreaterThanOrEqual(1000);
    expect(body.data.status).toBe(200);
  });

  it("logs failed responses as errors", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 500 }));

    await globalThis.fetch("/api/broken");

    const [log] = logCalls();
    expect(log).toBeDefined();
    expect(loggedBody(log)).toMatchObject({
      level: "error",
      message: "API error",
      data: { status: 500, method: "GET" },
    });
  });

  it("deduplicates identical log lines within five seconds", async () => {
    fetchMock.mockResolvedValue(new Response("{}", { status: 500 }));

    await globalThis.fetch("/api/dup");
    await globalThis.fetch("/api/dup");

    expect(logCalls()).toHaveLength(1);
  });

  it("logs and rethrows network failures", async () => {
    const boom = new TypeError("offline");
    fetchMock.mockImplementation((url: string) =>
      url === "/api/log" ? Promise.resolve(new Response(null, { status: 204 })) : Promise.reject(boom),
    );

    await expect(globalThis.fetch("/api/fail")).rejects.toBe(boom);

    const [log] = logCalls();
    expect(log).toBeDefined();
    expect(loggedBody(log)).toMatchObject({ level: "error", message: "API failed" });
  });

  it("survives a synchronous failure of the log transport", async () => {
    const boom = new TypeError("offline");
    fetchMock.mockImplementation((url: string) => {
      if (url === "/api/log") throw new Error("log transport broken");
      return Promise.reject(boom);
    });

    await expect(globalThis.fetch("/api/sync-fail")).rejects.toBe(boom);
  });

  it("wraps fetch only once", () => {
    const wrapped = globalThis.fetch;

    initApiLogger();

    expect(globalThis.fetch).toBe(wrapped);
  });
});
