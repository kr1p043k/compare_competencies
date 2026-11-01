import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, request } from "./api";

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

describe("request", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("parses a successful JSON response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ ok: true })));

    await expect(request<{ ok: boolean }>("/api/x")).resolves.toEqual({ ok: true });
  });

  it("returns undefined for 204", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    await expect(request("/api/x")).resolves.toBeUndefined();
  });

  it("throws ApiError with FastAPI detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Неверный логин или пароль" }, 401)),
    );

    await expect(request("/api/auth/login", { anonymous: true })).rejects.toMatchObject({
      name: "ApiError",
      status: 401,
      message: "Неверный логин или пароль",
    });
  });

  it("throws ApiError with a human message for 429", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({}, 429)));

    await expect(request("/api/x")).rejects.toMatchObject({
      status: 429,
      message: expect.stringContaining("Слишком много запросов"),
    });
  });

  it("wraps network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed to fetch")));

    const error = await request("/api/x").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(0);
    expect((error as ApiError).message).toBe("Нет связи с сервером");
  });

  it("rethrows AbortError instead of masking it as a network failure", async () => {
    const abort = new DOMException("signal is aborted without reason", "AbortError");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(abort));

    const error = await request("/api/x").catch((e: unknown) => e);

    expect(error).toBe(abort);
    expect((error as DOMException).name).toBe("AbortError");
  });

  it("serialises object bodies and sets Content-Type", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);

    await request("/api/x", { method: "POST", body: { a: 1 } });

    const [, init] = fetchMock.mock.calls[0];
    expect(init.body).toBe(JSON.stringify({ a: 1 }));
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
  });

  it("does not set Content-Type for FormData and sends it as is", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);

    const fd = new FormData();
    fd.append("file", "x");
    await request("/api/x", { method: "POST", body: fd });

    const [, init] = fetchMock.mock.calls[0];
    expect(init.body).toBe(fd);
    expect((init.headers as Record<string, string>)["Content-Type"]).toBeUndefined();
  });

  it("flags auth errors", async () => {
    expect(new ApiError("x", 401, null, "/").isAuthError).toBe(true);
    expect(new ApiError("x", 403, null, "/").isAuthError).toBe(true);
    expect(new ApiError("x", 500, null, "/").isAuthError).toBe(false);
    expect(new ApiError("x", 429, null, "/").isRateLimited).toBe(true);
  });
});