import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { handleUnauthorized, authHeaders } = vi.hoisted(() => ({
  handleUnauthorized: vi.fn(),
  authHeaders: vi.fn(() => ({ "X-Session": "1" })),
}));

vi.mock("./auth", () => ({ authHeaders, handleUnauthorized }));

import { ApiError, download, request, saveBlob } from "./api";

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("request details", () => {
  beforeEach(() => {
    handleUnauthorized.mockClear();
    authHeaders.mockClear();
  });

  it("joins FastAPI validation details", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: [{ msg: "a" }, { msg: "b" }, { other: 1 }] }, 422),
      ),
    );

    await expect(request("/api/x")).rejects.toMatchObject({ status: 422, message: "a; b" });
  });

  it("reads a nested message object", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: { message: "вложенное" } }, 400)),
    );

    await expect(request("/api/x")).rejects.toMatchObject({ message: "вложенное" });
  });

  it("uses the status text when the detail is empty", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ detail: null }, 400))
      .mockResolvedValueOnce(jsonResponse({}, 500))
      .mockResolvedValueOnce(jsonResponse({}, 418));
    vi.stubGlobal("fetch", fetchMock);

    await expect(request("/api/a")).rejects.toMatchObject({ message: "Некорректный запрос" });
    await expect(request("/api/b")).rejects.toMatchObject({
      message: "Внутренняя ошибка сервера",
    });
    await expect(request("/api/c")).rejects.toMatchObject({
      message: "Ошибка запроса (HTTP 418)",
    });
  });

  it("treats a non-JSON error body as an empty detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("oops", { status: 503 })),
    );

    await expect(request("/api/x")).rejects.toMatchObject({
      status: 503,
      message: "Сервис временно недоступен",
    });
  });

  it("rejects invalid JSON with a readable ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("<html>", { status: 200 })),
    );

    const error = await request("/api/x").catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 200,
      message: "Некорректный ответ сервера",
      url: "/api/x",
    });
  });

  it("resets the session on 401 for protected endpoints", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({}, 401)));

    await expect(request("/api/x")).rejects.toBeInstanceOf(ApiError);

    expect(handleUnauthorized).toHaveBeenCalledTimes(1);
  });

  it("does not reset the session for anonymous requests or login", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}, 401));
    vi.stubGlobal("fetch", fetchMock);

    await expect(request("/api/x", { anonymous: true })).rejects.toBeInstanceOf(ApiError);
    await expect(
      request("/api/auth/login", { method: "POST", body: {} }),
    ).rejects.toBeInstanceOf(ApiError);

    expect(handleUnauthorized).not.toHaveBeenCalled();
  });

  it("adds auth headers unless the request is anonymous", async () => {
    const fetchMock = vi.fn().mockImplementation(() => jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);

    await request("/api/a");
    await request("/api/b", { anonymous: true, headers: { "X-Extra": "2" } });

    expect(fetchMock.mock.calls[0][1].headers).toMatchObject({ "X-Session": "1" });
    expect(fetchMock.mock.calls[1][1].headers).toEqual({ "X-Extra": "2" });
  });

  it("passes Blob and string bodies through untouched", async () => {
    const fetchMock = vi.fn().mockImplementation(() => jsonResponse({}));
    vi.stubGlobal("fetch", fetchMock);
    const blob = new Blob(["x"]);

    await request("/api/blob", { method: "POST", body: blob });
    await request("/api/text", { method: "POST", body: "raw" });

    expect(fetchMock.mock.calls[0][1].body).toBe(blob);
    expect(fetchMock.mock.calls[0][1].headers["Content-Type"]).toBeUndefined();
    expect(fetchMock.mock.calls[1][1].body).toBe("raw");
  });
});

describe("download", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the blob and the filename from Content-Disposition", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("csv", {
          status: 200,
          headers: { "content-disposition": 'attachment; filename="report.csv"' },
        }),
      ),
    );

    const { blob, filename } = await download("/api/report");

    expect(blob).toBeInstanceOf(Blob);
    expect(filename).toBe("report.csv");
  });

  it("decodes RFC 5987 filenames and tolerates a missing header", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response("x", {
          headers: {
            "content-disposition": "attachment; filename*=UTF-8''%D0%BE%D1%82%D1%87%D1%91%D1%82.csv",
          },
        }),
      )
      .mockResolvedValueOnce(new Response("y"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(download("/api/a")).resolves.toMatchObject({ filename: "отчёт.csv" });
    await expect(download("/api/b")).resolves.toMatchObject({ filename: null });
  });
});

describe("saveBlob", () => {
  const createObjectURL = vi.fn(() => "blob:mock");
  const revokeObjectURL = vi.fn();

  beforeEach(() => {
    createObjectURL.mockClear();
    revokeObjectURL.mockClear();
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: createObjectURL,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: revokeObjectURL,
    });
  });

  afterEach(() => {
    delete (URL as unknown as Record<string, unknown>).createObjectURL;
    delete (URL as unknown as Record<string, unknown>).revokeObjectURL;
  });

  it("clicks a temporary link and revokes the object URL", () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const blob = new Blob(["x"]);

    saveBlob(blob, "report.csv");

    expect(createObjectURL).toHaveBeenCalledWith(blob);
    expect(click).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock");
    expect(document.querySelector("a[download='report.csv']")).toBeNull();
  });
});
