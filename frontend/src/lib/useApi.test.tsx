import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useApi, useMutation } from "./useApi";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useApi", () => {
  it("loads data and clears loading", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ value: 42 })));

    const { result } = renderHook(() => useApi<{ value: number }>("/api/x"));

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toEqual({ value: 42 });
    expect(result.current.error).toBeNull();
  });

  it("exposes ApiError instead of swallowing it", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ detail: "boom" }, 500)));

    const { result } = renderHook(() => useApi("/api/x"));

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.error?.status).toBe(500);
    expect(result.current.error?.message).toBe("boom");
    expect(result.current.data).toBeNull();
  });

  it("does not fetch when path is null", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useApi(null));

    expect(fetchMock).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(false);
  });

  it("refetch re-runs the request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ n: 1 }));
    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useApi("/api/x"));
    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => result.current.refetch());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });
});

describe("useMutation", () => {
  it("returns the result and clears error", async () => {
    const fn = vi.fn().mockResolvedValue({ ok: true });
    const { result } = renderHook(() => useMutation(fn));

    let value: unknown;
    await act(async () => {
      value = await result.current.run(undefined as never);
    });

    expect(value).toEqual({ ok: true });
    expect(result.current.error).toBeNull();
  });

  it("stores the error and rethrows", async () => {
    const { ApiError } = await import("./api");
    const failure = new ApiError("nope", 409, null, "/api/x");
    const fn = vi.fn().mockRejectedValue(failure);
    const { result } = renderHook(() => useMutation(fn));

    await act(async () => {
      await expect(result.current.run(undefined as never)).rejects.toBe(failure);
    });

    expect(result.current.error?.message).toBe("nope");
    expect(result.current.loading).toBe(false);
  });
});