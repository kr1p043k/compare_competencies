import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { KNOWN_TABS, useTabRoute } from "./router";

const reset = () => {
  window.location.hash = "";
};

afterEach(reset);

describe("useTabRoute", () => {
  it("falls back when there is no hash", () => {
    const { result } = renderHook(() => useTabRoute("vacancies"));
    expect(result.current[0]).toBe("vacancies");
  });

  it("reads the tab from the hash on mount", () => {
    window.location.hash = "#/teacher";
    const { result } = renderHook(() => useTabRoute("vacancies"));
    expect(result.current[0]).toBe("teacher");
  });

  it("ignores an unknown tab and uses the fallback", () => {
    window.location.hash = "#/nope";
    const { result } = renderHook(() => useTabRoute("vacancies"));
    expect(result.current[0]).toBe("vacancies");
  });

  it("navigate writes the hash", async () => {
    const { result } = renderHook(() => useTabRoute("vacancies"));

    act(() => result.current[1]("logs"));

    await waitFor(() => expect(window.location.hash).toBe("#/logs"));
    expect(result.current[0]).toBe("logs");
  });

  it("navigate accepts an updater function", async () => {
    window.location.hash = "#/logs";
    const { result } = renderHook(() => useTabRoute("vacancies"));

    act(() => result.current[1](() => "help"));

    await waitFor(() => expect(window.location.hash).toBe("#/help"));
  });

  it("reacts to external hash changes (back button)", async () => {
    window.location.hash = "#/logs";
    const { result } = renderHook(() => useTabRoute("vacancies"));

    await act(async () => {
      window.location.hash = "#/help";
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });

    expect(result.current[0]).toBe("help");
  });

  it("knows every nav tab", () => {
    expect(KNOWN_TABS).toContain("teacher");
    expect(KNOWN_TABS).toContain("vacancies");
    expect(new Set(KNOWN_TABS).size).toBe(KNOWN_TABS.length);
  });
});

describe("useTabRoute while disabled (login screen)", () => {
  it("does not default the hash on mount", () => {
    const { result } = renderHook(() => useTabRoute("vacancies", false));
    expect(window.location.hash).toBe("");
    // Память может знать fallback, но адрес не подменяется.
    expect(result.current[0]).toBe("vacancies");
  });

  it("does not normalize an unknown hash", () => {
    window.location.hash = "#/nope";
    renderHook(() => useTabRoute("vacancies", false));
    expect(window.location.hash).toBe("#/nope");
  });

  it("navigate() keeps the address untouched", () => {
    const { result } = renderHook(() => useTabRoute("vacancies", false));

    act(() => result.current[1]("logs"));

    expect(window.location.hash).toBe("");
    expect(result.current[0]).toBe("logs");
  });

  it("re-enabling restores the defaulting behavior", async () => {
    const { result, rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) => useTabRoute("vacancies", enabled),
      { initialProps: { enabled: false } },
    );
    expect(window.location.hash).toBe("");

    rerender({ enabled: true });

    await waitFor(() => expect(window.location.hash).toBe("#/vacancies"));
    expect(result.current[0]).toBe("vacancies");
  });

  it("re-enabling keeps a valid deep link", async () => {
    window.location.hash = "#/teacher";
    const { result, rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) => useTabRoute("vacancies", enabled),
      { initialProps: { enabled: false } },
    );

    rerender({ enabled: true });

    await waitFor(() => expect(result.current[0]).toBe("teacher"));
    expect(window.location.hash).toBe("#/teacher");
  });
});
