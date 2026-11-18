import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useTheme } from "./theme";

const KEY = "cc-theme";

afterEach(() => {
  localStorage.clear();
  document.documentElement.classList.remove("dark");
  document.documentElement.style.colorScheme = "";
});

describe("useTheme", () => {
  it("defaults to light and applies it to the document", () => {
    const { result } = renderHook(() => useTheme());

    expect(result.current.theme).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(document.documentElement.style.colorScheme).toBe("light");
    expect(localStorage.getItem(KEY)).toBe("light");
  });

  it("restores a saved dark theme", () => {
    localStorage.setItem(KEY, "dark");

    const { result } = renderHook(() => useTheme());

    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.style.colorScheme).toBe("dark");
  });

  it("ignores an unknown saved value", () => {
    localStorage.setItem(KEY, "sepia");

    const { result } = renderHook(() => useTheme());

    expect(result.current.theme).toBe("light");
  });

  it("toggles and persists the choice", () => {
    const { result } = renderHook(() => useTheme());

    act(() => result.current.toggle());
    expect(result.current.theme).toBe("dark");
    expect(localStorage.getItem(KEY)).toBe("dark");

    act(() => result.current.toggle());
    expect(result.current.theme).toBe("light");
    expect(localStorage.getItem(KEY)).toBe("light");
  });

  it("falls back to light when storage cannot be read", () => {
    const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    const { result } = renderHook(() => useTheme());

    expect(result.current.theme).toBe("light");
    spy.mockRestore();
  });

  it("still toggles when storage cannot be written", () => {
    const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    const { result } = renderHook(() => useTheme());
    act(() => result.current.toggle());

    expect(result.current.theme).toBe("dark");
    spy.mockRestore();
  });
});
