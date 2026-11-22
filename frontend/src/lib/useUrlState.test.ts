/**
 * useUrlState: чтение/запись одного параметра в hash (`#/admin?q=...`).
 * jsdom: hash меняется обычным присваиванием, но `hashchange` дополняем вручную, чтобы не
 * зависеть от тайминга навигации (тот приём уже используется в router.test.tsx).
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useUrlState } from "./useUrlState";

afterEach(() => {
  window.location.hash = "";
});

describe("useUrlState", () => {
  it("читает значение из hash при монтировании", () => {
    window.location.hash = "#/admin?q=%D0%B8%D0%B2%D0%B0%D0%BD&page=2";

    const search = renderHook(() => useUrlState<string>("q", ""));
    const page = renderHook(() => useUrlState<number>("page", 1));

    expect(search.result.current[0]).toBe("иван");
    expect(page.result.current[0]).toBe(2);
  });

  it("возвращает default, если параметра нет или он пуст", () => {
    window.location.hash = "#/admin?role=teacher&page=";

    expect(renderHook(() => useUrlState<string>("role", "all")).result.current[0]).toBe("teacher");
    expect(renderHook(() => useUrlState<string>("q", "")).result.current[0]).toBe("");
    expect(renderHook(() => useUrlState<number>("page", 1)).result.current[0]).toBe(1);
  });

  it("пишет через replaceState и сохраняет соседние параметры", () => {
    window.location.hash = "#/admin?role=teacher";
    const replace = vi.spyOn(window.history, "replaceState");
    const push = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useUrlState<string>("q", ""));

    act(() => result.current[1]("иван"));

    expect(result.current[0]).toBe("иван");
    expect(replace).toHaveBeenCalledWith(
      null,
      "",
      "#/admin?role=teacher&q=%D0%B8%D0%B2%D0%B0%D0%BD",
    );
    expect(push).not.toHaveBeenCalled();
    expect(window.location.hash).toBe(
      "#/admin?role=teacher&q=%D0%B8%D0%B2%D0%B0%D0%BD",
    );
  });

  it("pushState только по явной опции (смена страницы)", () => {
    window.location.hash = "#/admin";
    const replace = vi.spyOn(window.history, "replaceState");
    const push = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useUrlState<number>("page", 1));

    act(() => result.current[1](2));
    expect(replace).toHaveBeenCalledWith(null, "", "#/admin?page=2");

    act(() => result.current[1](3, { push: true }));
    expect(push).toHaveBeenCalledWith(null, "", "#/admin?page=3");
  });

  it("удаляет параметр при возврате к значению по умолчанию", () => {
    window.location.hash = "#/admin?page=3&q=ivan";
    const { result } = renderHook(() => useUrlState<number>("page", 1));

    act(() => result.current[1](1));

    expect(window.location.hash).toBe("#/admin?q=ivan");
  });

  it("реагирует на hashchange (назад/внешняя правка адреса)", () => {
    window.location.hash = "#/admin?q=old&page=1";
    const { result } = renderHook(() => useUrlState<string>("q", ""));

    act(() => {
      window.location.hash = "#/admin?q=new&page=1";
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });

    expect(result.current[0]).toBe("new");
  });
});
