import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { useTabNavigation } from "./useTabNavigation";

const reset = () => {
  window.location.hash = "";
};

afterEach(reset);

/**
 * Трединг флага `enabled` (в App это isAuth) через useTabNavigation в
 * useTabRoute: пока пользователь не вошёл, экран входа не должен подменять
 * адрес на `#/vacancies` (баг «login превращает / в #/vacancies»).
 *
 * Роль и enabled меняются в App одним setState (коммитом auth.tsx):
 * isAuth=true приходит в том же state, что и role. Сценарий «роль пришла,
 * а enabled ещё нет» в реальном APP невозможен, поэтому тесты гоняют оба
 * пропса вместе.
 */
describe("useTabNavigation router gating", () => {
  it("does not touch the host's address while disabled (login screen)", () => {
    const { result } = renderHook(() => useTabNavigation("vacancies", null, false));
    expect(window.location.hash).toBe("");

    act(() => result.current[1]("logs"));

    expect(window.location.hash).toBe("");
  });

  it("redirects from a role-denied tab in memory only, keeping the address", () => {
    // «admin» недоступен без роли: гард уводит на fallback,
    // но экран входа не должен чинить за собой URL.
    window.location.hash = "#/admin";
    const { result } = renderHook(() => useTabNavigation("vacancies", null, false));

    expect(result.current[0]).toBe("vacancies");
    expect(window.location.hash).toBe("#/admin");
  });

  it("defaults the address again once enabled (authed, прежнее поведение)", async () => {
    const { result, rerender } = renderHook(
      ({ enabled }: { enabled: boolean }) => useTabNavigation("vacancies", "student", enabled),
      { initialProps: { enabled: false } },
    );
    expect(window.location.hash).toBe("");

    rerender({ enabled: true });

    await waitFor(() => {
      expect(window.location.hash).toBe("#/vacancies");
      expect(result.current[0]).toBe("vacancies");
    });
  });

  it("takes a role-denied deep link away after login", async () => {
    window.location.hash = "#/admin";
    const { result, rerender } = renderHook(
      ({ role, enabled }: { role: string | null; enabled: boolean }) =>
        useTabNavigation("vacancies", role, enabled),
      { initialProps: { role: null as string | null, enabled: false } },
    );
    // Логин: адрес с чужой вкладкой не трогаем.
    expect(window.location.hash).toBe("#/admin");

    // Вход студентом: роль и enabled приходят одним коммитом.
    rerender({ role: "student", enabled: true });

    await waitFor(() => {
      expect(window.location.hash).toBe("#/vacancies");
      expect(result.current[0]).toBe("vacancies");
    });
  });
});
