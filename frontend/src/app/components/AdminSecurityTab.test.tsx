/**
 * AdminSecurityTab: глобальный переключатель 2FA. GET рисует состояние,
 * включение сразу шлёт PUT, выключение требует диалог подтверждения,
 * ошибка PUT откатывает оптимистичное состояние. Сеть мокается на уровне
 * lib/api.request - реальный fetch не вызывается.
 */
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, request } from "../../lib/api";
import { AdminSecurityTab } from "./AdminSecurityTab";

vi.mock("../../lib/api", () => {
  class ApiError extends Error {
    constructor(
      message: string,
      readonly status = 0,
      readonly detail: unknown = null,
      readonly url = "",
    ) {
      super(message);
      this.name = "ApiError";
    }
  }
  return { request: vi.fn(), ApiError };
});

// Явная очистка: при singleFork автоочистка RTL не срабатывает между тестами.
afterEach(() => {
  cleanup();
});

const requestMock = vi.mocked(request);

type SecuritySettings = {
  two_factor_enabled: boolean;
  enrolled_admins: number;
  active_admins: number;
};

/** PUT-вызовы переключателя (GET загрузки сюда не попадает). */
const putCalls = () =>
  requestMock.mock.calls.filter(
    ([url, options]) =>
      url === "/api/admin/security/settings" &&
      (options as { method?: string } | undefined)?.method === "PUT",
  );

/**
 * Роутер по URL: GET /api/admin/security/settings отвечает текущим state,
 * PUT применяет body и возвращает новый state (или уходит в reject через put).
 * Возвращает патчер state: тест 1 перерисовывает вкладку с другим ответом GET.
 */
function mockSecurityApi(
  initial: SecuritySettings,
  put?: (body: unknown) => Promise<unknown>,
) {
  let state = { ...initial };
  requestMock.mockImplementation(
    ((url: string, options?: { method?: string; body?: unknown }) => {
      if (url !== "/api/admin/security/settings") {
        return Promise.reject(new Error(`unexpected endpoint: ${url}`));
      }
      if (options?.method === "PUT") {
        if (put) return put(options.body);
        state = { ...state, ...(options.body as Partial<SecuritySettings>) };
        return Promise.resolve({ ...state });
      }
      return Promise.resolve({ ...state });
    }) as never,
  );
  return (patch: Partial<SecuritySettings>) => {
    state = { ...state, ...patch };
  };
}

describe("AdminSecurityTab", () => {
  beforeEach(() => {
    requestMock.mockReset();
  });

  it("рисует состояние из GET: aria-checked и подпись переключателя", async () => {
    const patch = mockSecurityApi({
      two_factor_enabled: true,
      enrolled_admins: 1,
      active_admins: 3,
    });
    render(<AdminSecurityTab />);

    const onSwitch = await screen.findByRole("switch");
    expect(onSwitch).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("Включена для всех администраторов")).toBeInTheDocument();

    cleanup();
    patch({ two_factor_enabled: false });
    render(<AdminSecurityTab />);

    const offSwitch = await screen.findByRole("switch");
    expect(offSwitch).toHaveAttribute("aria-checked", "false");
    expect(screen.getByText("Выключена")).toBeInTheDocument();
  });

  it("показывает счётчики настроивших администраторов", async () => {
    mockSecurityApi({ two_factor_enabled: false, enrolled_admins: 1, active_admins: 3 });
    render(<AdminSecurityTab />);

    await screen.findByRole("switch");
    expect(screen.getByText("Настроили: 1 из 3", { exact: false })).toBeInTheDocument();
  });

  it("включение сразу шлёт PUT и показывает aria-live сообщение об успехе", async () => {
    const user = userEvent.setup();
    mockSecurityApi({ two_factor_enabled: false, enrolled_admins: 1, active_admins: 3 });
    render(<AdminSecurityTab />);
    await screen.findByRole("switch");

    await user.click(screen.getByRole("switch"));

    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/admin/security/settings", {
        method: "PUT",
        body: { two_factor_enabled: true },
      }),
    );
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
    const notice = await screen.findByRole("status");
    expect(notice).toHaveAttribute("aria-live", "polite");
    expect(notice).toHaveTextContent(
      "Двухфакторная аутентификация включена: при входе администраторам потребуется код.",
    );
  });

  it("выключение требует диалог: отмена ничего не шлёт, подтверждение шлёт PUT false", async () => {
    const user = userEvent.setup();
    mockSecurityApi({ two_factor_enabled: true, enrolled_admins: 1, active_admins: 3 });
    render(<AdminSecurityTab />);
    expect(await screen.findByRole("switch")).toHaveAttribute("aria-checked", "true");

    await user.click(screen.getByRole("switch"));
    const dialog = await screen.findByRole("dialog", {
      name: "Выключить двухфакторную аутентификацию?",
    });
    // Диалог открыт, но без подтверждения PUT не отправлялся.
    expect(putCalls()).toHaveLength(0);

    await user.click(within(dialog).getByRole("button", { name: "Отмена" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(putCalls()).toHaveLength(0);

    // Повторное выключение с подтверждением отправляет PUT с false.
    await user.click(screen.getByRole("switch"));
    const dialog2 = await screen.findByRole("dialog", {
      name: "Выключить двухфакторную аутентификацию?",
    });
    await user.click(within(dialog2).getByRole("button", { name: "Выключить" }));

    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/admin/security/settings", {
        method: "PUT",
        body: { two_factor_enabled: false },
      }),
    );
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "false");
  });

  it("ошибка PUT откатывает переключатель и показывает текст ошибки", async () => {
    const user = userEvent.setup();
    mockSecurityApi(
      { two_factor_enabled: false, enrolled_admins: 1, active_admins: 3 },
      () =>
        Promise.reject(
          new ApiError("Внутренняя ошибка сервера", 500, null, "/api/admin/security/settings"),
        ),
    );
    render(<AdminSecurityTab />);
    const btn = await screen.findByRole("switch");
    expect(btn).toHaveAttribute("aria-checked", "false");

    await user.click(btn);

    // 5xx: универсальный текст вместо детали сервера, состояние откатилось.
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Сервер временно недоступен. Попробуйте позже.",
    );
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "false");
  });
});
