import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TwoFactorCard } from "./TwoFactorCard";

vi.mock("../../lib/errors", () => ({ reportError: vi.fn() }));

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const STATUS_GLOBALLY_OFF = { globally_enabled: false, configured: false, confirmed: false };
const STATUS_READY = { globally_enabled: true, configured: false, confirmed: false };
const STATUS_CONFIRMED = { globally_enabled: true, configured: true, confirmed: true };
const SETUP = {
  otpauth_uri: "otpauth://totp/Test:user?secret=ABC234",
  secret: "ABC234",
};

/**
 * Подменяет глобальный fetch таблицей маршрутов и возвращает журнал вызовов:
 * так тесты проходят через реальный request() и видят method/body дословно.
 */
function installRoutes(routes: Record<string, unknown>): { path: string; init?: RequestInit }[] {
  const calls: { path: string; init?: RequestInit }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      calls.push({ path, init });
      if (!(path in routes)) return json({ detail: `нет маршрута ${path}` }, 404);
      const body = routes[path];
      return json(typeof body === "function" ? (body as () => unknown)() : body);
    }),
  );
  return calls;
}

// Явный cleanup: при singleFork автоочистка RTL не срабатывает между файлами.
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("TwoFactorCard", () => {
  it("глобально выключена: статус «выключена администратором» и неактивная кнопка", async () => {
    installRoutes({ "/api/auth/2fa/status": STATUS_GLOBALLY_OFF });

    render(<TwoFactorCard />);

    expect(await screen.findByText("выключена администратором")).toBeInTheDocument();
    const enable = screen.getByRole("button", { name: "Включить 2FA" });
    expect(enable).toBeDisabled();
  });

  it("«Включить 2FA» вызывает setup и диалог показывает секрет, otpauth-uri и копирование", async () => {
    const calls = installRoutes({
      "/api/auth/2fa/status": STATUS_READY,
      "/api/auth/2fa/setup": SETUP,
    });
    const user = userEvent.setup();

    render(<TwoFactorCard />);

    await user.click(await screen.findByRole("button", { name: "Включить 2FA" }));

    expect(await screen.findByText("ABC234")).toBeInTheDocument();
    expect(screen.getByText("otpauth://totp/Test:user?secret=ABC234")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /Скопировать/ })).toHaveLength(2);
    const setupCall = calls.find((c) => c.path === "/api/auth/2fa/setup");
    expect(setupCall?.init?.method).toBe("POST");
  });

  it("подтверждение отправляет код на /api/auth/2fa/confirm и показывает recovery-коды", async () => {
    const calls = installRoutes({
      "/api/auth/2fa/status": STATUS_READY,
      "/api/auth/2fa/setup": SETUP,
      "/api/auth/2fa/confirm": { status: "enabled", recovery_codes: ["aa11bb22", "cc33dd44"] },
    });
    const user = userEvent.setup();

    render(<TwoFactorCard />);

    await user.click(await screen.findByRole("button", { name: "Включить 2FA" }));
    await screen.findByText("ABC234");
    await user.type(screen.getByLabelText("Код из приложения"), "123456");
    await user.click(screen.getByRole("button", { name: "Подтвердить" }));

    expect(await screen.findByText("aa11bb22")).toBeInTheDocument();
    expect(screen.getByText("cc33dd44")).toBeInTheDocument();
    const confirmCall = calls.find((c) => c.path === "/api/auth/2fa/confirm");
    expect(confirmCall?.init?.method).toBe("POST");
    expect(confirmCall?.init?.body).toBe(JSON.stringify({ code: "123456" }));

    // «Сохранил» закрывает диалог с recovery-кодами.
    await user.click(screen.getByRole("button", { name: "Сохранил" }));
    await waitFor(() => expect(screen.queryByText("aa11bb22")).not.toBeInTheDocument());
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("подтверждённая 2FA: «включена» и отключение отправляет код на /api/auth/2fa/disable", async () => {
    const calls = installRoutes({
      "/api/auth/2fa/status": STATUS_CONFIRMED,
      "/api/auth/2fa/disable": { status: "disabled" },
    });
    const user = userEvent.setup();

    render(<TwoFactorCard />);

    expect(await screen.findByText("включена")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Отключить" }));
    await user.type(
      screen.getByLabelText("Код из приложения или recovery-код"),
      "123456",
    );
    await user.click(screen.getByRole("button", { name: "Отключить 2FA" }));

    await waitFor(() => {
      const disableCall = calls.find((c) => c.path === "/api/auth/2fa/disable");
      expect(disableCall?.init?.method).toBe("POST");
      expect(disableCall?.init?.body).toBe(JSON.stringify({ code: "123456" }));
    });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});
