import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AdminTwoFactorCard } from "./AdminTwoFactorCard";

vi.mock("../../lib/errors", () => ({ reportError: vi.fn() }));

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

const FEATURES_ACTIVE = {
  features: { totp_2fa_enabled: { enabled: true, default: false, source: "file" as const } },
};

const FEATURES_OFF = {
  features: { totp_2fa_enabled: { enabled: false, default: false, source: "env" as const } },
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

describe("AdminTwoFactorCard", () => {
  it("выключенный флаг: заметка про вход по паролю и источник из окружения", async () => {
    installRoutes({ "/api/admin/features": FEATURES_OFF });

    render(<AdminTwoFactorCard />);

    expect(await screen.findByText("выключена")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Второй фактор сейчас выключен: вход по паролю. Настроенные ранее ключи сохранены.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("значение из окружения")).toBeInTheDocument();
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "false");
  });

  it("переключатель отправляет PUT /api/admin/features/totp_2fa_enabled с новым значением", async () => {
    const calls = installRoutes({
      "/api/admin/features": FEATURES_OFF,
      "/api/admin/features/totp_2fa_enabled": FEATURES_ACTIVE,
    });
    const user = userEvent.setup();

    render(<AdminTwoFactorCard />);

    const toggle = await screen.findByRole("switch");
    await user.click(toggle);

    await waitFor(() => {
      const put = calls.find((c) => c.path === "/api/admin/features/totp_2fa_enabled");
      expect(put?.init?.method).toBe("PUT");
      expect(put?.init?.body).toBe(JSON.stringify({ enabled: true }));
    });
    expect(await screen.findByText("включена")).toBeInTheDocument();
    expect(screen.getByText("изменено вручную")).toBeInTheDocument();
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
  });
});
