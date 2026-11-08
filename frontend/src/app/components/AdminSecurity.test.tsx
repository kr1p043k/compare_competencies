/**
 * AdminSecurity: глобальный тумблер 2FA и личное подключение TOTP.
 * Сеть мокается на уровне lib/api.request (маршрутизация по URL и методу),
 * реальный fetch не вызывается.
 *
 * «Тумблер» в компоненте - это Button с бейджем состояния: role="switch" и
 * aria-checked в DOM нет, поэтому состояние флага проверяется по бейджу
 * («Включена»/«Выключена») и подписи кнопки («Включить»/«Выключить»).
 * Skeleton или подписи «Загрузка» компонент не рисует: пока GET не ответил,
 * виден лишь задизейбленный тумблер и счётчик «подключено админов: 0».
 */
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, request } from "../../lib/api";
import { AdminSecurity } from "./AdminSecurity";

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

afterEach(() => {
  // Явная очистка: при singleFork автоочистка RTL не срабатывает между файлами.
  cleanup();
});

const requestMock = vi.mocked(request);

/** Состояние глобального флага: ответ GET и PUT /api/admin/security/2fa/settings. */
let sw: { enabled: boolean; read_ok?: boolean; enrolled_admins: number };

/**
 * Ответ GET /api/auth/2fa/status: сервер отдаёт оба набора имён сразу -
 * enabled/enrolled/available (читает компонент) и
 * globally_enabled/configured/confirmed (src/api_pkg/routers/auth.py).
 */
let status: Record<string, unknown>;

const SECRET = "JBSWY3DPEHPK3PXP";
const OTPAUTH_URI = `otpauth://totp/Compare:admin@sfedu.ru?secret=${SECRET}&issuer=Compare`;
const RECOVERY_CODES = ["1111-aaaa", "2222-bbbb", "3333-cccc", "4444-dddd"];

function statusResponse(globalOn: boolean, enrolled: boolean) {
  return {
    enabled: globalOn,
    enrolled,
    available: true,
    globally_enabled: globalOn,
    configured: enrolled,
    confirmed: enrolled,
  };
}

function route(url: string, options?: { method?: string; body?: unknown }): Promise<unknown> {
  if (url === "/api/admin/security/2fa/settings") {
    if (options?.method === "PUT") {
      // PUT возвращает новое состояние флага целиком (как бэкенд).
      sw = { ...sw, ...(options.body as { enabled: boolean }) };
    }
    return Promise.resolve({ ...sw });
  }
  if (url === "/api/auth/2fa/status") return Promise.resolve({ ...status });
  if (url === "/api/auth/2fa/setup" && options?.method === "POST") {
    return Promise.resolve({ secret: SECRET, otpauth_uri: OTPAUTH_URI });
  }
  if (url === "/api/auth/2fa/confirm" && options?.method === "POST") {
    // После confirm статус админа на бэкенде уже «подключён».
    status = statusResponse(true, true);
    return Promise.resolve({ recovery_codes: RECOVERY_CODES });
  }
  return Promise.reject(new ApiError(`unexpected endpoint: ${url}`, 500, null, url));
}

/** Вызовы PUT глобального флага. */
const putCalls = () =>
  requestMock.mock.calls.filter(
    ([url, options]) => url === "/api/admin/security/2fa/settings" && options?.method === "PUT",
  );

describe("AdminSecurity", () => {
  beforeEach(() => {
    sw = { enabled: false, enrolled_admins: 2 };
    status = statusResponse(false, false);
    requestMock.mockReset();
    requestMock.mockImplementation(route as never);
  });

  it("пока настройки не загружены, тумблер задизейблен; после ответа видно реальное состояние", async () => {
    // Держим первый GET флага неразрешённым, статус отвечает сразу.
    let releaseSettings!: (value: unknown) => void;
    requestMock.mockImplementation(((url: string) => {
      if (url === "/api/admin/security/2fa/settings") {
        return new Promise((resolve) => {
          releaseSettings = resolve;
        });
      }
      return route(url);
    }) as never);

    render(<AdminSecurity />);

    // До ответа: нейтральное состояние (счётчик 0) и выключенный тумблер.
    expect(screen.getByRole("button", { name: "Включить" })).toBeDisabled();
    expect(screen.queryByText("подключено админов: 2")).toBeNull();

    releaseSettings({ enabled: false, enrolled_admins: 2 });

    await screen.findByText("подключено админов: 2");
    expect(screen.getByRole("button", { name: "Включить" })).toBeEnabled();
  });

  it("глобальный флаг из GET отражается бейджем и подписью кнопки", async () => {
    sw = { enabled: true, enrolled_admins: 1 };
    const enabled = render(<AdminSecurity />);
    await screen.findByText("Включена");
    expect(screen.getByRole("button", { name: "Выключить" })).toBeInTheDocument();
    enabled.unmount();

    sw = { enabled: false, enrolled_admins: 0 };
    render(<AdminSecurity />);
    await screen.findByText("Выключена");
    expect(screen.getByRole("button", { name: "Включить" })).toBeInTheDocument();
  });

  it("включение шлёт PUT с enabled:true и обновляет состояние", async () => {
    sw = { enabled: false, enrolled_admins: 2 };
    const user = userEvent.setup();
    render(<AdminSecurity />);

    await user.click(await screen.findByRole("button", { name: "Включить" }));

    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/admin/security/2fa/settings", {
        method: "PUT",
        body: { enabled: true },
      }),
    );
    await screen.findByText("Включена");
    expect(screen.getByText("Двухфакторная аутентификация включена")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Выключить" })).toBeInTheDocument();
  });

  it("выключение применяется сразу: PUT без диалога подтверждения", async () => {
    // Диалога подтверждения у глобального тумблера в компоненте нет -
    // toggle() шлёт PUT сразу по клику (AdminSecurity.tsx, toggle).
    sw = { enabled: true, enrolled_admins: 1 };
    const user = userEvent.setup();
    render(<AdminSecurity />);
    await screen.findByText("Включена");

    expect(putCalls()).toHaveLength(0);
    expect(screen.queryByRole("dialog")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Выключить" }));

    await waitFor(() => expect(putCalls()).toHaveLength(1));
    expect(requestMock).toHaveBeenCalledWith("/api/admin/security/2fa/settings", {
      method: "PUT",
      body: { enabled: false },
    });
    expect(screen.queryByRole("dialog")).toBeNull();
    await screen.findByText("Выключена");
    expect(screen.getByText("Двухфакторная аутентификация выключена")).toBeInTheDocument();
  });

  it("подключение доступно при включённом флаге; при выключенном - подсказка и disabled", async () => {
    sw = { enabled: true, enrolled_admins: 0 };
    status = statusResponse(true, false); // configured:false, подтверждения нет
    const globalOn = render(<AdminSecurity />);
    const setup = await screen.findByRole("button", { name: "Подключить приложение-аутентификатор" });
    expect(setup).toBeEnabled();
    expect(screen.getByText("без кода")).toBeInTheDocument();
    expect(screen.queryByText("Сначала включите 2FA тумблером выше.")).toBeNull();
    globalOn.unmount();

    sw = { enabled: false, enrolled_admins: 0 };
    status = statusResponse(false, false); // globally_enabled:false
    render(<AdminSecurity />);
    const setupOff = await screen.findByRole("button", { name: "Подключить приложение-аутентификатор" });
    expect(setupOff).toBeDisabled();
    expect(screen.getByText("Сначала включите 2FA тумблером выше.")).toBeInTheDocument();
  });

  it("подключение: setup показывает секрет, confirm сохраняет резервные коды", async () => {
    sw = { enabled: true, enrolled_admins: 0 };
    const user = userEvent.setup();
    render(<AdminSecurity />);

    await user.click(
      await screen.findByRole("button", { name: "Подключить приложение-аутентификатор" }),
    );
    expect(requestMock).toHaveBeenCalledWith("/api/auth/2fa/setup", {
      method: "POST",
      body: {},
    });
    expect(await screen.findByText(SECRET)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Открыть ссылку otpauth://" })).toHaveAttribute(
      "href",
      OTPAUTH_URI,
    );

    await user.type(screen.getByPlaceholderText("123456"), "123456");
    await user.click(screen.getByRole("button", { name: "Подтвердить" }));

    await screen.findByText("Резервные коды - сохраните, показываются один раз");
    for (const code of RECOVERY_CODES) {
      expect(screen.getByText(code)).toBeInTheDocument();
    }
    expect(requestMock).toHaveBeenCalledWith("/api/auth/2fa/confirm", {
      method: "POST",
      body: { code: "123456" },
    });
    // Секрет больше не показывается, аккаунт после перезагрузки статуса защищён.
    expect(screen.queryByText(SECRET)).toBeNull();
    expect(screen.getByText("защищена")).toBeInTheDocument();
  });

  it("read_ok=false: предупреждает, что состояние не читается и действует значение по умолчанию", async () => {
    sw = { enabled: false, read_ok: false, enrolled_admins: 2 };
    render(<AdminSecurity />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "состояние не читается: действует значение по умолчанию (выключено)",
    );
    // Показанное состояние - значение по умолчанию, а не прочитанное из БД.
    expect(screen.getByText("Выключена")).toBeInTheDocument();
  });

  it("read_ok=true: предупреждения о нечитаемом состоянии нет", async () => {
    sw = { enabled: false, read_ok: true, enrolled_admins: 2 };
    render(<AdminSecurity />);

    await screen.findByText("Выключена");
    expect(screen.queryByRole("alert")).toBeNull();
  });
});


describe("AdminSecurity: флаг прочитан не был", () => {
  beforeEach(() => {
    // Свой beforeEach: у соседнего describe он локальный и сюда не попадает.
    requestMock.mockReset();
    requestMock.mockImplementation(route as never);
  });

  it("показывает предупреждение, когда read_ok=false", async () => {
    sw = { enabled: false, enrolled_admins: 0, read_ok: false };
    status = statusResponse(false, false);
    render(<AdminSecurity />);
    expect(await screen.findByText(/Состояние флага не прочитано/i)).toBeTruthy();
  });

  it("не показывает предупреждение, когда read_ok не задан", async () => {
    sw = { enabled: true, enrolled_admins: 1 };
    status = statusResponse(true, true);
    render(<AdminSecurity />);
    await screen.findByText("Включена");
    expect(screen.queryByText(/Состояние флага не прочитано/i)).toBeNull();
  });
});
