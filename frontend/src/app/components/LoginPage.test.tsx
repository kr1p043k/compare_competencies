/**
 * LoginPage: локальный вход, шаг 2FA и первичная настройка TOTP
 * (totp_setup_required: у админа включена защита, но нет ключа).
 * Сеть мокается на уровне lib/api.request, реальный fetch не вызывается.
 */
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, request } from "../../lib/api";
import { LoginPage } from "./LoginPage";

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

// Явный cleanup: при singleFork автоочистка RTL не срабатывает между файлами.
afterEach(cleanup);

const requestMock = vi.mocked(request);
const onLogin = vi.fn();
const CHALLENGE = { status: "totp_required", challenge_id: "ch-42", expires_in: 300 };
type User = ReturnType<typeof userEvent.setup>;

async function submitCredentials(user: User, email = "admin@sfedu.ru") {
  await user.type(screen.getByLabelText("Почта"), email);
  await user.type(screen.getByLabelText("Пароль"), "secret-pass");
  await user.click(screen.getByRole("button", { name: "Войти" }));
}

/** Логин отвечает challenge, verifyUrl - готовой сессией. */
function routeChallenge(verifyUrl: string, session: { token: string; role: string; name: string }) {
  requestMock.mockImplementation(((url: string) => {
    if (url === "/api/auth/login") return Promise.resolve(CHALLENGE);
    if (url === verifyUrl) return Promise.resolve(session);
    return Promise.reject(new Error(`unexpected endpoint: ${url}`));
  }) as never);
}

describe("LoginPage", () => {
  beforeEach(() => {
    requestMock.mockReset();
    onLogin.mockClear();
  });

  it("передаёт токен, роль и имя в onLogin при успешном входе", async () => {
    const user = userEvent.setup();
    requestMock.mockResolvedValue({ token: "tok-1", role: "admin", name: "Иван Петров" });
    render(<LoginPage onLogin={onLogin} />);

    await submitCredentials(user);

    await waitFor(() => expect(onLogin).toHaveBeenCalledWith("tok-1", "admin", "Иван Петров"));
    expect(requestMock).toHaveBeenCalledWith("/api/auth/login", {
      method: "POST",
      body: { email: "admin@sfedu.ru", password: "secret-pass" },
      anonymous: true,
    });
  });

  it("вход без challenge зовёт onLogin один раз и не открывает шаг кода", async () => {
    const user = userEvent.setup();
    requestMock.mockResolvedValue({ token: "tok-plain", role: "admin", name: "Иван Петров" });
    render(<LoginPage onLogin={onLogin} />);

    await submitCredentials(user);

    await waitFor(() => expect(onLogin).toHaveBeenCalledWith("tok-plain", "admin", "Иван Петров"));
    expect(onLogin).toHaveBeenCalledTimes(1);
    // 2FA выключена: шаг с кодом не появляется, форма остаётся обычной.
    expect(screen.queryByLabelText(/Код из приложения/)).toBeNull();
  });

  it("401 показывает «Неверный логин или пароль»", async () => {
    const user = userEvent.setup();
    requestMock.mockRejectedValue(
      new ApiError("Неверные учётные данные", 401, null, "/api/auth/login"),
    );
    render(<LoginPage onLogin={onLogin} />);

    await submitCredentials(user);

    expect(await screen.findByRole("alert")).toHaveTextContent("Неверный логин или пароль");
    expect(onLogin).not.toHaveBeenCalled();
    // Ошибка не застревает в состоянии загрузки и шаг 2FA не открылся.
    expect(screen.getByRole("button", { name: "Войти" })).toBeEnabled();
    expect(screen.queryByLabelText(/Код из приложения/)).toBeNull();
  });

  it("ответ totp_required открывает шаг с кодом и не вызывает onLogin", async () => {
    const user = userEvent.setup();
    requestMock.mockResolvedValue(CHALLENGE);
    render(<LoginPage onLogin={onLogin} />);

    await submitCredentials(user);
    const step = await screen.findByRole("region", { name: "Двухфакторное подтверждение входа" });

    expect(within(step).getByLabelText("Код из приложения")).toBeInTheDocument();
    expect(onLogin).not.toHaveBeenCalled();
  });

  it("6-значный код уходит на /api/auth/2fa/verify и завершает вход", async () => {
    const user = userEvent.setup();
    routeChallenge("/api/auth/2fa/verify", { token: "tok-2fa", role: "teacher", name: "Ольга" });
    render(<LoginPage onLogin={onLogin} />);

    await submitCredentials(user);
    await screen.findByRole("region", { name: "Двухфакторное подтверждение входа" });
    await user.type(screen.getByLabelText("Код из приложения"), "123456");
    await user.click(screen.getByRole("button", { name: "Подтвердить" }));

    await waitFor(() => expect(onLogin).toHaveBeenCalledWith("tok-2fa", "teacher", "Ольга"));
    expect(requestMock).toHaveBeenCalledWith("/api/auth/2fa/verify", {
      method: "POST",
      body: { challenge_id: "ch-42", code: "123456" },
      anonymous: true,
    });
  });

  it("резервный код переключает поле и уходит на /api/auth/2fa/recovery", async () => {
    const user = userEvent.setup();
    routeChallenge("/api/auth/2fa/recovery", { token: "tok-rec", role: "rop", name: "Мария" });
    render(<LoginPage onLogin={onLogin} />);

    await submitCredentials(user);
    await screen.findByRole("region", { name: "Двухфакторное подтверждение входа" });
    await user.click(screen.getByRole("button", { name: "Использовать резервный код" }));
    await user.type(screen.getByLabelText("Резервный код"), "a1b2c3d4");
    await user.click(screen.getByRole("button", { name: "Подтвердить" }));

    await waitFor(() => expect(onLogin).toHaveBeenCalledWith("tok-rec", "rop", "Мария"));
    expect(requestMock).toHaveBeenCalledWith("/api/auth/2fa/recovery", {
      method: "POST",
      body: { challenge_id: "ch-42", code: "a1b2c3d4" },
      anonymous: true,
    });
  });

  it("Назад сбрасывает challenge и возвращает форму реквизитов", async () => {
    const user = userEvent.setup();
    requestMock.mockResolvedValue(CHALLENGE);
    render(<LoginPage onLogin={onLogin} />);

    await submitCredentials(user);
    await screen.findByRole("region", { name: "Двухфакторное подтверждение входа" });
    await user.type(screen.getByLabelText("Код из приложения"), "123456");
    await user.click(screen.getByRole("button", { name: "Назад" }));

    expect(screen.getByLabelText("Почта")).toBeInTheDocument();
    expect(screen.queryByLabelText(/Код из приложения/)).not.toBeInTheDocument();
    expect(onLogin).not.toHaveBeenCalled();
  });

  it("totp_setup_required: настройка 2FA целиком на экране входа", async () => {
    const user = userEvent.setup();
    // login -> enroll_id; setup -> секрет; confirm -> recovery-коды + сессия.
    requestMock.mockImplementation(((url: string) => {
      if (url === "/api/auth/login")
        return Promise.resolve({
          status: "totp_setup_required",
          enroll_id: "enr-7",
          expires_in: 600,
        });
      if (url === "/api/auth/2fa/setup")
        return Promise.resolve({
          otpauth_uri: "otpauth://totp/CGA:admin%40sfedu.ru?secret=KQXG5C2M&issuer=CGA",
          secret: "KQXG5C2M",
          enroll_id: "enr-7",
        });
      if (url === "/api/auth/2fa/confirm")
        return Promise.resolve({
          status: "enabled",
          recovery_codes: ["aa11bb22cc33dd44", "11ff22ee33dd44cc"],
          token: "tok-enroll",
          role: "admin",
          name: "Иван Петров",
        });
      return Promise.reject(new Error(`unexpected endpoint: ${url}`));
    }) as never);
    render(<LoginPage onLogin={onLogin} />);

    await submitCredentials(user);
    const step = await screen.findByRole("region", { name: "Настройка двухфакторной защиты" });

    // Секрет запрашивается сразу с enroll_id из ответа логина.
    expect(requestMock).toHaveBeenCalledWith("/api/auth/2fa/setup", {
      method: "POST",
      body: { enroll_id: "enr-7" },
      anonymous: true,
    });
    // Секрет и ссылку видно - аккаунт можно добавить вручную.
    expect(await within(step).findByText("KQXG5C2M")).toBeInTheDocument();
    expect(within(step).getByRole("link", { name: /otpauth/ })).toHaveAttribute(
      "href",
      "otpauth://totp/CGA:admin%40sfedu.ru?secret=KQXG5C2M&issuer=CGA",
    );

    await user.type(within(step).getByLabelText("Код из приложения"), "123456");
    await user.click(within(step).getByRole("button", { name: "Подтвердить" }));

    expect(requestMock).toHaveBeenCalledWith("/api/auth/2fa/confirm", {
      method: "POST",
      body: { code: "123456", enroll_id: "enr-7" },
      anonymous: true,
    });

    // Recovery-коды видны ДО onLogin: сервер показывает их один раз,
    // и продолжить можно только явной кнопкой.
    const codesStep = await screen.findByRole("region", {
      name: "Настройка двухфакторной защиты",
    });
    expect(await within(codesStep).findByText("aa11bb22cc33dd44")).toBeInTheDocument();
    expect(within(codesStep).getByText("11ff22ee33dd44cc")).toBeInTheDocument();
    expect(onLogin).not.toHaveBeenCalled();

    await user.click(
      within(codesStep).getByRole("button", { name: "Я сохранил коды, продолжить" }),
    );

    await waitFor(() =>
      expect(onLogin).toHaveBeenCalledWith("tok-enroll", "admin", "Иван Петров"),
    );
  });
});
