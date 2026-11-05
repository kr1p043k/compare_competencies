/**
 * LoginPage: локальный вход и шаг 2FA. Сеть мокается на уровне lib/api.request,
 * реальный fetch не вызывается.
 */
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { LoginPage } from "./LoginPage";

vi.mock("../../lib/api", () => {
  class ApiError extends Error {}
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
    expect(screen.queryByLabelText("Код из приложения")).not.toBeInTheDocument();
    expect(onLogin).not.toHaveBeenCalled();
  });
});
