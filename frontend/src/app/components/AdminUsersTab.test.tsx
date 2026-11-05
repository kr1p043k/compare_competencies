/**
 * AdminUsersTab: список, поиск с debounce, диалог деактивации, шаг elevated
 * для действий над другим администратором (caller_password). Сеть мокается
 * на уровне lib/api.request, авторизация - lib/auth.useAuth.
 */
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, request } from "../../lib/api";
import { AdminUsersTab } from "./AdminUsersTab";

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

// Текущий админ - chief@sfedu.ru: цель-админ maria@sfedu.ru считается «другим».
vi.mock("../../lib/auth", () => ({
  useAuth: () => ({ username: "chief@sfedu.ru" }),
}));

/**
 * Radix DropdownMenu в jsdom зависает: пункт меню синхронно открывает Radix Dialog,
 * и фокус-ловушка вместе с pointer-capture уходит в бесконечный цикл (тест висит
 * без единого assertion). Меню здесь не является предметом теста, поэтому
 * подменяем его простыми кнопками: пункты видны сразу и вызывают onSelect.
 */
vi.mock("./ui/dropdown-menu", () => {
  const Box = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
  return {
    DropdownMenu: Box,
    DropdownMenuContent: Box,
    DropdownMenuSeparator: () => <hr />,
    DropdownMenuTrigger: ({
      children,
      ...rest
    }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
      <button type="button" {...rest}>
        {children}
      </button>
    ),
    DropdownMenuItem: ({
      children,
      onSelect,
    }: {
      children?: React.ReactNode;
      onSelect?: () => void;
    }) => (
      <button type="button" role="menuitem" onClick={() => onSelect?.()}>
        {children}
      </button>
    ),
  };
});

afterEach(() => {
  // Явная очистка: при singleFork автоочистка RTL не срабатывает между файлами.
  cleanup();
});

const requestMock = vi.mocked(request);

function makeUser(id: string, name: string, username: string, role: "admin" | "teacher") {
  return { id, name, username, role, total_requests: 10, is_active: true,
    directions: [] as string[], totp_enabled: true, open_sessions: 1, last_activity: null };
}

const TEACHER = makeUser("u-teacher", "Иван Петров", "ivan@sfedu.ru", "teacher");
const ADMIN = makeUser("u-admin", "Мария Смирнова", "maria@sfedu.ru", "admin");
const LIST = { users: [TEACHER, ADMIN], total: 2, limit: 25, offset: 0 };

function route(url: string, options?: { method?: string }): Promise<unknown> {
  if (url === "/api/admin/users/u-admin" && options?.method === "PATCH") {
    // Деактивация последнего админа: сервер отвечает 409 в теле detail.
    return Promise.reject(
      new ApiError(
        "Нельзя деактивировать последнего активного администратора",
        409,
        null,
        "/api/admin/users/u-admin",
      ),
    );
  }
  if (url === "/api/admin/users/u-teacher" && options?.method === "PATCH") {
    return Promise.resolve({ status: "ok" });
  }
  if (url.startsWith("/api/admin/users?role=rop")) return Promise.resolve({ users: [], total: 0 });
  if (url.startsWith("/api/admin/users?")) return Promise.resolve(LIST);
  if (url.startsWith("/api/admin/directions")) return Promise.resolve([]);
  return Promise.reject(new Error(`unexpected endpoint: ${url}`));
}

/** Запросы списка таблицы (справочник РОП не считаем). */
const listCalls = () =>
  requestMock.mock.calls
    .map(([url]) => String(url))
    .filter((url) => url.startsWith("/api/admin/users?") && !url.includes("role=rop"));

async function openDeactivateDialog(
  user: ReturnType<typeof userEvent.setup>,
  user2: typeof TEACHER,
) {
  // Меню подменено на всегда раскрытое: ищем пункт внутри строки нужного пользователя,
  // иначе пункты разных строк неразличимы.
  const row = screen.getByText(user2.username).closest("tr") as HTMLElement;
  await user.click(within(row).getByRole("menuitem", { name: "Деактивировать" }));
  return screen.findByRole("dialog", { name: "Деактивация пользователя" });
}

describe("AdminUsersTab", () => {
  beforeEach(() => {
    requestMock.mockReset();
    requestMock.mockImplementation(route as never);
  });

  it("рисует строки списка: имя, email и бейдж роли", async () => {
    render(<AdminUsersTab />);

    const teacherRow = (await screen.findByText("ivan@sfedu.ru")).closest("tr") as HTMLElement;
    expect(within(teacherRow).getByText("Иван Петров")).toBeInTheDocument();
    expect(within(teacherRow).getByText("Преподаватель")).toBeInTheDocument();

    const adminRow = screen.getByText("maria@sfedu.ru").closest("tr") as HTMLElement;
    expect(within(adminRow).getByText("Админ")).toBeInTheDocument();
  });

  it("поиск уходит на сервер один раз после паузы, с q=", async () => {
    const user = userEvent.setup();
    render(<AdminUsersTab />);
    await screen.findByText("ivan@sfedu.ru");
    expect(listCalls()).toHaveLength(1);

    // Набираем «ivan»: пока пауза debounce не вышла, новых запросов нет.
    // Fake timers не используем: user-event.type с ними зависает в этом vitest.
    await user.type(screen.getByLabelText("Поиск"), "ivan");
    expect(listCalls()).toHaveLength(1);

    // Ровно один повторный запрос (не по букве), и в нём q=ivan.
    await waitFor(() => expect(listCalls()).toHaveLength(2));
    expect(listCalls()[1]).toContain("q=ivan");
  });

  it("деактивация требует слова УДАЛИТЬ и отправляет PATCH с is_active=false", async () => {
    const user = userEvent.setup();
    render(<AdminUsersTab />);
    await screen.findByText("ivan@sfedu.ru");

    const dialog = await openDeactivateDialog(user, TEACHER);
    const confirm = within(dialog).getByRole("button", { name: "Деактивировать" });
    const field = within(dialog).getByLabelText("Введите слово УДАЛИТЬ для подтверждения");

    expect(confirm).toBeDisabled();
    await user.type(field, "удал");
    expect(confirm).toBeDisabled();
    await user.type(field, "ить");
    expect(confirm).toBeEnabled();

    await user.click(confirm);
    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/admin/users/u-teacher", {
        method: "PATCH",
        // UI не передаёт caller_password - тело PATCH ровно такое
        body: { is_active: false },
      }),
    );
    expect(within(dialog).getByRole("status")).toHaveTextContent("Пользователь деактивирован");
  });

  it("409 показывает сообщение сервера в диалоге и не закрывает его", async () => {
    const user = userEvent.setup();
    render(<AdminUsersTab />);
    await screen.findByText("maria@sfedu.ru");

    const dialog = await openDeactivateDialog(user, ADMIN);
    await user.type(within(dialog).getByLabelText("Введите слово УДАЛИТЬ для подтверждения"), "УДАЛИТЬ");
    // maria - другой админ: без слова И пароля кнопка не откроется
    await user.type(within(dialog).getByLabelText("Пароль администратора"), "chief-secret");
    await user.click(within(dialog).getByRole("button", { name: "Деактивировать" }));

    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      "Нельзя деактивировать последнего активного администратора",
    );
    expect(
      screen.getByRole("dialog", { name: "Деактивация пользователя" }),
    ).toBeInTheDocument();
  });

  it("деактивация другого админа требует слово И пароль, PATCH уходит с caller_password", async () => {
    const user = userEvent.setup();
    render(<AdminUsersTab />);
    await screen.findByText("maria@sfedu.ru");

    const dialog = await openDeactivateDialog(user, ADMIN);
    const confirm = within(dialog).getByRole("button", { name: "Деактивировать" });
    const password = within(dialog).getByLabelText("Пароль администратора");

    expect(confirm).toBeDisabled();
    await user.type(within(dialog).getByLabelText("Введите слово УДАЛИТЬ для подтверждения"), "УДАЛИТЬ");
    expect(confirm).toBeDisabled(); // слова мало - нужен ещё пароль текущего админа
    await user.type(password, "chief-secret");
    expect(confirm).toBeEnabled();

    await user.click(confirm);
    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith("/api/admin/users/u-admin", {
        method: "PATCH",
        body: { is_active: false, caller_password: "chief-secret" },
      }),
    );
  });
});
