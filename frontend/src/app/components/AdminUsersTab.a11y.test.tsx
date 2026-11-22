/**
 * a11y: AdminUsersTab без нарушений axe после загрузки списка.
 *
 * Сеть мокается на уровне lib/api.request, авторизация - lib/auth.useAuth
 * (по образцу AdminUsersTab.test.tsx). DropdownMenu подменён простыми
 * кнопками: Radix-меню в jsdom зависает, а предмет теста - разметка таблицы.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, it, vi } from "vitest";

import { request } from "../../lib/api";
import { expectNoA11yViolations } from "../../test/a11y";
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

vi.mock("../../lib/auth", () => ({
  useAuth: () => ({ username: "chief@sfedu.ru" }),
}));

vi.mock("./ui/dropdown-menu", () => {
  const Box = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
  // role="menu" обязателен: иначе axe ругается aria-required-parent на menuitem,
  // и это была бы ошибка мока, а не компонента.
  const MenuBox = ({ children }: { children?: React.ReactNode }) => (
    <div role="menu">{children}</div>
  );
  return {
    DropdownMenu: Box,
    DropdownMenuContent: MenuBox,
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
  cleanup();
  window.location.hash = "";
});

const requestMock = vi.mocked(request);

const TEACHER = {
  id: "u-teacher",
  name: "Иван Петров",
  username: "ivan@sfedu.ru",
  role: "teacher",
  total_requests: 10,
  is_active: true,
  directions: [] as string[],
  totp_enabled: true,
  open_sessions: 1,
  last_activity: null,
};

const ADMIN = {
  id: "u-admin",
  name: "Мария Смирнова",
  username: "maria@sfedu.ru",
  role: "admin",
  total_requests: 10,
  is_active: true,
  directions: [] as string[],
  totp_enabled: true,
  open_sessions: 1,
  last_activity: null,
};

beforeEach(() => {
  requestMock.mockReset();
  requestMock.mockImplementation(((url: string) => {
    const u = String(url);
    if (u.startsWith("/api/admin/users?role=rop")) {
      return Promise.resolve({ users: [], total: 0 });
    }
    if (u.startsWith("/api/admin/users?")) {
      return Promise.resolve({ users: [TEACHER, ADMIN], total: 2, limit: 25, offset: 0 });
    }
    if (u.startsWith("/api/admin/directions")) return Promise.resolve([]);
    return Promise.reject(new Error(`unexpected endpoint: ${u}`));
  }) as never);
});

describe("AdminUsersTab (a11y)", () => {
  it("таблица пользователей и фильтры - без нарушений", async () => {
    const { container } = render(<AdminUsersTab />);
    await screen.findByText("ivan@sfedu.ru");

    await expectNoA11yViolations(container);
  });
});
