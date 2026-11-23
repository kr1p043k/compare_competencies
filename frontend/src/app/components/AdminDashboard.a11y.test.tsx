/**
 * a11y: AdminDashboard без нарушений axe на вкладках по умолчанию и «Логи».
 *
 * Сеть мокается на уровне lib/api.request, lib/auth мокается целиком
 * (по образцу AdminDashboard.coverage.test.tsx). Дочерние экраны
 * (AdminUsersTab, AdminAuditTab и др.) имеют свои тесты, здесь важна
 * оболочка и таблица логов.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, it, vi } from "vitest";

import { request } from "../../lib/api";
import { expectNoA11yViolations } from "../../test/a11y";
import { AdminDashboard } from "./AdminDashboard";

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
  useAuth: () => ({
    username: "chief@sfedu.ru",
    role: "admin",
    name: "Начальник",
    authed: true,
    isAuth: true,
  }),
  logAction: vi.fn(),
}));

vi.mock("./ui/dropdown-menu", () => {
  const Box = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
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

const requestMock = vi.mocked(request);

const USERS = [
  {
    id: "u-admin",
    name: "Мария Смирнова",
    username: "admin@sfedu.ru",
    role: "admin",
    total_requests: 10,
    is_active: true,
    directions: [] as string[],
    totp_enabled: true,
    open_sessions: 1,
    last_activity: null,
  },
];

const LOGS = [
  {
    timestamp: "2026-11-18T10:00:00Z",
    user: "ivan@sfedu.ru",
    method: "GET",
    path: "/api/vacancies",
    detail: null,
    status: 200,
    duration_ms: 1234.9,
  },
];

const SCHED = {
  collector_enabled: true,
  daily_gap_enabled: false,
  collect_interval_hours: 12,
  last_collect_ts: 1700000000,
  last_gap_date: "2026-11-17",
  next_collect_ts: 1700043200,
  busy: null,
  pipeline_running: true,
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

beforeEach(() => {
  requestMock.mockReset();
  requestMock.mockImplementation(((url: string) => {
    const u = String(url);
    if (u === "/api/admin/users") return Promise.resolve({ users: USERS });
    if (u.startsWith("/api/admin/users?")) {
      if (u.includes("role=rop")) return Promise.resolve({ users: [], total: 0 });
      return Promise.resolve({ users: USERS, total: USERS.length, limit: 25, offset: 0 });
    }
    if (u === "/api/admin/logs?limit=200") return Promise.resolve({ logs: LOGS });
    if (u === "/api/admin/directions") return Promise.resolve([]);
    if (u === "/api/teacher/rpd/sources") return Promise.resolve({ yandex_covered: [] });
    if (u === "/api/admin/scheduler/status") return Promise.resolve(SCHED);
    if (u === "/api/admin/skills/suggestions?status=pending") {
      return Promise.resolve({ suggestions: [] });
    }
    if (u === "/api/taxonomy/categories") return Promise.resolve({ categories: [] });
    return Promise.reject(new Error(`unexpected endpoint: ${u}`));
  }) as never);
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response("{}", { status: 200 }))),
  );
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("AdminDashboard (a11y)", () => {
  it("вкладка по умолчанию - без нарушений", async () => {
    const { container } = render(<AdminDashboard />);
    await screen.findByRole("tab", { name: "Логи" });

    await expectNoA11yViolations(container);
  });

  it("вкладка «Логи» с таблицей - без нарушений", async () => {
    const user = userEvent.setup();
    const { container } = render(<AdminDashboard />);
    const tab = await screen.findByRole("tab", { name: "Логи" });

    await user.click(tab);
    await screen.findByText("/api/vacancies");

    await expectNoA11yViolations(container);
  });
});
