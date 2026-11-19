/**
 * AdminDashboard: оболочка админки (шапка, девять вкладок) и собственные
 * ветки вкладок «Логи», «БД», «Фоновые задачи», «Навыки» плюс ветка ошибки
 * загрузки. Дочерние экраны (AdminUsersTab, AdminAuditTab, AdminSecurity,
 * AdminLlmCostCard) подключены как есть - у них свои тесты.
 *
 * Сеть мокается на уровне lib/api.request; fetch подменён заглушкой, чтобы
 * logAction/reportError не уходили в /api/log. lib/auth мокается целиком:
 * в дашборде нужен logAction, в AdminUsersTab - useAuth, а вне AuthProvider
 * реальный контекст даёт username = null.
 */
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
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

const requestMock = vi.mocked(request);

function makeUser(id: string, name: string, username: string, role: "admin" | "teacher") {
  return {
    id,
    name,
    username,
    role,
    total_requests: 10,
    is_active: true,
    directions: [] as string[],
    totp_enabled: true,
    open_sessions: 1,
    last_activity: null,
  };
}

const USERS = [
  makeUser("u-admin", "Мария Смирнова", "admin@sfedu.ru", "admin"),
  makeUser("u-teacher", "Иван Петров", "ivan@sfedu.ru", "teacher"),
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
  {
    timestamp: "2026-11-18T10:01:00Z",
    user: "admin@sfedu.ru",
    method: "POST",
    path: "/api/admin/db/seed",
    detail: null,
    status: 500,
    duration_ms: 3.2,
  },
];

const DIRECTIONS = [{ dir_code: "09.03.02" }, { dir_code: "09.04.01" }];

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

function route(url: string): Promise<unknown> {
  if (url === "/api/admin/users") return Promise.resolve({ users: USERS });
  // AdminUsersTab: список, справочник РОП и фильтры.
  if (url.startsWith("/api/admin/users?")) {
    if (url.includes("role=rop")) return Promise.resolve({ users: [], total: 0 });
    return Promise.resolve({ users: USERS, total: USERS.length, limit: 25, offset: 0 });
  }
  if (url === "/api/admin/logs?limit=200") return Promise.resolve({ logs: LOGS });
  if (url === "/api/admin/directions") return Promise.resolve(DIRECTIONS);
  if (url === "/api/teacher/rpd/sources") {
    return Promise.resolve({ yandex_covered: ["09.03.02"] });
  }
  if (url === "/api/admin/scheduler/status") return Promise.resolve(SCHED);
  if (url === "/api/admin/skills/suggestions?status=pending") {
    return Promise.resolve({ suggestions: [] });
  }
  if (url === "/api/taxonomy/categories") {
    return Promise.resolve({ categories: [] });
  }
  return Promise.reject(new Error(`unexpected endpoint: ${url}`));
}

/** Вкладки в порядке отрисовки (src: AdminDashboard.tsx:481-489). */
const TAB_LABELS = [
  "Пользователи",
  "Логи",
  "БД",
  "Импорт",
  "Навыки",
  "Фоновые задачи",
  "Проверка",
  "Аудит",
  "Безопасность",
];

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

beforeEach(() => {
  requestMock.mockReset();
  requestMock.mockImplementation(route as never);
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response("{}", { status: 200 }))),
  );
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("AdminDashboard", () => {
  it("рисует шапку и все девять вкладок", async () => {
    render(<AdminDashboard />);

    expect(
      screen.getByRole("heading", { level: 1, name: "Администрирование" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Управление системой")).toBeInTheDocument();

    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(TAB_LABELS.length);
    for (const label of TAB_LABELS) {
      expect(screen.getByRole("tab", { name: label })).toBeInTheDocument();
    }

    // Данные админки запрошены при монтировании.
    await waitFor(() => {
      expect(requestMock).toHaveBeenCalledWith("/api/admin/users");
      expect(requestMock).toHaveBeenCalledWith("/api/admin/logs?limit=200");
      expect(requestMock).toHaveBeenCalledWith("/api/admin/directions");
    });
  });

  it("вкладка «Логи»: история запросов с фильтром по пользователю", async () => {
    const user = userEvent.setup();
    render(<AdminDashboard />);

    await user.click(screen.getByRole("tab", { name: "Логи" }));
    expect(await screen.findByText("История запросов")).toBeInTheDocument();

    const teacherRow = screen.getByText("/api/vacancies").closest("tr") as HTMLElement;
    expect(within(teacherRow).getByText("ivan@sfedu.ru")).toBeInTheDocument();
    expect(within(teacherRow).getByText("GET")).toBeInTheDocument();
    expect(within(teacherRow).getByText("200")).toBeInTheDocument();
    expect(within(teacherRow).getByText("1235")).toBeInTheDocument();

    const adminRow = screen.getByText("/api/admin/db/seed").closest("tr") as HTMLElement;
    expect(within(adminRow).getByText("POST")).toBeInTheDocument();
    expect(within(adminRow).getByText("500")).toBeInTheDocument();

    // Фильтр «Все пользователи» -> конкретный пользователь.
    const filter = screen.getByRole("combobox");
    expect(within(filter).getByRole("option", { name: "Мария Смирнова" })).toBeInTheDocument();
    await user.selectOptions(filter, "admin@sfedu.ru");

    expect(screen.queryByText("/api/vacancies")).toBeNull();
    expect(screen.getByText("/api/admin/db/seed")).toBeInTheDocument();
  });

  it("вкладка «БД»: действия с базой и подгрузка РПД направления", async () => {
    const user = userEvent.setup();
    render(<AdminDashboard />);

    await user.click(screen.getByRole("tab", { name: "БД" }));

    expect(await screen.findByText("Управление БД")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Заполнить БД" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Drop + Seed" })).toBeEnabled();
    expect(screen.getByText("Подгрузка РПД")).toBeInTheDocument();

    // Направления приходят из /api/admin/directions, выбран 09.03.02.
    const dirSelect = screen.getByRole("combobox");
    expect(dirSelect).toHaveValue("09.03.02");
    expect(within(dirSelect).getByRole("option", { name: "09.04.01" })).toBeInTheDocument();

    // yandex_covered содержит 09.03.02 -> кнопка сбора доступна.
    expect(
      await screen.findByRole("button", { name: "Собрать с Yandex Disk" }),
    ).toBeEnabled();
  });

  it("вкладка «Фоновые задачи»: статус планировщика и ручные запуски", async () => {
    const user = userEvent.setup();
    render(<AdminDashboard />);

    await user.click(screen.getByRole("tab", { name: "Фоновые задачи" }));

    expect(await screen.findByText("Планировщик")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Сбор: включён" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ночной gap: выключен" })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton")).toHaveValue(12);

    expect(screen.getByText(/Последний сбор:/)).toHaveTextContent("2026-11-17");
    expect(screen.getByText(/Следующий сбор:/)).toHaveTextContent("Конвейер активен: да");

    expect(screen.getByText("Запуск вне расписания")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Собрать сейчас" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Gap сейчас" })).toBeInTheDocument();
  });

  it("вкладка «Навыки»: пустые очереди предложений и некатегоризованных", async () => {
    const user = userEvent.setup();
    render(<AdminDashboard />);

    await user.click(screen.getByRole("tab", { name: "Навыки" }));

    expect(await screen.findByText("Предложения от преподавателей")).toBeInTheDocument();
    expect(await screen.findByText("Очередь пуста.")).toBeInTheDocument();
    expect(await screen.findByText("Все навыки категоризованы.")).toBeInTheDocument();
    expect(screen.getByText("Расширение таксономии")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Анализировать и добавить" })).toBeInTheDocument();
  });

  it("ошибка загрузки данных показывается в callout с ролью alert", async () => {
    requestMock.mockImplementation(((url: string) =>
      url === "/api/admin/users"
        ? Promise.reject(new Error("Сервер недоступен"))
        : route(url)) as never);

    render(<AdminDashboard />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Сервер недоступен");
    // Оболочка остаётся рабочей: вкладки на месте.
    expect(screen.getAllByRole("tab")).toHaveLength(TAB_LABELS.length);
  });
});
