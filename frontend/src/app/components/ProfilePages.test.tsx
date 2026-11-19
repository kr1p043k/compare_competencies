/**
 * ProfilePages: smoke-покрытие (7.5.4). Мокаем lib/api; проверяем состояния
 * загрузки/пусто/ошибка/список и навигацию. TwoFactorCard дёргает 2FA-status -
 * отдаём нейтральный дефолт.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { AdminProfilePage, TeacherProfilePage } from "./ProfilePages";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../lib/errors", () => ({
  reportError: vi.fn(),
  errMessage: (e: unknown) => (e instanceof Error ? e.message : String(e ?? "")),
}));
// Изолируем тяжёлые дочерние карточки: у них свои сетевые вызовы и состояния,
// здесь проверяем только ProfilePages.
vi.mock("./TwoFactorCard", () => ({ TwoFactorCard: () => null }));
vi.mock("./AdminSecurity", () => ({ AdminSecurity: () => null }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const requestMock = vi.mocked(request);

function route(handlers: Record<string, () => unknown>) {
  requestMock.mockImplementation(async (url: string) => {
    for (const [prefix, fn] of Object.entries(handlers)) {
      if (url.startsWith(prefix)) return fn() as never;
    }
    return undefined as never;
  });
}

const TWoFA = { enabled: false, pending: false, recovery_codes_left: 0 };

describe("TeacherProfilePage", () => {
  it("рендерит дисциплины с бейджами", async () => {
    route({
      "/api/auth/2fa/status": () => TWoFA,
      "/api/teacher/krm/disciplines": () => [
        { name: "Веб-технологии", semester: 3, competencies_count: 4, skills_count: 12 },
      ],
    });
    render(<TeacherProfilePage displayName="Иван" email="t@sfedu.ru" />);
    expect(screen.getByText("Мой профиль")).toBeTruthy();
    await waitFor(() => expect(screen.getByText("Веб-технологии")).toBeTruthy());
    expect(screen.getByText("компетенций: 4")).toBeTruthy();
    expect(screen.getByText("навыков: 12")).toBeTruthy();
  });

  it("пустой список -> EmptyState", async () => {
    route({ "/api/auth/2fa/status": () => TWoFA, "/api/teacher/krm/disciplines": () => [] });
    render(<TeacherProfilePage displayName="Иван" email="t@sfedu.ru" />);
    await waitFor(() => expect(screen.getByText("Дисциплины не найдены")).toBeTruthy());
  });

  it("ошибка загрузки -> сообщение", async () => {
    route({
      "/api/auth/2fa/status": () => TWoFA,
      "/api/teacher/krm/disciplines": () => {
        throw new Error("boom");
      },
    });
    render(<TeacherProfilePage displayName="Иван" email="t@sfedu.ru" />);
    const alert = await waitFor(() => screen.getByRole("alert"));
    expect(alert.textContent).toMatch(/boom/);
  });

  it("навигация: кнопка «Таксономия» зовёт onNavigate", async () => {
    route({ "/api/auth/2fa/status": () => TWoFA, "/api/teacher/krm/disciplines": () => [] });
    const onNavigate = vi.fn();
    render(<TeacherProfilePage displayName="Иван" email="t@sfedu.ru" onNavigate={onNavigate} />);
    fireEvent.click(screen.getByText("Таксономия"));
    expect(onNavigate).toHaveBeenCalledWith("taxonomy");
  });
});

describe("AdminProfilePage", () => {
  it("рендерит живую сводку", async () => {
    route({
      "/api/auth/2fa/status": () => TWoFA,
      "/api/admin/monitoring/summary": () => ({
        sessions_active: 3,
        errors_1h: { "4xx": 2, "5xx": 1 },
        freshness: { vacancies: 1234, date_from: "2026-01-01", date_to: "2026-02-01" },
      }),
    });
    render(<AdminProfilePage displayName="Админ" email="a@sfedu.ru" onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Сессии активны")).toBeTruthy());
    expect(screen.getByText("Ошибки за час")).toBeTruthy();
    await waitFor(() => expect(screen.getByText(/1\s*234/u)).toBeTruthy());
  });

  it("навигация: «Пользователи» зовёт onNavigate('admin')", async () => {
    route({
      "/api/auth/2fa/status": () => TWoFA,
      "/api/admin/monitoring/summary": () => ({ sessions_active: 0, errors_1h: {}, freshness: {} }),
    });
    const onNavigate = vi.fn();
    render(<AdminProfilePage displayName="Админ" email="a@sfedu.ru" onNavigate={onNavigate} />);
    fireEvent.click(screen.getByText("Пользователи"));
    expect(onNavigate).toHaveBeenCalledWith("admin");
  });
});
