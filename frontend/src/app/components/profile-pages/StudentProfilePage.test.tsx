/**
 * StudentProfilePage (7.5.4): контейнер профиля студента.
 * Проверяем загрузку (профиль + направления + КРМ), KPI, ошибку с retry,
 * оптимистичное добавление/удаление навыков и технологий с откатом,
 * направление КРМ берётся из /api/krm/directions.
 * Тяжёлые карточки MyProfilesCard/TwoFactorCard замоканы (свои тесты есть).
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../../lib/api";
import { useAuth } from "../../../lib/auth";
import { StudentProfilePage } from "./StudentProfilePage";

vi.mock("../../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../../lib/auth", () => ({ useAuth: vi.fn() }));
vi.mock("../../../lib/errors", () => ({
  errMessage: (e: unknown) => (e instanceof Error ? e.message : String(e ?? "")),
  reportError: vi.fn(),
}));
vi.mock("../TwoFactorCard", () => ({ TwoFactorCard: () => <div data-testid="2fa" /> }));
vi.mock("./student/MyProfilesCard", () => ({
  MyProfilesCard: () => <div data-testid="my-profiles" />,
}));

const SELF = {
  profile: "backend",
  target_level: "middle",
  skills: ["Python", "SQL"],
  user_added: ["Python"],
  technologies: ["Docker"],
};

const KRM = {
  direction: "09.03.04",
  direction_name: "Программная инженерия",
  profile: "backend",
  counts: { krm: 12, student: 3, overlap: 2, merged: 13, student_only: 1 },
  disciplines: [
    {
      discipline: "Базы данных",
      competencies: [
        {
          code: "ПК-1",
          skills: [
            { text: "SQL", student_has: true },
            { text: "Индексы", student_has: false },
          ],
        },
      ],
    },
  ],
};

beforeEach(() => {
  vi.mocked(useAuth).mockReturnValue({ name: "Иван", username: "ivan@sfedu.ru" } as never);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const requestMock = vi.mocked(request);

function route(handlers: Record<string, unknown>) {
  requestMock.mockImplementation(async (url: string) => {
    for (const [key, value] of Object.entries(handlers)) {
      if (url.startsWith(key)) {
        if (value instanceof Error) throw value;
        return typeof value === "function" ? (value as () => unknown)() : value;
      }
    }
    return {} as never;
  });
}

const OK = {
  "/api/profiles/custom/options": {
    competencies: [{ code: "ПК-1", title: "БД", skills_count: 2 }],
    technologies_suggest: ["Kafka", "Docker"],
  },
  "/api/profiles/self": SELF,
  "/api/krm/directions": { directions: [{ dir_code: "09.03.04" }] },
  "/api/krm/student/competencies": KRM,
  "/api/profiles": { profiles: ["base", "backend"] },
};

describe("StudentProfilePage", () => {
  it("загружает профиль, направление и КРМ, показывает KPI", async () => {
    route(OK);
    render(<StudentProfilePage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Иван")).toBeTruthy());
    expect(screen.getByText(/ivan@sfedu\.ru · уровень middle · профиль backend/)).toBeTruthy();
    await waitFor(() => expect(screen.getByText("Компетенции КРМ")).toBeTruthy());
    // KPI: навыки 2, компетенций КРМ 12, совпало 2, дисциплин 1
    expect(screen.getByText("12")).toBeTruthy();
    expect(screen.getByText("Мои навыки (2)")).toBeTruthy();
    // Направление взято из /api/krm/directions.
    expect(requestMock).toHaveBeenCalledWith(
      expect.stringContaining("direction=09.03.04"),
    );
  });

  it("ошибка загрузки профиля -> ErrorState и retry перезапрашивает", async () => {
    route({ ...OK, "/api/profiles/self": new Error("Профиль недоступен") });
    render(<StudentProfilePage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getAllByText("Профиль недоступен").length).toBeGreaterThan(0));
    const before = requestMock.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: /Повтор|Retry|Попроб/i }));
    await waitFor(() => expect(requestMock.mock.calls.length).toBeGreaterThan(before));
  });

  it("ошибка КРМ не ломает страницу, карточка показывает retry", async () => {
    route({ ...OK, "/api/krm/student/competencies": new Error("КРМ недоступна") });
    render(<StudentProfilePage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("КРМ недоступна")).toBeTruthy());
    expect(screen.getByText("Мои навыки (2)")).toBeTruthy();
  });

  it("добавление навыка: оптимистично, PATCH и перезагрузка КРМ", async () => {
    const updated = { ...SELF, skills: ["Python", "SQL", "Docker"] };
    route({
      ...OK,
      "/api/profiles/self": (() => {
        let n = 0;
        return () => (n++ === 0 ? SELF : updated);
      })(),
    });
    render(<StudentProfilePage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getByLabelText("Новый навык")).toBeTruthy());
    fireEvent.change(screen.getByLabelText("Новый навык"), { target: { value: "Docker" } });
    fireEvent.click(screen.getByRole("button", { name: "Добавить навык" }));
    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith(
        "/api/profiles/self",
        expect.objectContaining({ method: "PATCH", body: { add_skills: ["Docker"] } }),
      ),
    );
    // После PATCH КРМ перезагружается ещё раз.
    await waitFor(() => {
      const krmCalls = requestMock.mock.calls.filter(([u]) =>
        String(u).startsWith("/api/krm/student/competencies"),
      );
      expect(krmCalls.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("ошибка PATCH навыка откатывает состояние и показывает текст", async () => {
    route({
      ...OK,
      "/api/profiles/self": (() => {
        let n = 0;
        return () => {
          if (n++ === 0) return SELF;
          throw new Error("Не удалось сохранить навык");
        };
      })(),
    });
    render(<StudentProfilePage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getByLabelText("Новый навык")).toBeTruthy());
    fireEvent.change(screen.getByLabelText("Новый навык"), { target: { value: "Kafka" } });
    fireEvent.click(screen.getByRole("button", { name: "Добавить навык" }));
    await waitFor(() => expect(screen.getByText("Не удалось сохранить навык")).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Удалить Kafka" })).toBeNull();
  });

  it("добавление технологии: PATCH add_technologies и подсказки без уже добавленных", async () => {
    route(OK);
    render(<StudentProfilePage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Мои технологии (1)")).toBeTruthy());
    // Docker уже есть -> подсказкой остаётся только Kafka.
    expect(screen.getByRole("button", { name: "Добавить Kafka" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Добавить Docker" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Добавить Kafka" }));
    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith(
        "/api/profiles/self",
        expect.objectContaining({ method: "PATCH", body: { add_technologies: ["Kafka"] } }),
      ),
    );
  });

  it("удаление технологии шлёт remove_technologies", async () => {
    route(OK);
    render(<StudentProfilePage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Убрать Docker" })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "Убрать Docker" }));
    await waitFor(() =>
      expect(requestMock).toHaveBeenCalledWith(
        "/api/profiles/self",
        expect.objectContaining({ method: "PATCH", body: { remove_technologies: ["Docker"] } }),
      ),
    );
  });

  it("монтирует карточки своих профилей и 2FA", async () => {
    route(OK);
    render(<StudentProfilePage onNavigate={vi.fn()} />);
    expect(screen.getByTestId("my-profiles")).toBeTruthy();
    expect(screen.getByTestId("2fa")).toBeTruthy();
  });
});