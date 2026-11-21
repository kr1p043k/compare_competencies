/**
 * ScientificTrendsTab (7.5.4): пустое состояние, валидация ввода, успешные
 * тренды и анализ разрыва, ошибки API, ограничение для роли student.
 * Сеть мокается на уровне lib/api (request), авторизация - useAuth.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import { ScientificTrendsTab } from "./ScientificTrendsTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));
vi.mock("../../lib/auth", () => ({ useAuth: vi.fn() }));
vi.mock("../../lib/errors", () => ({
  reportError: vi.fn(),
  errMessage: (e: unknown) => (e instanceof Error ? e.message : String(e ?? "")),
}));

const TREND = {
  topic: "нейросети",
  found_trends: ['{title: "Глубокое обучение", summary: "рост", keywords: [cnn, rnn]}'],
  trends_source: "market_skills",
  recommended_competencies: [
    {
      code: "ПК-1",
      description: "Машинное обучение",
      keywords: "нейросети",
      trend_source: "",
      coverage_percent: 35,
      status: "gap",
    },
  ],
  rationale: "Тема востребована",
};

const GAP = {
  overall_score: 0.42,
  summary: "Покрытие частичное",
  detailed_analysis: [
    {
      code: "ПК-2",
      status: "gap",
      coverage_percent: 20,
      reason: "нет дисциплин",
      recommendation: "добавить курс",
    },
  ],
};

function rawResponse(payload: unknown, ok = true, status = 200) {
  return { ok, status, json: async () => payload };
}

beforeEach(() => {
  vi.mocked(useAuth).mockReturnValue({ role: "teacher" } as never);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const requestMock = vi.mocked(request);
const text = () => document.body.textContent ?? "";

describe("ScientificTrendsTab", () => {
  it("по умолчанию показывает пустое состояние «Задайте тему»", () => {
    render(<ScientificTrendsTab />);
    expect(screen.getByText("Научные тренды")).toBeTruthy();
    expect(screen.getByText("Задайте тему")).toBeTruthy();
  });

  it("пустая тема -> ошибка валидации без сети", async () => {
    render(<ScientificTrendsTab />);
    fireEvent.click(screen.getByRole("button", { name: /Рекомендуемые компетенции/ }));
    await waitFor(() => expect(screen.getByText("Введите тему для поиска")).toBeTruthy());
    expect(requestMock).not.toHaveBeenCalled();
  });

  it("успешный поиск трендов рисует компетенции", async () => {
    requestMock.mockResolvedValue(rawResponse(TREND) as never);
    render(<ScientificTrendsTab />);
    fireEvent.change(screen.getByPlaceholderText(/нейросетевые методы/), {
      target: { value: "нейросети" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Рекомендуемые компетенции/ }));
    await waitFor(() => expect(screen.getByText("Тема: нейросети")).toBeTruthy());
    expect(requestMock).toHaveBeenCalledWith(
      "/api/academic/get-competencies",
      expect.objectContaining({ method: "POST", raw: true }),
    );
    expect(screen.getByText("ПК-1")).toBeTruthy();
    expect(screen.getByText("Тема востребована", { exact: false })).toBeTruthy();
  });

  it("ошибка академического API показывается в ErrorState", async () => {
    requestMock.mockResolvedValue(rawResponse({ detail: "сервис недоступен" }, false, 503) as never);
    render(<ScientificTrendsTab />);
    fireEvent.change(screen.getByPlaceholderText(/нейросетевые методы/), {
      target: { value: "тема" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Рекомендуемые компетенции/ }));
    await waitFor(() => expect(screen.getByText("сервис недоступен")).toBeTruthy());
  });

  it("анализ разрыва: сперва КРМ, затем отчёт с покрытием", async () => {
    requestMock.mockImplementation(async (url: string) => {
      if (url === "/api/academic/krm-competencies") {
        return { codes: ["ПК-1", "ПК-2"], count: 2 } as never;
      }
      return rawResponse(GAP) as never;
    });
    render(<ScientificTrendsTab />);
    fireEvent.change(screen.getByPlaceholderText(/нейросетевые методы/), {
      target: { value: "нейросети" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Анализ разрыва/ }));
    await waitFor(() => expect(screen.getByText("42%")).toBeTruthy());
    expect(screen.getByText("Анализ разрыва компетенций")).toBeTruthy();
    expect(text()).toContain("учтено компетенций КРМ");
    expect(screen.getByText(/добавить курс/)).toBeTruthy();
  });

  it("студенту анализ разрыва недоступен", () => {
    vi.mocked(useAuth).mockReturnValue({ role: "student" } as never);
    render(<ScientificTrendsTab />);
    const btn = screen.getByRole("button", { name: /Анализ разрыва/ }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.title).toMatch(/доступен преподавателям/);
  });
});