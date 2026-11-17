import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { AnalysisPanel } from "./AnalysisPanel";

// Сетевой слой подменён целиком: панель тянет /teacher/analysis через
// useApi(request), тест проверяет рендер payload, а не HTTP. ApiError
// остаётся настоящим - useApi сверяет instanceof на ошибочной ветке.
vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, request: vi.fn() };
});

/**
 * Срез дисциплины как в disc_data_out пайплайна. coverage_ratio - бинарное
 * покрытие (55%), weighted_coverage - качество (40%): в тесте оба, потом
 * без второго - проверка, что пара не рисуется из ничего.
 */
function analysisPayload(withQuality: boolean) {
  const metrics: Record<string, unknown> = {
    total_rpd_skills: 20,
    market_matched: 11,
    gaps: 9,
    coverage_ratio: 0.55,
    coverage_level: "medium",
    top_market_matched_skills: [
      { skill: "PostgreSQL", frequency: 120, match_type: "exact" },
    ],
    gaps_in_curriculum: ["OceanBase"],
    emerging_market_skills_not_in_rpd: [],
  };
  if (withQuality) metrics.weighted_coverage = 0.4;
  return {
    direction: "09.03.02",
    direction_name: "Информационные системы",
    discipline: "Базы данных",
    metrics,
    competencies: [],
    recommendations: [],
  };
}

beforeEach(() => {
  vi.mocked(request).mockReset();
});

describe("AnalysisPanel", () => {
  it("показывает качество рядом с покрытием в карточке покрытия", async () => {
    vi.mocked(request).mockResolvedValue(analysisPayload(true) as never);
    render(<AnalysisPanel disciplineName="Базы данных" />);

    // Бинарное покрытие осталось на месте (существующие тексты не удалены).
    const cov = await screen.findByText("55.0%");
    expect(cov).toBeInTheDocument();
    // Пара «кач.» при покрытии: значение качества из weighted_coverage.
    expect(screen.getByText("кач.: 40.0%")).toBeInTheDocument();
    expect(
      screen.getByTitle(
        "Качество совпадений: покрытие с учётом уверенности матчера, всегда не выше бинарного. Поля нет в payload - не выводится",
      ),
    ).toBeInTheDocument();
    // Отдельная карточка «Взвешенное покрытие» сохранена: значение 40.0%
    // живёт в ней, в паре оно внутри строки «кач.: 40.0%» (не точный match).
    expect(screen.getByText("Взвешенное покрытие")).toBeInTheDocument();
    expect(screen.getByText("40.0%")).toBeInTheDocument();
  });

  it("не рисует качество, если поля нет в payload", async () => {
    vi.mocked(request).mockResolvedValue(analysisPayload(false) as never);
    render(<AnalysisPanel disciplineName="Базы данных" />);

    await screen.findByText("55.0%");
    // Покрытию неоткуда брать качество: подпись «кач.» не появляется.
    expect(screen.queryByText(/кач\./)).not.toBeInTheDocument();
    expect(screen.queryByText("Взвешенное покрытие")).not.toBeInTheDocument();
  });
});
