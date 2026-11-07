import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { RecommendationsReport } from "./RecommendationsReport";

function makeData() {
  return {
    llm_enhanced: false,
    summary: {
      match_score: 44.2,
      // Дубль readiness/100: НЕ должен отображаться как отдельная метрика.
      confidence: 41.2,
      market_coverage_score: 46.4,
      skill_coverage: 58.9,
      domain_coverage_score: 33.3,
      readiness_score: 41.2,
      avg_gap: 54.9,
      coverage: 0.589,
      coverage_details: { covered_skills_count: 41, total_market_skills: 70 },
      market_skill_coverage: 28.1,
      coverage_strict: 31.4,
      coverage_weighted: 58.9,
      coverage_strict_scope: "топ-200 рынка",
    },
    closest_roles: [
      {
        role: "Data Engineer",
        semantic_similarity: 62.5,
        similarity_explanation: "",
        skills_covered: "12/18",
        cluster_level: "middle",
        coverage_percent: 40,
        coverage_explanation: "",
      },
    ],
    recommendations: [
      {
        rank: 1,
        skill: "Docker",
        importance_score: 0.87,
        importance_base: 0.62,
        importance_bonus: 0.25,
        priority: "HIGH",
        category: "missing",
        why_important: "Нужен для контейнеризации",
        how_to_learn: "Курс по Docker",
        expected_timeframe: "3 месяца",
        expected_outcome: "Соберёте CI/CD",
        is_soft_skill: false,
        market_frequency_percent: 43.7,
      },
    ],
    gaps: {
      docker: {
        skill: "Docker",
        gap_j: 1,
        gap_m: 0.8,
        gap_s: 0.6,
        demand_j: 0.7,
        demand_m: 0.75,
        demand_s: 0.8,
        cluster_relevance: 0.66,
        user_level: 0.1,
        importance: 0.8,
        category: "missing",
      },
    },
  };
}

describe("RecommendationsReport", () => {
  it("описывает составные проценты хинтами на StatCard", () => {
    render(<RecommendationsReport data={makeData()} />);
    expect(
      screen.getByText("рынок: 46% · средний разрыв: 54.9%"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("рынок 46% · навыки 59% · готовность 41%"),
    ).toBeInTheDocument();
    expect(screen.getByText("строгое: 28.1%")).toBeInTheDocument();
    expect(
      screen.getByText("Взвешенное по спросу: 58.9% · область: топ-200 рынка"),
    ).toBeInTheDocument();
  });

  it("не показывает summary.confidence отдельной метрикой", () => {
    render(<RecommendationsReport data={makeData()} />);
    expect(screen.queryByText(/уверенность/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/confidence/i)).not.toBeInTheDocument();
  });

  it("показывает market_frequency_percent и категорию в карточке навыка", () => {
    render(<RecommendationsReport data={makeData()} />);
    // Происхождение рекомендации: доля вакансий с навыком.
    expect(screen.getByText("43.7% вакансий")).toBeInTheDocument();
    // Категория навыка (нет/слабый/сильный) рядом с приоритетом.
    expect(screen.getAllByText("нет").length).toBeGreaterThan(0);
    expect(screen.getByText("Высокий")).toBeInTheDocument();
  });

  it("раскрывает происхождение разрыва: уровни, спрос, кластер", async () => {
    const user = userEvent.setup();
    render(<RecommendationsReport data={makeData()} />);

    await user.click(screen.getByText("подробнее"));
    expect(screen.getByText("gap_j: 100%")).toBeInTheDocument();
    expect(screen.getByText("gap_m: 80%")).toBeInTheDocument();
    expect(screen.getByText("gap_s: 60%")).toBeInTheDocument();
    expect(screen.getByText("demand_j: 70%")).toBeInTheDocument();
    expect(screen.getByText("user_level: 10%")).toBeInTheDocument();
    expect(screen.getByText("importance: 80%")).toBeInTheDocument();
    expect(screen.getByText("кластер: 66%")).toBeInTheDocument();
  });
});
