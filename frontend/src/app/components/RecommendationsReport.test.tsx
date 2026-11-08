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
    // Готовность: рядом только средний разрыв, без дубля «рынка»
    // (рынок - значение соседней карточки, а не формула готовности).
    expect(
      screen.getByText("средний разрыв по навыкам: 54.9%"),
    ).toBeInTheDocument();
    // Соответствие рынку: расшифровка состава вместо повтора чужих чисел.
    expect(
      screen.getByText("среднее трёх метрик: покрытие · навыки · готовность"),
    ).toBeInTheDocument();
    // Покрытие рынка: состав формулы, без повторов значений других карточек.
    expect(
      screen.getByText("складывается из навыков (60%) и доменов (40%)"),
    ).toBeInTheDocument();
    // Навыки профиля: область сравнения + взвешенная пара в одной подписи.
    expect(
      screen.getByText("топ-200 рынка · взвешенное по спросу: 58.9%"),
    ).toBeInTheDocument();
  });

  it("заменяет сырой scope «market» на человекочитаемую область", () => {
    const data = makeData();
    data.summary.coverage_strict_scope = "market";
    render(<RecommendationsReport data={data} />);
    expect(
      screen.getByText("по всему рынку · взвешенное по спросу: 58.9%"),
    ).toBeInTheDocument();
    // Сырой ключ из API больше не попадает в интерфейс.
    expect(screen.queryByText(/область:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/область: market/)).not.toBeInTheDocument();
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
    // Человекочитаемые подписи вместо сырых ключей gap_j / demand_j / ...
    expect(screen.getByText("Разрыв (J/M/S)")).toBeInTheDocument();
    expect(screen.getByText("Спрос (J/M/S)")).toBeInTheDocument();
    expect(screen.getByText("Ваш уровень")).toBeInTheDocument();
    expect(screen.getByText("Важность")).toBeInTheDocument();
    expect(screen.getByText("Кластер")).toBeInTheDocument();
    // Те же значения, что и раньше; доступное имя бара = tooltip «что это».
    expect(
      screen.getByRole("meter", { name: "Разрыв на уровне Junior: насколько не хватает до требуемого" }),
    ).toHaveTextContent("100%");
    expect(
      screen.getByRole("meter", { name: "Разрыв на уровне Middle: насколько не хватает до требуемого" }),
    ).toHaveTextContent("80%");
    expect(
      screen.getByRole("meter", { name: "Разрыв на уровне Senior: насколько не хватает до требуемого" }),
    ).toHaveTextContent("60%");
    expect(
      screen.getByRole("meter", { name: "Востребованность на рынке, уровень Junior" }),
    ).toHaveTextContent("70%");
    expect(
      screen.getByRole("meter", { name: "Ваш текущий уровень владения навыком" }),
    ).toHaveTextContent("10%");
    expect(
      screen.getByRole("meter", { name: "Общая важность навыка" }),
    ).toHaveTextContent("80%");
    expect(
      screen.getByRole("meter", { name: "Релевантность навыка кластеру целевых вакансий" }),
    ).toHaveTextContent("66%");
  });
});
