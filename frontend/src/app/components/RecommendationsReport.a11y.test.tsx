/**
 * a11y: RecommendationsReport без нарушений axe. Компонент презентационный:
 * данные приходят пропсом, точных запросов нет.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, it } from "vitest";

import { expectNoA11yViolations } from "../../test/a11y";
import { RecommendationsReport } from "./RecommendationsReport";

const data = {
  llm_enhanced: false,
  summary: {
    match_score: 44.2,
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

afterEach(cleanup);

describe("RecommendationsReport (a11y)", () => {
  it("сводка и рекомендации - без нарушений", async () => {
    const { container } = render(<RecommendationsReport data={data} />);

    await expectNoA11yViolations(container);
  });
});
