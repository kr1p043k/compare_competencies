/**
 * a11y: SummaryReport без нарушений axe. Компонент презентационный:
 * данные приходят пропсом, точных запросов нет.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, it } from "vitest";

import { expectNoA11yViolations } from "../../test/a11y";
import { SummaryReport } from "./SummaryReport";

const fullGap = {
  skill: "Docker",
  gap_j: 1,
  gap_m: 0.8,
  gap_s: 0.6,
  demand_j: 0.7,
  demand_m: 0.75,
  demand_s: 0.8,
  user_level: 0.1,
  cluster_relevance: 0.66,
  importance: 0.8,
  category: "missing",
};

const slimGap = {
  skill: "Windows Server",
  gap_j: 0.9,
  importance: 0.4,
  category: "missing",
};

const data = {
  profiles: ["Python-разработчик", "Go-разработчик"],
  evaluations: {
    "Python-разработчик": {
      target_profession: "Data Engineer",
      dominant_domain_name: "Backend",
      market_coverage_score: 46.4,
      skill_coverage: 58.9,
      readiness_score: 41.2,
      domain_coverage_score: 33.3,
      market_skill_coverage: 28.1,
      coverage_weighted: 58.9,
      coverage_strict_scope: "топ-200 рынка",
      avg_gap: 54.9,
      closest_roles: [{ role: "Data Engineer", semantic_similarity: 62.5 }],
      gaps: { docker: fullGap, "windows-admin": slimGap },
    },
    "Go-разработчик": {
      target_profession: "Backend-разработчик",
      market_coverage_score: 51.1,
      gaps: {},
    },
  },
};

afterEach(cleanup);

describe("SummaryReport (a11y)", () => {
  it("сводка с раскрытым профилем - без нарушений", async () => {
    const user = userEvent.setup();
    const { container } = render(<SummaryReport data={data} />);

    await user.click(screen.getByRole("button", { name: /Python-разработчик/ }));
    await screen.findByText(/Docker/);

    await expectNoA11yViolations(container);
  });
});
