/**
 * a11y: ArticlesPage без нарушений axe после загрузки трендов по профессиям.
 *
 * Сеть мокается на уровне app/api (обёртка api()); AnalyticsCharts
 * подменён заглушкой, у него свой a11y-тест. По образцу
 * ArticlesPage.test.tsx.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, it, vi } from "vitest";

import { api } from "../api";
import { expectNoA11yViolations } from "../../test/a11y";
import { ArticlesPage } from "./ArticlesPage";

vi.mock("../api", () => ({ api: vi.fn() }));
vi.mock("./AnalyticsCharts", () => ({
  AnalyticsCharts: () => <div data-testid="analytics-charts-stub" />,
}));

const apiMock = vi.mocked(api);

const STATS = {
  total: 120,
  by_experience: { junior: 10, middle: 20, senior: 30 },
  salary: { average: 150000, min: 30000, max: 400000, count: 60 },
};

const ANALYTICS = {
  salary_by_level: { junior: { count: 10, average: 90000 } },
  top_regions: [{ name: "Москва", count: 12 }],
  top_employers: [{ name: "Яндекс", count: 5 }],
  skills: { with_skills: 80, total: 120, percent: 66, avg_skills_per_vacancy: 5 },
};

const PROFESSIONS = [
  {
    name: "Аналитик данных",
    snapshot: "freq_profession_Аналитик данных_2026-11.json",
    date: "2026-11-02",
  },
];

afterEach(cleanup);

beforeEach(() => {
  apiMock.mockReset();
  apiMock.mockImplementation(((endpoint: string) => {
    if (endpoint.startsWith("/trends/professions")) {
      return Promise.resolve({ professions: PROFESSIONS });
    }
    if (endpoint.startsWith("/trends/by-profession")) {
      return Promise.resolve({
        profession: "Аналитик данных",
        source: "snapshot",
        snapshot_date: "2026-11-02",
        skills: [
          { skill: "SQL", frequency: 40 },
          { skill: "Python", frequency: 30 },
        ],
      });
    }
    if (endpoint.startsWith("/trends")) {
      return Promise.resolve({ trends: { rising: [], falling: [] } });
    }
    if (endpoint.startsWith("/vacancies/stats/summary")) return Promise.resolve(STATS);
    if (endpoint.startsWith("/vacancies/stats/analytics")) return Promise.resolve(ANALYTICS);
    if (endpoint.startsWith("/market/top-skills")) return Promise.resolve({ skills: [] });
    if (endpoint.startsWith("/taxonomy/coverage")) return Promise.resolve({ coverage: {} });
    return Promise.reject(new Error(`unexpected endpoint: ${endpoint}`));
  }) as never);
});

describe("ArticlesPage (a11y)", () => {
  it("тренды по профессиям - без нарушений", async () => {
    const { container } = render(<ArticlesPage />);
    await screen.findByText(/Данные по профессиям|Аналитик данных/);

    await expectNoA11yViolations(container);
  });
});
