/**
 * a11y: TeacherDashboard без нарушений axe после загрузки анализа направления.
 *
 * Сеть мокается на уровне app/api (обёртка api()), LlmChatPanel подменён
 * заглушкой (у него свой тест), reportError замокан, чтобы не ходить
 * в /api/log. По образцу TeacherDashboard.test.tsx.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, it, vi } from "vitest";

import { api } from "../api";
import { expectNoA11yViolations } from "../../test/a11y";
import { TeacherDashboard } from "./TeacherDashboard";

vi.mock("../api", () => ({
  api: vi.fn(() => Promise.reject(new Error("api mock not configured"))),
}));

vi.mock("./LlmChatPanel", () => ({ LlmChatPanel: () => null }));

vi.mock("../../lib/errors", () => ({
  reportError: vi.fn(() => ""),
  onError: vi.fn(() => () => {}),
  errMessage: (e: unknown) => (e instanceof Error ? e.message : undefined),
}));

const DIRECTIONS = [
  { dir_code: "09.03.02", name: "Информационные системы", profile: "", disciplines_count: 2 },
];

const DISCIPLINES = [
  { id: "d1", name: "Базы данных", in_scope: true, competencies_count: 3, skills_count: 12 },
  { id: "d2", name: "Web-разработка", in_scope: true, competencies_count: 2, skills_count: 9 },
];

const DISCIPLINE_DETAIL = { id: "d1", name: "Базы данных", competencies: [] };

const ANALYSIS = {
  direction: "09.03.02",
  direction_name: "Информационные системы",
  profile: "",
  total_disciplines: 2,
  average_coverage: 0.9,
  average_market_top_coverage: 0.62,
  average_market_top_reliable: true,
  average_market_weight_coverage: 0.48,
  coverage_level: "medium",
  total_gaps_across_all: 9,
  top_cross_discipline_gaps: [
    { skill: "OceanBase", disciplines: 3 },
    { skill: "Berkeley DBXML", disciplines: 1 },
  ],
  top_emerging_across_all: [
    { skill: "Ray ML", frequency: 3, change_pct: 12.7, basis: "рост спроса" },
    { skill: "k6", frequency: 4 },
  ],
  recommendations: [],
  trends: { rising: [], declining: [] },
  disciplines: [
    {
      name: "Базы данных",
      coverage_ratio: 0.55,
      weighted_coverage: 0.4,
      coverage_level: "medium",
      gaps: 2,
      emerging: 1,
      market_top_coverage: 0.6,
      market_weight_coverage: 0.5,
      market_top_total: 10,
      market_top_reliable: true,
      market_top_source: "explicit",
      years: [2025],
    },
    {
      name: "Web-разработка",
      coverage_ratio: 0.3,
      coverage_level: "low",
      gaps: 4,
      emerging: 0,
      years: [2025],
    },
  ],
  generated_at: "2026-01-01T00:00:00Z",
};

afterEach(cleanup);

beforeEach(() => {
  vi.mocked(api).mockReset();
  vi.mocked(api).mockImplementation(((endpoint: string) => {
    if (endpoint.startsWith("/teacher/analysis/meta")) return Promise.resolve({ stale: false });
    if (endpoint.startsWith("/teacher/analysis")) return Promise.resolve(ANALYSIS);
    if (endpoint.startsWith("/teacher/krm/recommendations")) return Promise.resolve([]);
    if (endpoint.startsWith("/teacher/krm/directions")) return Promise.resolve(DIRECTIONS);
    if (endpoint.startsWith("/teacher/krm/disciplines/")) {
      return Promise.resolve(DISCIPLINE_DETAIL);
    }
    if (endpoint.startsWith("/teacher/krm/disciplines")) return Promise.resolve(DISCIPLINES);
    if (endpoint.startsWith("/teacher/krm/scope-summary")) {
      return Promise.resolve({
        total: 2,
        in_scope: 2,
        excluded_manual: 0,
        excluded_no_data: 0,
        analyzed: 2,
      });
    }
    if (endpoint.startsWith("/teacher/krm/stats")) {
      return Promise.resolve({
        total_disciplines: 2,
        total_competencies: 5,
        total_skills: 30,
        total_reports: 2,
      });
    }
    if (endpoint.startsWith("/teacher/rpd/sources")) {
      return Promise.resolve({ yandex_covered: [] });
    }
    return Promise.reject(new Error(`unexpected endpoint: ${endpoint}`));
  }) as never);
});

describe("TeacherDashboard (a11y)", () => {
  it("загруженный анализ направления - без нарушений", async () => {
    const { container } = render(<TeacherDashboard />);
    await screen.findByText("по спросу: 48.0%");

    await expectNoA11yViolations(container);
  });
});
