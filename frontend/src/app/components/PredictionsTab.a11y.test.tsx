/**
 * a11y: PredictionsTab без нарушений axe после загрузки прогнозов.
 *
 * Компонент сам тянет данные из /api/forecast/* в useEffect, поэтому
 * мокаем lib/api.request по образцу PredictionsTab.test.tsx и ждём
 * появления строки навыка через findBy, то есть окончания загрузки.
 *
 * Без установленного jest-axe тест падает с подсказкой по установке,
 * а не пропускается (см. src/test/a11y.ts).
 */
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, it, vi } from "vitest";

import { request } from "../../lib/api";
import { expectNoA11yViolations } from "../../test/a11y";
import { PredictionsTab } from "./PredictionsTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));

const requestMock = vi.mocked(request);

const GROWING = {
  forecasts: [
    {
      skill: "kubernetes",
      current_frequency: 456,
      predicted_growth: 1.4,
      predicted_change_pct: 143.9,
      confidence: 0.16,
      next_year_frequency: 1112,
      method: "prophet",
      trend_direction: "up",
      data_points: 4,
      forecast_months: 3,
      forecast_steps: [456, 700, 1112],
    },
  ],
  engine: "prophet",
  prophet_models: 5,
  vacancies_count: 100,
  data_from: "2026-06-01",
  data_to: "2026-11-01",
  snapshots_count: 6,
};

beforeEach(() => {
  requestMock.mockImplementation((url: string) => {
    const u = String(url);
    if (u.includes("/forecast/engine")) {
      return Promise.resolve({ engine: "prophet", prophet_models: 5, state: "ready" });
    }
    if (u.includes("/forecast/top")) return Promise.resolve(GROWING);
    return Promise.resolve({});
  });
});

describe("PredictionsTab (a11y)", () => {
  it("загруженный список растущих навыков - без нарушений", async () => {
    const { container } = render(<PredictionsTab />);
    await screen.findByText("kubernetes");

    await expectNoA11yViolations(container);
  });
});
