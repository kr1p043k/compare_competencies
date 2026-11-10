/**
 * PredictionsTab - регресс-щит для UI-фиксов вкладки «Прогнозы»:
 * - в падающих строках нет стрелки «→» (формат «с X до Y · точек: N»);
 * - пустой период не показывается прочерком «· -»;
 * - чип «низкая достоверность» использует тот же нейтральный стиль, что «месяцев истории» (bg-muted).
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "../../lib/api";
import { PredictionsTab } from "./PredictionsTab";

vi.mock("../../lib/api", () => ({ request: vi.fn() }));

afterEach(cleanup);

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

const OBSERVED = {
  forecasts: [
    { skill: "linux", observed_change_pct: -12.3, first_frequency: 456, last_frequency: 400, points: 4 },
  ],
  required_points: 4,
  max_points: 6,
  engine: "prophet",
  prophet_models: 5,
  vacancies_count: 100,
  data_from: null,
  data_to: null,
};

beforeEach(() => {
  requestMock.mockImplementation((url: string) => {
    const u = String(url);
    if (u.includes("/forecast/engine")) {
      return Promise.resolve({ engine: "prophet", prophet_models: 5, state: "ready" });
    }
    if (u.includes("/forecast/observed")) return Promise.resolve(OBSERVED);
    if (u.includes("/forecast/top")) return Promise.resolve(GROWING);
    return Promise.resolve({});
  });
});

describe("PredictionsTab", () => {
  it("низкая достоверность оформлена как чип месяцев истории (bg-muted)", async () => {
    render(<PredictionsTab />);
    const row = await screen.findByText("kubernetes");
    fireEvent.click(row); // развернуть строку прогноза

    const chip = await screen.findByText("низкая достоверность");
    expect(chip.className).toContain("bg-muted");
    expect(screen.getByText(/месяцев истории: 4/)).toBeInTheDocument();
  });

  it("в падающих нет стрелки «→» и показан период-диапазон", async () => {
    const user = userEvent.setup();
    const { container } = render(<PredictionsTab />);
    await screen.findByText("kubernetes");

    await user.click(screen.getByRole("tab", { name: /Падающие/ }));
    await screen.findByText("linux");

    expect(screen.getByText(/с 456 до 400 · точек: 4/)).toBeInTheDocument();
    expect(container.textContent ?? "").not.toContain("→");
  });

  it("пустой период не показывается прочерком", async () => {
    const user = userEvent.setup();
    const { container } = render(<PredictionsTab />);
    await screen.findByText("kubernetes");

    await user.click(screen.getByRole("tab", { name: /Падающие/ }));
    await screen.findByText("linux");

    // data_from/data_to = null -> сегмент периода не рендерится вовсе.
    await waitFor(() => expect(container.textContent ?? "").not.toContain("· - · -"));
    expect(screen.getByText(/Измеренное падение спроса за период/)).toBeInTheDocument();
  });
});