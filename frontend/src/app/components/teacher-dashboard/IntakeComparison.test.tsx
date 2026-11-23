import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api";
import { IntakeComparison } from "./IntakeComparison";

// Сетевой слой подменён: тест проверяет рендер отчёта, а не HTTP.
vi.mock("../../api", () => ({ api: vi.fn() }));
vi.mock("../../../lib/errors", () => ({
  errMessage: (e: unknown) => (e instanceof Error ? e.message : undefined),
}));

/** Ответ /teacher/intake-comparison: один добавленный, один убранный,
 *  одна сохранённая дисциплина со значимым падением покрытия. */
const REPORT = {
  direction: "09.03.02",
  old_year: 2024,
  new_year: 2025,
  direction_name: "Перспективные информационные технологии",
  added: [
    {
      name: "Основы Python",
      coverage_ratio: 0.89,
      coverage_level: "high",
      gaps: 12,
    },
  ],
  removed: [
    {
      name: "Математика",
      coverage_ratio: 0,
      coverage_level: "no_data",
      gaps: 0,
    },
  ],
  kept: [
    {
      name: "Алгоритмизация и программирование",
      old_name: null,
      coverage_ratio_old: 1.0,
      coverage_ratio_new: 0.733,
      coverage_delta: -0.267,
      gaps_old: 0,
      gaps_new: 51,
      gaps_delta: 51,
      significant_change: true,
    },
  ],
  summary: {
    added: 1,
    removed: 1,
    kept: 1,
    avg_coverage_old: 0.566,
    avg_coverage_new: 0.7318,
    avg_coverage_delta: 0.1658,
    improved: 13,
    worsened: 1,
    unchanged: 0,
  },
};

const mockedApi = vi.mocked(api);

describe("IntakeComparison", () => {
  beforeEach(() => {
    mockedApi.mockReset();
  });

  it("показывает дельту среднего покрытия и счётчики по сводке", async () => {
    mockedApi.mockResolvedValue(REPORT);
    render(<IntakeComparison dirCode="09.03.02" />);

    await waitFor(() =>
      expect(screen.getByText("Сравнение наборов: 2024 → 2025")).toBeInTheDocument(),
    );
    // Среднее покрытие: 56.6% -> 73.2% с дельтой в п.п.
    expect(screen.getByText(/56\.6%/)).toBeInTheDocument();
    expect(screen.getByText(/73\.2%/)).toBeInTheDocument();
    expect(screen.getByText(/\+16\.6 п\.п\./)).toBeInTheDocument();
    // Счётчики: сохранено / рост / падение.
    expect(screen.getByText("сохранено 1")).toBeInTheDocument();
    expect(screen.getByText("рост 13")).toBeInTheDocument();
    expect(screen.getByText("падение 1")).toBeInTheDocument();
    // Запрос идёт с годами из пропсов по умолчанию.
    expect(mockedApi).toHaveBeenCalledWith(
      "/teacher/intake-comparison?dir_code=09.03.02&old=2024&new=2025",
    );
  });

  it("раскрывает списки добавленных и убранных дисциплин", async () => {
    mockedApi.mockResolvedValue(REPORT);
    const user = userEvent.setup();
    render(<IntakeComparison dirCode="09.03.02" />);

    await screen.findByText("Добавлены в 2025 (1)");
    await user.click(screen.getByText("Добавлены в 2025 (1)"));
    expect(screen.getByText("Основы Python")).toBeInTheDocument();
    expect(screen.getByText("89.0%")).toBeInTheDocument();

    await user.click(screen.getByText("Убраны из плана (1)"));
    expect(screen.getByText("Математика")).toBeInTheDocument();
  });

  it("показывает падение покрытия у сохранённой дисциплины", async () => {
    mockedApi.mockResolvedValue(REPORT);
    const user = userEvent.setup();
    render(<IntakeComparison dirCode="09.03.02" />);

    await user.click(await screen.findByText("Изменилось покрытие (1)"));
    expect(
      screen.getByText("Алгоритмизация и программирование"),
    ).toBeInTheDocument();
    expect(screen.getByText(/100\.0% →/)).toBeInTheDocument();
    expect(screen.getByText(/-26\.7 п\.п\./)).toBeInTheDocument();
  });

  it("объясняет отсутствие сводок вместо пустого экрана", async () => {
    mockedApi.mockRejectedValue(new Error("Нет данных анализа для набора 2024"));
    render(<IntakeComparison dirCode="09.03.02" />);

    await waitFor(() =>
      expect(
        screen.getByText("Нет данных анализа для набора 2024"),
      ).toBeInTheDocument(),
    );
    // Заголовок остаётся, чтобы карточка не «исчезала» из интерфейса.
    expect(screen.getByText("Сравнение наборов: 2024 → 2025")).toBeInTheDocument();
  });

  it("перезапрашивает отчёт при смене направления", async () => {
    mockedApi.mockResolvedValue(REPORT);
    const { rerender } = render(<IntakeComparison dirCode="09.03.02" />);
    await screen.findByText("Сравнение наборов: 2024 → 2025");

    rerender(<IntakeComparison dirCode="02.03.02" />);
    await waitFor(() =>
      expect(mockedApi).toHaveBeenCalledWith(
        "/teacher/intake-comparison?dir_code=02.03.02&old=2024&new=2025",
      ),
    );
  });
});