/**
 * GapAnalysisVisualizer (7.5.4): состояния ML-графика (loading/ready/empty/error),
 * переключение на кластерные инсайты и обработка ошибки картинки.
 * recharts/motion/SHAP-карточка замоканы: тестируем логику компонента.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import { GapAnalysisVisualizer } from "./GapAnalysisVisualizer";

vi.mock("../api", () => ({ api: vi.fn() }));
vi.mock("../../lib/errors", () => ({
  reportError: vi.fn(),
  errMessage: (e: unknown) => (e instanceof Error ? e.message : String(e ?? "")),
}));

vi.mock("./ShapExplanationCard", () => ({
  ShapExplanationCard: () => <div data-testid="shap-card" />,
}));

vi.mock("motion/react", () => {
  const Passthrough = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
  return { motion: { div: Passthrough, button: Passthrough } };
});

vi.mock("recharts", () => {
  const Passthrough = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
  // ResponsiveContainer в jsdom без размеров не рендерит детей - пропускаем.
  const Container = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
  return {
    Bar: Passthrough,
    BarChart: Passthrough,
    CartesianGrid: () => null,
    Cell: () => null,
    ResponsiveContainer: Container,
    Tooltip: () => null,
    XAxis: () => null,
    YAxis: () => null,
  };
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const apiMock = vi.mocked(api);

describe("GapAnalysisVisualizer", () => {
  it("рисует шапку и кастомный профиль", async () => {
    apiMock.mockResolvedValue({ recommendations: [] } as never);
    render(<GapAnalysisVisualizer profile="dc" />);
    expect(screen.getByText("Визуализация GAP-анализа")).toBeTruthy();
    await waitFor(() =>
      expect(screen.getByText(/Нет ML-рекомендаций для 'dc'/)).toBeTruthy(),
    );
  });

  it("готовые рекомендации -> график и текстовая альтернатива", async () => {
    apiMock.mockResolvedValue({
      recommendations: [
        { skill: "python", importance_score: 0.9 },
        { skill: "sql", importance_score: 0.4 },
      ],
    } as never);
    render(<GapAnalysisVisualizer profile="base" />);
    await waitFor(() =>
      expect(screen.getByText("ТОП важных навыков (LTR XGBoost) - base")).toBeTruthy(),
    );
    expect(
      screen.getByRole("img", { name: /Важность недостающих навыков по ML-модели, топ-2/ }),
    ).toBeTruthy();
    expect(screen.getByText(/python: 90\.0%/)).toBeTruthy();
    expect(screen.getByTestId("shap-card")).toBeTruthy();
  });

  it("пустые рекомендации -> EmptyState с сообщением сервера", async () => {
    apiMock.mockResolvedValue({ recommendations: [], message: "Модель не обучена" } as never);
    render(<GapAnalysisVisualizer profile="top_dc" />);
    await waitFor(() =>
      expect(screen.getByText(/Нет ML-рекомендаций для 'top_dc'/)).toBeTruthy(),
    );
    expect(screen.getByText("Модель не обучена")).toBeTruthy();
  });

  it("ошибка API -> ErrorState и retry повторяет запрос", async () => {
    apiMock.mockRejectedValueOnce(new Error("boom")).mockResolvedValue({ recommendations: [] } as never);
    render(<GapAnalysisVisualizer profile="base" />);
    await waitFor(() =>
      expect(screen.getByText("Не удалось загрузить ML-рекомендации профиля.")).toBeTruthy(),
    );
    const callsBefore = apiMock.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: /Повтор|Retry|Попроб/i }));
    await waitFor(() => expect(apiMock.mock.calls.length).toBeGreaterThan(callsBefore));
  });

  it("кластерные инсайты: картинка и её ошибка", async () => {
    apiMock.mockResolvedValue({ recommendations: [] } as never);
    render(<GapAnalysisVisualizer profile="base" />);
    fireEvent.click(screen.getByRole("button", { name: /Кластерные инсайты/ }));
    const img = await waitFor(() => screen.getByAltText("Кластерные инсайты"));
    expect(img.getAttribute("src")).toContain("/api/results/images/base/cluster_insights");
    fireEvent.error(img);
    await waitFor(() => expect(screen.getByText("Визуализация недоступна")).toBeTruthy());
    expect(screen.getByText(/Изображение не найдено/)).toBeTruthy();
  });
});