/**
 * DataTab (7.5.4, поведение; a11y уже покрыт DataTab.a11y.test.tsx).
 * Компонент принимает всё пропсами, свои запросы не делает - тестируем
 * колбэки, ограничения по роли, состояния результата и выбор рендера.
 * Дочерние отчёты замоканы: их покрытие отдельное.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps, ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DataTab } from "./DataTab";

vi.mock("./RecommendationsReport", () => ({
  RecommendationsReport: () => <div data-testid="recommendations-report" />,
}));
vi.mock("./SummaryReport", () => ({
  SummaryReport: () => <div data-testid="summary-report" />,
}));
vi.mock("./MarketView", () => ({ MarketView: () => <div data-testid="market-view" /> }));
vi.mock("./ProfileView", () => ({ ProfileView: () => <div data-testid="profile-view" /> }));
vi.mock("./DataViewer", () => ({ DataViewer: () => <div data-testid="data-viewer" /> }));
vi.mock("./ui/select", () => {
  const Passthrough = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
  return {
    Select: Passthrough,
    SelectContent: Passthrough,
    SelectItem: Passthrough,
    SelectTrigger: Passthrough,
    SelectValue: () => null,
  };
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

type DataTabProps = ComponentProps<typeof DataTab>;

function base(): DataTabProps {
  return {
    profile: "backend",
    profilesList: ["backend", "frontend"],
    onProfileChange: vi.fn(),
    loading: false,
    isStudent: false,
    gapRunning: false,
    gapMsg: "",
    onLoadRecommendations: vi.fn(),
    onLoadSummary: vi.fn(),
    onRunGapAnalysis: vi.fn(),
    professionsList: ["Backend", "Frontend"],
    targetProfession: "",
    onTargetProfessionChange: vi.fn(),
    onCompareWithProfession: vi.fn(),
    resultLoadedAt: {},
    lastResult: null,
    onDownloadPdf: vi.fn(),
  };
}

const spy = (p: DataTabProps, key: keyof DataTabProps) => p[key] as unknown as ReturnType<typeof vi.fn>;

describe("DataTab", () => {
  it("кнопки зовут свои колбэки", () => {
    const p = base();
    render(<DataTab {...p} />);
    fireEvent.click(screen.getByRole("button", { name: /Показать сохранённые/ }));
    expect(spy(p, "onLoadRecommendations")).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Сводка/ }));
    expect(spy(p, "onLoadSummary")).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Запустить gap-анализ/ }));
    expect(spy(p, "onRunGapAnalysis")).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Отчёт \(PDF\)/ }));
    expect(spy(p, "onDownloadPdf")).toHaveBeenCalled();
  });

  it("студенту запрещены рекомендации, сводка, gap и PDF", () => {
    render(<DataTab {...base()} isStudent />);
    for (const name of [/Рекомендации/, /Сводка/, /Запустить gap-анализ/, /Отчёт \(PDF\)/]) {
      expect((screen.getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true);
    }
    // «Показать сохранённые» доступна студенту.
    expect((screen.getByRole("button", { name: /Показать сохранённые/ }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("loading без результата -> скелетон; без loading -> пустое состояние", () => {
    const { rerender } = render(<DataTab {...base()} loading />);
    expect(screen.getByRole("status", { name: "Загрузка результата" })).toBeTruthy();
    rerender(<DataTab {...base()} />);
    expect(screen.getByText("Результатов пока нет")).toBeTruthy();
  });

  it("gapRunning показывает сообщение о ходе", () => {
    render(<DataTab {...base()} gapRunning gapMsg="Считаем покрытие" />);
    expect(screen.getByText("Считаем покрытие")).toBeTruthy();
  });

  it("свежесть результата и признак устаревания", () => {
    const iso = new Date(Date.now() - 10 * 864e5).toISOString();
    render(
      <DataTab
        {...base()}
        resultLoadedAt={{ backend: iso }}
        lastResult={{ generated_at: iso }}
      />,
    );
    expect(screen.getByText(/данные устарели \(10 дн\.\)/)).toBeTruthy();
  });

  it("выбирает рендер результата по форме данных", () => {
    const cases: [Record<string, unknown>, string][] = [
      [{ recommendations: [] }, "recommendations-report"],
      [{ closest_roles: [] }, "recommendations-report"],
      [{ evaluations: [], profiles: [] }, "summary-report"],
      [{ skills: [], total: 0 }, "market-view"],
      [{ skills: [], profile_name: "backend" }, "profile-view"],
      [{ some: "json" }, "data-viewer"],
    ];
    for (const [lastResult, testId] of cases) {
      const { unmount } = render(<DataTab {...base()} lastResult={lastResult} />);
      expect(screen.getByTestId(testId)).toBeTruthy();
      unmount();
    }
  });

  it("сообщение «не найдены» превращается в карточку с запуском gap", () => {
    const p = base();
    render(<DataTab {...p} lastResult={{ message: "Данные не найдены" }} />);
    expect(screen.getByText("Данные не найдены")).toBeTruthy();
    expect(screen.getByText("Запустите gap-анализ для расчёта покрытия")).toBeTruthy();
    // В этом состоянии есть отдельная кнопка запуска внутри карточки.
    const buttons = screen.getAllByRole("button", { name: /Запустить gap-анализ/ });
    expect(buttons.length).toBe(2);
    fireEvent.click(buttons[1]);
    expect(spy(p, "onRunGapAnalysis")).toHaveBeenCalled();
  });
});