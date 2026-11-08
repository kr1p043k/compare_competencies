import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PipelineProgress } from "./PipelineProgress";
import type { PipelineStep } from "./PipelineProgress";

// Явный cleanup: при singleFork автоочистка RTL не срабатывает между файлами.
afterEach(() => {
  cleanup();
});

const step = (overrides: Partial<PipelineStep> = {}): PipelineStep => ({
  step: 2,
  total: 4,
  status: "running",
  message: "Очистка весов навыков...",
  progress: 50,
  logs: [],
  ...overrides,
});

describe("PipelineProgress", () => {
  it("без currentStep не рендерит ничего", () => {
    const { container } = render(<PipelineProgress currentStep={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("идущий шаг 2 из 4: статусы выполнено/выполняется/ожидание, бейджи процента и шага, акцент текущего шага", () => {
    render(
      <PipelineProgress
        currentStep={step({ subProgress: 35, logs: ["поиск вакансий на hh.ru"] })}
        onCancel={vi.fn()}
        onRestart={vi.fn()}
        showLogs
      />,
    );

    // Заголовок и живое сообщение из бэкенда.
    expect(screen.getByText("Обновление данных")).toBeInTheDocument();
    expect(screen.getByText("Очистка весов навыков...")).toBeInTheDocument();

    // Бейджи «шаг» и «процент».
    expect(screen.getByText("Шаг 2 из 4")).toBeInTheDocument();
    expect(screen.getByText("50%")).toBeInTheDocument();

    // Статусы по списку шагов: 1 выполнен, 2 идёт, 3-4 ожидают.
    const rows = [
      { title: "Сбор вакансий", label: "Выполнено" },
      { title: "Подготовка данных", label: "Выполняется" },
      { title: "Обучение моделей", label: "Ожидание" },
      { title: "GAP-анализ и рекомендации", label: "Ожидание" },
    ] as const;
    for (const { title, label } of rows) {
      const li = screen.getByText(title).closest("li");
      expect(li).not.toBeNull();
      expect(within(li as HTMLElement).getByText(label)).toBeInTheDocument();
    }
    expect(screen.getAllByText("Выполнено")).toHaveLength(1);
    expect(screen.getAllByText("Ожидание")).toHaveLength(2);

    // Текущий шаг помечен для скринридеров и подсвечен токеном primary-soft.
    const activeLi = screen.getByText("Подготовка данных").closest("li");
    expect(activeLi).toHaveAttribute("aria-current", "step");

    // Прогресс-бары: общий 50 и под-прогресс 35.
    const bars = screen.getAllByRole("progressbar");
    const mainBar = bars.find((b) => b.getAttribute("aria-label") === "Общий прогресс пайплайна");
    expect(mainBar).toHaveAttribute("aria-valuenow", "50");
    const subBar = bars.find((b) => b.getAttribute("aria-label") === "Прогресс текущего шага");
    expect(subBar).toHaveAttribute("aria-valuenow", "35");
    expect(screen.getByText("35%")).toBeInTheDocument();

    // Журнал виден только при Running + showLogs.
    expect(screen.getByText("Журнал пайплайна")).toBeInTheDocument();
    expect(screen.getByText("поиск вакансий на hh.ru")).toBeInTheDocument();
  });

  it("showLogs=false скрывает терминал журнала", () => {
    render(
      <PipelineProgress
        currentStep={step({ logs: ["строка лога"] })}
        onCancel={vi.fn()}
        showLogs={false}
      />,
    );
    expect(screen.queryByText("Журнал пайплайна")).not.toBeInTheDocument();
    expect(screen.queryByText("строка лога")).not.toBeInTheDocument();
  });

  it("«Остановить» вызывает onCancel, по завершении кнопка не рендерится", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    const { rerender } = render(
      <PipelineProgress currentStep={step()} onCancel={onCancel} />,
    );

    await user.click(screen.getByRole("button", { name: "Остановить" }));
    expect(onCancel).toHaveBeenCalledTimes(1);

    rerender(<PipelineProgress currentStep={step()} onCancel={onCancel} />);
    expect(screen.getByRole("button", { name: "Остановить" })).toBeInTheDocument();

    rerender(
      <PipelineProgress currentStep={step({ status: "completed", progress: 100 })} onCancel={onCancel} />,
    );
    expect(screen.queryByRole("button", { name: "Остановить" })).not.toBeInTheDocument();
  });

  it("completed: все шаги выполнены, бейдж процента 100%, «Запустить ещё» зовёт onRestart", async () => {
    const user = userEvent.setup();
    const onRestart = vi.fn();
    render(
      <PipelineProgress
        currentStep={step({ step: 4, status: "completed", progress: 100, message: "Готово" })}
        onRestart={onRestart}
      />,
    );

    expect(screen.getByText("Анализ завершён")).toBeInTheDocument();
    expect(screen.getByText("Готово")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getAllByText("Выполнено")).toHaveLength(4);
    expect(screen.queryByText("Выполняется")).not.toBeInTheDocument();
    expect(screen.queryByText("Ожидание")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Успешно! Все данные обновлены.");

    await user.click(screen.getByRole("button", { name: "Запустить ещё" }));
    expect(onRestart).toHaveBeenCalledTimes(1);
  });

  it("error на шаге 3: бейдж «Ошибка» у упавшего шага, alert-баннер, «Повторить» зовёт onRestart", async () => {
    const user = userEvent.setup();
    const onRestart = vi.fn();
    render(
      <PipelineProgress
        currentStep={step({
          step: 3,
          status: "error",
          message: "Обучение LTR-модели (может занять время) упало",
        })}
        onRestart={onRestart}
      />,
    );

    expect(screen.getByText("Ошибка выполнения")).toBeInTheDocument();
    const failedLi = screen.getByText("Обучение моделей").closest("li");
    expect(within(failedLi as HTMLElement).getByText("Ошибка")).toBeInTheDocument();
    expect(screen.getAllByText("Выполнено")).toHaveLength(2);

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Обучение LTR-модели (может занять время) упало");

    await user.click(screen.getByRole("button", { name: "Повторить" }));
    expect(onRestart).toHaveBeenCalledTimes(1);
  });

  it("отмена: «Пайплайн остановлен», бейдж «Остановлено» на текущем шаге, рестарт доступен", async () => {
    const user = userEvent.setup();
    const onRestart = vi.fn();
    render(
      <PipelineProgress
        currentStep={step({ status: "error", message: "Отменено" })}
        onRestart={onRestart}
      />,
    );

    expect(screen.getByText("Пайплайн остановлен")).toBeInTheDocument();
    expect(screen.queryByText("Ошибка выполнения")).not.toBeInTheDocument();

    const stoppedLi = screen.getByText("Подготовка данных").closest("li");
    expect(within(stoppedLi as HTMLElement).getByText("Остановлено")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Повторить" }));
    expect(onRestart).toHaveBeenCalledTimes(1);
  });

  it("падение до первого шага (step 0): «Ошибка запуска», без списка шагов и бейджа шага", () => {
    render(
      <PipelineProgress
        currentStep={step({ step: 0, total: 1, status: "error", message: "Не получен ID задачи", progress: 0 })}
        onRestart={vi.fn()}
      />,
    );

    expect(screen.getByText("Ошибка запуска")).toBeInTheDocument();
    expect(screen.queryByText(/Шаг \d из \d/)).not.toBeInTheDocument();
    expect(screen.queryByText("Ожидание")).not.toBeInTheDocument();
    expect(screen.queryAllByRole("progressbar")).toHaveLength(0);
    expect(screen.getByRole("alert")).toHaveTextContent("Не получен ID задачи");
  });
});
