/**
 * MyCompetenciesCard (7.5.4): скелетон, ошибка с retry, пусто, топ пробелов,
 * раскрытие дисциплины, совпавшие/недостающие навыки с добавлением, навигация.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MyCompetenciesCard } from "./MyCompetenciesCard";

afterEach(cleanup);

type Props = ComponentProps<typeof MyCompetenciesCard>;

const KRM = {
  direction: "09.03.01",
  direction_name: "Информатика и ВТ",
  profile: "backend",
  counts: { krm: 3, student: 2, overlap: 1, merged: 4, student_only: 1 },
  disciplines: [
    {
      discipline: "Базы данных",
      competencies: [
        {
          code: "ПК-1",
          skills: [
            { text: "SQL", student_has: true },
            { text: "Индексы", student_has: false },
          ],
        },
      ],
    },
  ],
};

function base(overrides: Partial<Props> = {}): Props {
  return {
    krmLoading: false,
    krmError: null,
    onRetry: vi.fn(),
    krm: KRM,
    dirCode: "09.03.01",
    topGaps: [],
    missingPool: [],
    discStats: [{ name: "Базы данных", total: 2, has: 1, missing: 1 }],
    openDisc: {},
    onToggleDisc: vi.fn(),
    saving: false,
    onAddSkill: vi.fn(),
    onNavigate: vi.fn(),
    ...overrides,
  } as Props;
}

describe("MyCompetenciesCard", () => {
  it("загрузка -> скелетоны", () => {
    render(<MyCompetenciesCard {...base({ krm: null, krmLoading: true })} />);
    expect(screen.getByLabelText("Загрузка компетенций")).toBeTruthy();
  });

  it("ошибка -> ErrorState с retry", () => {
    const onRetry = vi.fn();
    render(<MyCompetenciesCard {...base({ krm: null, krmError: "КРМ недоступна", onRetry })} />);
    expect(screen.getByText("КРМ недоступна")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Повтор|Retry|Попроб/i }));
    expect(onRetry).toHaveBeenCalled();
  });

  it("нет дисциплин -> EmptyState", () => {
    render(<MyCompetenciesCard {...base({ krm: { ...KRM, disciplines: [] } })} />);
    expect(screen.getByText("Нет данных")).toBeTruthy();
  });

  it("заголовок с названием программы и счётчик дисциплины", () => {
    render(<MyCompetenciesCard {...base()} />);
    expect(screen.getByText("Мои компетенции")).toBeTruthy();
    expect(screen.getByText(/Программа Информатика и ВТ/)).toBeTruthy();
    expect(screen.getByText("1/2")).toBeTruthy();
  });

  it("топ пробелов и быстрые кнопки добавления из пула", () => {
    const onAddSkill = vi.fn();
    render(
      <MyCompetenciesCard
        {...base({
          onAddSkill,
          topGaps: [{ name: "Базы данных", total: 5, has: 2, missing: 3 }],
          missingPool: [
            { text: "Индексы", code: "ПК-1", discipline: "Базы данных" },
            { text: "Транзакции", code: "ПК-1", discipline: "Базы данных" },
          ],
        })}
      />,
    );
    expect(screen.getByText("Топ пробелов (1)")).toBeTruthy();
    expect(screen.getByText("не хватает 3/5")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Добавить «Индексы»" }));
    expect(onAddSkill).toHaveBeenCalledWith("Индексы");
  });

  it("раскрытие дисциплины: совпало и не хватает, добавление навыка", () => {
    const onToggleDisc = vi.fn();
    const onAddSkill = vi.fn();
    const { rerender } = render(
      <MyCompetenciesCard {...base({ onToggleDisc, onAddSkill })} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Базы данных/ }));
    expect(onToggleDisc).toHaveBeenCalledWith("Базы данных");

    rerender(<MyCompetenciesCard {...base({ onAddSkill, openDisc: { "Базы данных": true } })} />);
    expect(screen.getByText("Совпало (1)")).toBeTruthy();
    expect(screen.getByText("Не хватает (1)")).toBeTruthy();
    expect(screen.getByText("SQL")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Добавить «Индексы»" }));
    expect(onAddSkill).toHaveBeenCalledWith("Индексы");
  });

  it("дисциплина без данных показывает подсказку", () => {
    render(
      <MyCompetenciesCard
        {...base({ openDisc: { "Базы данных": true }, discStats: [] })}
      />,
    );
    expect(screen.getByText("нет данных - добавьте навыки выше")).toBeTruthy();
  });

  it("сохранение блокирует кнопки добавления", () => {
    render(
      <MyCompetenciesCard
        {...base({
          saving: true,
          openDisc: { "Базы данных": true },
          missingPool: [{ text: "Индексы", code: "ПК-1", discipline: "Базы данных" }],
          topGaps: [{ name: "Базы данных", total: 5, has: 2, missing: 3 }],
        })}
      />,
    );
    for (const b of screen.getAllByRole("button", { name: "Добавить «Индексы»" })) {
      expect((b as HTMLButtonElement).disabled).toBe(true);
    }
  });

  it("кнопка «Открыть результаты» ведёт на вкладку data", () => {
    const onNavigate = vi.fn();
    render(<MyCompetenciesCard {...base({ onNavigate })} />);
    fireEvent.click(screen.getByRole("button", { name: /Открыть результаты/ }));
    expect(onNavigate).toHaveBeenCalledWith("data");
  });
});