/**
 * DisciplineRecsSection (7.5.4): счётчик, формы добавления (add/modify/remove),
 * фильтры по компетенции и типу, сброс фильтров, удаление с подтверждением.
 * Компонент презентационный - всё состояние приходит пропсами.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DisciplineRecsSection } from "./DisciplineRecsSection";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

type Props = ComponentProps<typeof DisciplineRecsSection>;

const SELECTED = {
  id: "d1",
  name: "Базы данных",
  competencies: [
    { id: "c1", code: "ПК-1", name: "Проектирование БД", skills: [] },
    { id: "c2", code: "ПК-2", name: "Оптимизация", skills: [] },
  ],
};

const RECS = [
  { id: 1, discipline_id: "Базы данных", competency_id: "ПК-1", suggestion: "Добавить практику", suggestion_type: "add" },
  { id: 2, discipline_id: "Базы данных", competency_id: "ПК-2", suggestion: "Убрать дубли", suggestion_type: "remove" },
];

function base(overrides: Partial<Props> = {}): Props {
  return {
    selected: SELECTED,
    recs: RECS,
    discRecCount: 2,
    recsVisible: RECS,
    recFiltersActive: false,
    recComp: "",
    setRecComp: vi.fn(),
    recTypeF: "",
    setRecTypeF: vi.fn(),
    showAddForm: false,
    setShowAddForm: vi.fn(),
    recType: "modify",
    setRecType: vi.fn(),
    selectedCompetency: "",
    setSelectedCompetency: vi.fn(),
    suggestion: "",
    setSuggestion: vi.fn(),
    recError: "",
    setRecError: vi.fn(),
    seedMsg: "",
    seedLoading: false,
    seedAutoRecs: vi.fn(),
    addRec: vi.fn().mockResolvedValue(true),
    deleteRec: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  } as Props;
}

describe("DisciplineRecsSection", () => {
  it("счётчик рекомендаций дисциплины и список", () => {
    render(<DisciplineRecsSection {...base()} />);
    expect(screen.getByText("Рекомендации")).toBeTruthy();
    expect(screen.getByText("(2)")).toBeTruthy();
    expect(screen.getByText("Добавить практику")).toBeTruthy();
    expect(screen.getByText("[add] ПК-1")).toBeTruthy();
  });

  it("пусто для дисциплины", () => {
    render(<DisciplineRecsSection {...base({ recs: [], discRecCount: 0, recsVisible: [] })} />);
    expect(screen.getByText("Для этой дисциплины рекомендаций пока нет")).toBeTruthy();
  });

  it("переключение формы добавления и заполнение из анализа", () => {
    const setShowAddForm = vi.fn();
    const seedAutoRecs = vi.fn();
    const { rerender } = render(
      <DisciplineRecsSection {...base({ setShowAddForm, seedAutoRecs })} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Добавить рекомендацию" }));
    expect(setShowAddForm).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("button", { name: "Заполнить из анализа" }));
    expect(seedAutoRecs).toHaveBeenCalled();

    rerender(<DisciplineRecsSection {...base({ showAddForm: true, seedLoading: true })} />);
    expect((screen.getByRole("button", { name: "Заполнение…" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("форма modify: выбор компетенции, ввод и отправка закрывает форму", async () => {
    const setSuggestion = vi.fn();
    const setRecError = vi.fn();
    const setShowAddForm = vi.fn();
    const setSelectedCompetency = vi.fn();
    const addRec = vi.fn().mockResolvedValue(true);
    render(
      <DisciplineRecsSection
        {...base({
          showAddForm: true,
          recType: "modify",
          setSuggestion,
          setRecError,
          setShowAddForm,
          setSelectedCompetency,
          addRec,
        })}
      />,
    );
    // Селект компетенций виден только для modify/remove.
    expect(screen.getByText("-- Select competency --")).toBeTruthy();
    fireEvent.change(screen.getByText("-- Select competency --").parentElement as HTMLSelectElement, {
      target: { value: "ПК-2" },
    });
    expect(setSelectedCompetency).toHaveBeenCalledWith("ПК-2");
    fireEvent.change(screen.getByPlaceholderText(/Your recommendation for this competency/), {
      target: { value: "Обновить" },
    });
    expect(setSuggestion).toHaveBeenCalledWith("Обновить");
    fireEvent.click(screen.getByRole("button", { name: "Отправить" }));
    expect(setRecError).toHaveBeenCalledWith("");
    await vi.waitFor(() => expect(addRec).toHaveBeenCalled());
    await vi.waitFor(() => expect(setShowAddForm).toHaveBeenCalledWith(false));
    expect(setSelectedCompetency).toHaveBeenCalledWith("");
  });

  it("форма add не показывает выбор компетенции и содержит ошибку", () => {
    render(
      <DisciplineRecsSection {...base({ showAddForm: true, recType: "add", recError: "сервер лёг" })} />,
    );
    expect(screen.queryByText("-- Select competency --")).toBeNull();
    expect(screen.getByPlaceholderText(/Your recommendation for this discipline/)).toBeTruthy();
    expect(screen.getByText("Не удалось сохранить: сервер лёг")).toBeTruthy();
  });

  it("смена типа рекомендации сбрасывает выбранную компетенцию", () => {
    const setRecType = vi.fn();
    const setSelectedCompetency = vi.fn();
    render(
      <DisciplineRecsSection {...base({ showAddForm: true, setRecType, setSelectedCompetency })} />,
    );
    fireEvent.change(screen.getByDisplayValue("Изменить"), { target: { value: "remove" } });
    expect(setRecType).toHaveBeenCalledWith("remove");
    expect(setSelectedCompetency).toHaveBeenCalledWith("");
  });

  it("фильтры: компетенция, тип, счётчик и сброс", () => {
    const setRecComp = vi.fn();
    const setRecTypeF = vi.fn();
    const { rerender } = render(
      <DisciplineRecsSection {...base({ setRecComp, setRecTypeF, recFiltersActive: true })} />,
    );
    fireEvent.change(screen.getByLabelText("Фильтр по компетенции"), { target: { value: "ПК-1" } });
    expect(setRecComp).toHaveBeenCalledWith("ПК-1");
    fireEvent.click(screen.getByRole("button", { name: "Изменить" }));
    expect(setRecTypeF).toHaveBeenCalledWith("modify");
    expect(screen.getByText("2 из 2")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Сбросить" }));
    expect(setRecComp).toHaveBeenCalledWith("");
    expect(setRecTypeF).toHaveBeenCalledWith("");

    rerender(<DisciplineRecsSection {...base({ recsVisible: [], discRecCount: 2 })} />);
    expect(screen.getByText(/По фильтру ничего не найдено/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "сбросить" }));
  });

  it("удаление спрашивает подтверждение", async () => {
    const deleteRec = vi.fn().mockResolvedValue(undefined);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<DisciplineRecsSection {...base({ deleteRec })} />);
    fireEvent.click(screen.getAllByRole("button", { name: "Удалить" })[1]);
    expect(confirmSpy).toHaveBeenCalled();
    await vi.waitFor(() => expect(deleteRec).toHaveBeenCalledWith(1));
  });

  it("отказ в подтверждении не удаляет", () => {
    const deleteRec = vi.fn();
    vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<DisciplineRecsSection {...base({ deleteRec })} />);
    fireEvent.click(screen.getAllByRole("button", { name: "Удалить" })[1]);
    expect(deleteRec).not.toHaveBeenCalled();
  });
});