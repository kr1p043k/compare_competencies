/**
 * DisciplineList: презентационный список дисциплин (7.5.4).
 * Всё состояние приходит пропсами - сеть не нужна.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DisciplineList } from "./DisciplineList";
import type { Discipline } from "./types";

afterEach(cleanup);

const D1: Discipline = {
  id: "d1",
  name: "Веб-технологии",
  in_scope: true,
  competencies_count: 4,
  skills_count: 12,
  semester: 3,
};
const D2: Discipline = {
  id: "d2",
  name: "Базы данных",
  in_scope: false,
  competencies_count: 2,
  skills_count: 7,
};

type ListProps = React.ComponentProps<typeof DisciplineList>;

function props(over: Partial<ListProps> = {}): ListProps {
  return {
    filtered: [D1, D2],
    disciplines: [D1, D2],
    analysisByName: new Map(),
    selected: null,
    search: "",
    setSearch: vi.fn(),
    yearFilter: 0,
    setYearFilter: vi.fn(),
    discFilter: "all",
    setDiscFilter: vi.fn(),
    discSort: "plan",
    setDiscSort: vi.fn(),
    discCounts: { all: 2, gaps: 1, low: 0, nodata: 1 },
    discFiltersActive: false,
    scopeSaving: null,
    loadDiscipline: vi.fn().mockResolvedValue(undefined),
    saveScope: vi.fn().mockResolvedValue(undefined),
    ...over,
  };
}

describe("DisciplineList", () => {
  it("рендерит поиск, счётчик и строки дисциплин", () => {
    render(<DisciplineList {...props()} />);
    expect(screen.getByLabelText("Поиск дисциплины")).toBeTruthy();
    expect(screen.getByText("2 из 2")).toBeTruthy();
    expect(screen.getByText("Веб-технологии")).toBeTruthy();
    expect(screen.getByText("Базы данных")).toBeTruthy();
  });

  it("фильтры спрятаны в поповер и открываются по кнопке", () => {
    render(<DisciplineList {...props()} />);
    // До клика фильтров в разметке нет: они не занимают место над списком.
    expect(screen.queryByText("Показать в списке")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /Фильтры/ }));
    expect(screen.getByText("Показать в списке")).toBeTruthy();
    expect(screen.getByText("Все (2)")).toBeTruthy();
    expect(screen.getByText("С пробелами (1)")).toBeTruthy();
  });

  it("поиск передаёт введённое значение наверх", () => {
    const setSearch = vi.fn();
    render(<DisciplineList {...props({ setSearch })} />);
    fireEvent.change(screen.getByLabelText("Поиск дисциплины"), { target: { value: "баз" } });
    expect(setSearch).toHaveBeenCalledWith("баз");
  });

  it("клик по строке грузит дисциплину", () => {
    const loadDiscipline = vi.fn().mockResolvedValue(undefined);
    render(<DisciplineList {...props({ loadDiscipline })} />);
    fireEvent.click(screen.getByText("Веб-технологии"));
    expect(loadDiscipline).toHaveBeenCalledWith("Веб-технологии");
  });

  it("пустой отфильтрованный список -> «Ничего не найдено»", () => {
    render(<DisciplineList {...props({ filtered: [] })} />);
    expect(screen.getByText(/Ничего не найдено/)).toBeTruthy();
    expect(screen.getByText("0 из 2")).toBeTruthy();
  });

  it("кнопки фильтра года переключают значение", () => {
    const setYearFilter = vi.fn();
    render(<DisciplineList {...props({ setYearFilter })} />);
    fireEvent.click(screen.getByRole("button", { name: /Фильтры/ }));
    fireEvent.click(screen.getByText("в плане 2024"));
    expect(setYearFilter).toHaveBeenCalledWith(2024);
  });
});
