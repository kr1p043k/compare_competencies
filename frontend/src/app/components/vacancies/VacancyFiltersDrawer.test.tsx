/**
 * VacancyFiltersDrawer: выезжающая панель фильтров вакансий (7.5.4).
 * Презентационный компонент: состояние и колбэки приходят пропсами.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { VacancyFiltersDrawer } from "./VacancyFiltersDrawer";
import type { AppliedFilters, VacancyInfo } from "./types";

afterEach(cleanup);

const NO_FILTERS: AppliedFilters = {
  search: "",
  experience: "all",
  city: "all",
  months: null,
  date_from: "",
  date_to: "",
};

const VACANCY_INFO: VacancyInfo = {
  count: 42,
  with_skills: 30,
  file_modified: "2026-11-20T10:00:00",
  date_range: {
    from: "2026-09-01T00:00:00",
    to: "2026-11-20T00:00:00",
  },
  load_error: null,
};

type DrawerProps = React.ComponentProps<typeof VacancyFiltersDrawer>;

function props(over: Partial<DrawerProps> = {}): DrawerProps {
  return {
    open: true,
    onClose: vi.fn(),
    activeFilterCount: 0,
    searchQuery: "",
    onSearchQueryChange: vi.fn(),
    onSearchKeyPress: vi.fn(),
    onSearch: vi.fn(),
    loading: false,
    experienceFilter: "all",
    onExperienceChange: vi.fn(),
    cityFilter: "all",
    onCityChange: vi.fn(),
    allCityOptions: ["Москва", "Санкт-Петербург"],
    monthsFilter: null,
    onMonthsChange: vi.fn(),
    dateFrom: "",
    onDateFromChange: vi.fn(),
    dateTo: "",
    onDateToChange: vi.fn(),
    vacancyInfo: VACANCY_INFO,
    applied: NO_FILTERS,
    onApplyFilters: vi.fn(),
    total: 0,
    onClearFilters: vi.fn(),
    ...over,
  };
}

describe("VacancyFiltersDrawer", () => {
  it("при open=false ничего не рендерит", () => {
    render(<VacancyFiltersDrawer {...props({ open: false })} />);
    expect(screen.queryByText("Фильтры и поиск")).toBeNull();
  });

  it("минимальный фикстур: заголовок, поля и пресеты периода", () => {
    render(<VacancyFiltersDrawer {...props()} />);
    expect(screen.getByText("Фильтры и поиск")).toBeTruthy();
    expect(screen.getByText("Показаны все вакансии")).toBeTruthy();
    expect(screen.getByPlaceholderText("Должность или компания")).toBeTruthy();
    expect(screen.getByText("Поиск по названию")).toBeTruthy();
    expect(screen.getByText("Опыт")).toBeTruthy();
    expect(screen.getByText("Город")).toBeTruthy();
    expect(screen.getByText("Период")).toBeTruthy();
    expect(screen.getByText("Всё время")).toBeTruthy();
    expect(screen.getByText("Месяц")).toBeTruthy();
    expect(screen.getByText("3 месяца")).toBeTruthy();
    expect(screen.getByText("Полгода")).toBeTruthy();
    expect(screen.getByText("Год")).toBeTruthy();
    expect(screen.getByLabelText("Дата от")).toBeTruthy();
    expect(screen.getByLabelText("Дата до")).toBeTruthy();
    expect(screen.getByText("Записи с 01.09.2026")).toBeTruthy();
  });

  it("пустое состояние: без счётчика, «Сбросить» недоступна, бейджей нет", () => {
    render(<VacancyFiltersDrawer {...props()} />);
    expect(screen.getByRole("button", { name: "Показать вакансии" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Сбросить" })).toBeDisabled();
    expect(screen.queryByText("senior ✕")).toBeNull();
    expect(screen.queryByText("Москва ✕")).toBeNull();
  });

  it("заполненное состояние: счётчики и бейджи применённых фильтров", () => {
    render(
      <VacancyFiltersDrawer
        {...props({
          total: 42,
          activeFilterCount: 4,
          applied: {
            search: "python",
            experience: "senior",
            city: "Москва",
            months: 3,
            date_from: "",
            date_to: "",
          },
        })}
      />,
    );
    expect(screen.getByText("Активно: 4")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Показать вакансии (42)" })).toBeTruthy();
    expect(screen.getByText("senior ✕")).toBeTruthy();
    expect(screen.getByText("Москва ✕")).toBeTruthy();
    expect(screen.getByText('"python" ✕')).toBeTruthy();
    expect(screen.getByText("3 мес ✕")).toBeTruthy();
  });

  it("клик по бейджу города снимает фильтр и применяет его", () => {
    const onCityChange = vi.fn();
    const onApplyFilters = vi.fn();
    render(
      <VacancyFiltersDrawer
        {...props({
          onCityChange,
          onApplyFilters,
          activeFilterCount: 1,
          applied: { ...NO_FILTERS, city: "Москва" },
        })}
      />,
    );
    fireEvent.click(screen.getByText("Москва ✕"));
    expect(onCityChange).toHaveBeenCalledWith("all");
    expect(onApplyFilters).toHaveBeenCalledWith({ city: "all" });
  });

  it("пресет периода выставляет месяцы и сбрасывает даты", () => {
    const onMonthsChange = vi.fn();
    const onDateFromChange = vi.fn();
    const onDateToChange = vi.fn();
    render(
      <VacancyFiltersDrawer
        {...props({ onMonthsChange, onDateFromChange, onDateToChange })}
      />,
    );
    fireEvent.click(screen.getByText("Месяц"));
    expect(onMonthsChange).toHaveBeenCalledWith(1);
    expect(onDateFromChange).toHaveBeenCalledWith("");
    expect(onDateToChange).toHaveBeenCalledWith("");
  });

  it("поиск: ввод, Enter и кнопка «Найти»", () => {
    const onSearchQueryChange = vi.fn();
    const onSearchKeyPress = vi.fn();
    const onSearch = vi.fn();
    render(
      <VacancyFiltersDrawer
        {...props({ onSearchQueryChange, onSearchKeyPress, onSearch })}
      />,
    );
    fireEvent.change(screen.getByPlaceholderText("Должность или компания"), {
      target: { value: "java" },
    });
    expect(onSearchQueryChange).toHaveBeenCalledWith("java");
    fireEvent.keyPress(screen.getByPlaceholderText("Должность или компания"), {
      key: "Enter",
      charCode: 13,
    });
    expect(onSearchKeyPress).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Найти" }));
    expect(onSearch).toHaveBeenCalledTimes(1);
  });

  it("«Показать вакансии» применяет фильтры и закрывает панель", () => {
    const onApplyFilters = vi.fn();
    const onClose = vi.fn();
    render(
      <VacancyFiltersDrawer {...props({ onApplyFilters, onClose, total: 7 })} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Показать вакансии (7)" }));
    expect(onApplyFilters).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("«Сбросить» вызывает onClearFilters, крестик закрывает панель", () => {
    const onClearFilters = vi.fn();
    const onClose = vi.fn();
    render(
      <VacancyFiltersDrawer
        {...props({ onClearFilters, onClose, activeFilterCount: 2 })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Сбросить" }));
    expect(onClearFilters).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Закрыть фильтры" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("при загрузке кнопки «Найти» и «Показать вакансии» заблокированы", () => {
    render(<VacancyFiltersDrawer {...props({ loading: true, total: 5 })} />);
    expect(screen.getByRole("button", { name: "Найти" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Показать вакансии (5)" }),
    ).toBeDisabled();
  });
});
