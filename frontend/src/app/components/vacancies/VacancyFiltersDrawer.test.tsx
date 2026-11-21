/**
 * VacancyFiltersDrawer: выезжающая панель фильтров (7.5.4). Пропсы - без сети.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { VacancyFiltersDrawer } from "./VacancyFiltersDrawer";

afterEach(cleanup);

type Props = React.ComponentProps<typeof VacancyFiltersDrawer>;

function props(over: Partial<Props> = {}): Props {
  return {
    open: true,
    onClose: vi.fn(),
    activeFilterCount: 0,
    searchQuery: "",
    onSearchQueryChange: vi.fn(),
    onSearchKeyPress: vi.fn(),
    onSearch: vi.fn(),
    loading: false,
    experienceFilter: "",
    onExperienceChange: vi.fn(),
    cityFilter: "",
    onCityChange: vi.fn(),
    allCityOptions: ["Москва", "Казань"],
    monthsFilter: null,
    onMonthsChange: vi.fn(),
    dateFrom: "",
    onDateFromChange: vi.fn(),
    dateTo: "",
    onDateToChange: vi.fn(),
    vacancyInfo: null,
    applied: { search: "", experience: "", city: "", months: null, date_from: "", date_to: "" },
    onApplyFilters: vi.fn(),
    total: 0,
    onClearFilters: vi.fn(),
    ...over,
  };
}

describe("VacancyFiltersDrawer", () => {
  it("закрыт -> ничего не рендерит", () => {
    render(<VacancyFiltersDrawer {...props({ open: false })} />);
    expect(screen.queryByText("Фильтры и поиск")).toBeNull();
  });

  it("открыт -> показывает заголовок и состояние фильтров", () => {
    render(<VacancyFiltersDrawer {...props({ activeFilterCount: 2 })} />);
    expect(screen.getByText("Фильтры и поиск")).toBeTruthy();
    expect(screen.getByText("Активно: 2")).toBeTruthy();
  });

  it("ввод в поиск зовёт onSearchQueryChange", () => {
    const onSearchQueryChange = vi.fn();
    render(<VacancyFiltersDrawer {...props({ onSearchQueryChange })} />);
    fireEvent.change(screen.getByPlaceholderText("Должность или компания"), {
      target: { value: "python" },
    });
    expect(onSearchQueryChange).toHaveBeenCalledWith("python");
  });

  it("«Показать вакансии» показывает total и применяет фильтры с закрытием", () => {
    const onApplyFilters = vi.fn();
    const onClose = vi.fn();
    render(<VacancyFiltersDrawer {...props({ total: 42, onApplyFilters, onClose })} />);
    fireEvent.click(screen.getByText("Показать вакансии (42)"));
    expect(onApplyFilters).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it("«Сбросить» заблокирован без активных фильтров", () => {
    const onClearFilters = vi.fn();
    render(<VacancyFiltersDrawer {...props({ activeFilterCount: 0, onClearFilters })} />);
    const reset = screen.getByText("Сбросить").closest("button") as HTMLButtonElement;
    expect(reset.disabled).toBe(true);
  });

  it("закрытие по кнопке зовёт onClose", () => {
    const onClose = vi.fn();
    render(<VacancyFiltersDrawer {...props({ onClose })} />);
    fireEvent.click(screen.getByLabelText("Закрыть фильтры"));
    expect(onClose).toHaveBeenCalled();
  });
});