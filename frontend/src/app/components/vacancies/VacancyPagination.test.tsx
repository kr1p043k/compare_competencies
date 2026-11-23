/**
 * VacancyPagination (7.5.4): окно страниц, границы, блокировка при загрузке.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { VacancyPagination } from "./VacancyPagination";

afterEach(cleanup);

type Props = ComponentProps<typeof VacancyPagination>;

function base(overrides: Partial<Props> = {}): Props {
  return {
    currentPage: 1,
    totalPages: 3,
    loading: false,
    onPageChange: vi.fn(),
    ...overrides,
  };
}

describe("VacancyPagination", () => {
  it("короткий список: все страницы, первая активна", () => {
    render(<VacancyPagination {...base()} />);
    for (const n of ["1", "2", "3"]) expect(screen.getByRole("button", { name: n })).toBeTruthy();
    expect(screen.getByRole("button", { name: "1" }).getAttribute("aria-current")).toBe("page");
    expect(screen.queryByRole("button", { name: "4" })).toBeNull();
  });

  it("на первой странице «назад» заблокировано", () => {
    render(<VacancyPagination {...base()} />);
    expect((screen.getByRole("button", { name: "Предыдущая страница" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Следующая страница" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("на последней странице «вперёд» заблокировано", () => {
    render(<VacancyPagination {...base({ currentPage: 3 })} />);
    expect((screen.getByRole("button", { name: "Следующая страница" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("навигация: назад и вперёд передают апдейтер", () => {
    const onPageChange = vi.fn();
    render(<VacancyPagination {...base({ currentPage: 2, onPageChange })} />);
    fireEvent.click(screen.getByRole("button", { name: "Предыдущая страница" }));
    // Апдейтер применяется к текущей странице внутри компонента-родителя.
    expect(onPageChange).toHaveBeenCalledTimes(1);
    const back = onPageChange.mock.calls[0][0] as (p: number) => number;
    expect(back(2)).toBe(1);
    expect(back(1)).toBe(1);
    fireEvent.click(screen.getByRole("button", { name: "Следующая страница" }));
    const fwd = onPageChange.mock.calls[1][0] as (p: number) => number;
    expect(fwd(2)).toBe(3);
    expect(fwd(3)).toBe(3);
  });

  it("клик по номеру ставит именно эту страницу", () => {
    const onPageChange = vi.fn();
    render(<VacancyPagination {...base({ onPageChange, totalPages: 3 })} />);
    fireEvent.click(screen.getByRole("button", { name: "3" }));
    const setter = onPageChange.mock.calls[0][0] as (p: number) => number;
    expect(setter(1)).toBe(3);
  });

  it("окно из 5 страниц: сдвиг в начале, середине и конце", () => {
    const { rerender } = render(<VacancyPagination {...base({ currentPage: 2, totalPages: 20 })} />);
    expect([1, 2, 3, 4, 5].map((n) => !!screen.queryByRole("button", { name: String(n) }))).toEqual([true, true, true, true, true]);
    expect(screen.queryByRole("button", { name: "6" })).toBeNull();

    rerender(<VacancyPagination {...base({ currentPage: 10, totalPages: 20 })} />);
    expect([8, 9, 10, 11, 12].every((n) => !!screen.queryByRole("button", { name: String(n) }))).toBe(true);
    expect(screen.queryByRole("button", { name: "7" })).toBeNull();

    rerender(<VacancyPagination {...base({ currentPage: 19, totalPages: 20 })} />);
    expect([16, 17, 18, 19, 20].every((n) => !!screen.queryByRole("button", { name: String(n) }))).toBe(true);
    expect(screen.queryByRole("button", { name: "15" })).toBeNull();
  });

  it("загрузка блокирует все кнопки", () => {
    render(<VacancyPagination {...base({ currentPage: 2, totalPages: 3, loading: true })} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons.length).toBe(5);
    for (const b of buttons) expect((b as HTMLButtonElement).disabled).toBe(true);
  });
});