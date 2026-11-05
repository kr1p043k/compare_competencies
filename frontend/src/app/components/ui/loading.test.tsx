import { Suspense, lazy } from "react";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { DashboardSkeleton, ListSkeleton } from "./loading";

// Явный cleanup: при singleFork автоочистка RTL не всегда срабатывает
// между файлами, и разметка предыдущего теста остаётся в document.
afterEach(cleanup);

/**
 * Вкладка, чей чанк «не грузится вообще»: промис никогда не резолвится и
 * не реджектится, поэтому компонент зависает в Suspense навсегда. Это
 * устойчивая модель «вечной загрузки» - ровно то состояние, в котором
 * заглушка должна висеть на экране, а не молчать.
 */
const NeverLoads = lazy(
  () => new Promise<{ default: () => null }>(() => {}),
);

describe("ListSkeleton", () => {
  it("держит скелетон вкладки, пока промис ленивой загрузки не резолвится", () => {
    const { container } = render(
      <Suspense fallback={<ListSkeleton label="Загружается раздел «Публикации»…" />}>
        <NeverLoads />
      </Suspense>,
    );

    // Контент вкладки так и не появился - на экране заглушка.
    expect(screen.queryByText("контент вкладки")).not.toBeInTheDocument();

    // Скринридер слышит, что раздел грузится (sr-only текст).
    expect(screen.getByText("Загружается раздел «Публикации»…")).toBeInTheDocument();

    // Заглушка объявлена как busy-статус.
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");

    // Заглушка действительно скелетонная: строки шиммера вместо пустоты.
    expect(container.querySelectorAll(".skeleton").length).toBeGreaterThanOrEqual(5);
  });
});

describe("DashboardSkeleton", () => {
  it("показывает сетку дашборда, пока ленивая вкладка не загрузилась", () => {
    const { container } = render(
      <Suspense fallback={<DashboardSkeleton label="Загружается раздел «Мониторинг»…" />}>
        <NeverLoads />
      </Suspense>,
    );

    expect(screen.getByText("Загружается раздел «Мониторинг»…")).toBeInTheDocument();
    expect(screen.getByRole("status")).toBeInTheDocument();

    // Строка статистики (4 плитки) + два блока контента.
    const tiles = container.querySelectorAll(".skeleton");
    expect(tiles.length).toBe(7); // 1 заголовок + 4 плитки + 2 блока
  });
});
