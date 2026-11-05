import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";

// reportError внутри границы уходит в /api/log. В jsdom это реальный fetch,
// который завершается уже после теста и портит счётчики мокам в других файлах
// (vitest гоняет их в одном процессе). Подменяем модуль целиком - как в
// ErrorBoundary.test.tsx.
vi.mock("../lib/errors", () => ({
  reportError: vi.fn(() => ""),
  onError: vi.fn(() => () => {}),
}));

import ErrorBoundary from "./ErrorBoundary";

afterEach(() => {
  // Явный cleanup: упавший boundary оставляет DOM с role="alert", и без этого
  // он «протекает» в следующие тестовые файлы (singleFork), ломая поиск по роли.
  cleanup();
  vi.restoreAllMocks();
});

/** Мок-компонент, который всегда падает в рендере. */
function Boom(): ReactNode {
  throw new Error("компонент упал");
}

describe("ErrorBoundary: кнопка «Обновить»", () => {
  it("показывает «Обновить» в фолбэке рядом с «Повторить»", () => {
    // React пишет ошибку в console.error: глушим, чтобы не засорять вывод.
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ErrorBoundary scope="test" onReload={() => {}}>
        <Boom />
      </ErrorBoundary>,
    );

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Повторить" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Обновить" })).toBeInTheDocument();

    spy.mockRestore();
  });

  it("вызывает переданный обработчик перезагрузки при клике на «Обновить»", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const user = userEvent.setup();
    // Корневая граница передаёт window.location.reload; в тестах подменяем
    // обработчик, чтобы клик не перезагружал страницы и был видим мокам.
    const onReload = vi.fn();

    render(
      <ErrorBoundary scope="test" onReload={onReload}>
        <Boom />
      </ErrorBoundary>,
    );

    await user.click(screen.getByRole("button", { name: "Обновить" }));

    expect(onReload).toHaveBeenCalledTimes(1);
    // «Повторить» не нажимали: фолбэк остаётся, пока обработчик не перезагрузит страницу.
    expect(screen.getByRole("alert")).toBeInTheDocument();

    spy.mockRestore();
  });

  it("после «Обновить» вкладку по-прежнему можно вернуть кнопкой «Повторить»", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const user = userEvent.setup();

    let shouldThrow = true;
    function Flaky() {
      if (shouldThrow) throw new Error("разово");
      return <div>восстановлено</div>;
    }

    render(
      <ErrorBoundary scope="test" onReload={() => {}}>
        <Flaky />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("button", { name: "Обновить" })).toBeInTheDocument();

    shouldThrow = false;
    await user.click(screen.getByRole("button", { name: "Повторить" }));
    expect(screen.getByText("восстановлено")).toBeInTheDocument();

    spy.mockRestore();
  });
});
