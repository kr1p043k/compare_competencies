import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

// reportError внутри границы уходит в /api/log. В jsdom это реальный fetch,
// который завершается уже после теста и портит счётчики мокам в других файлах
// (vitest гоняет их в одном процессе). Подменяем модуль целиком.
vi.mock("../lib/errors", () => ({
  reportError: vi.fn(() => ""),
  onError: vi.fn(() => () => {}),
}));

import ErrorBoundary from "./ErrorBoundary";

afterEach(() => {
  // Явный cleanup: упавший boundary оставляет DOM с role="alert", и без этого
  // он «протекает» в следующие тестовые файлы (vitest гоняет их в одном
  // процессе: singleFork), ломая поиск по роли.
  cleanup();
  vi.restoreAllMocks();
});

function Boom({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error("компонент упал");
  return <div>всё хорошо</div>;
}

describe("ErrorBoundary", () => {
  it("renders children when there is no error", () => {
    render(
      <ErrorBoundary scope="test">
        <Boom shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("всё хорошо")).toBeInTheDocument();
  });

  it("shows a fallback instead of unmounting the whole tree", () => {
    // React пишет ошибку в console.error: глушим, чтобы не засорять вывод.
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ErrorBoundary scope="test" title="Вкладка сломалась">
        <Boom shouldThrow={true} />
      </ErrorBoundary>,
    );

    const alert = screen.getByRole("alert");
    expect(alert).toBeInTheDocument();
    expect(screen.getByText("Вкладка сломалась")).toBeInTheDocument();
    // Текст ошибки виден: по нему пользователь может описать проблему.
    expect(screen.getByText(/компонент упал/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Повторить" })).toBeInTheDocument();

    spy.mockRestore();
  });

  it("recovers after clicking «Повторить» when the cause is gone", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const user = userEvent.setup();

    let shouldThrow = true;
    function Flaky() {
      if (shouldThrow) throw new Error("разово");
      return <div>восстановлено</div>;
    }

    render(
      <ErrorBoundary scope="test">
        <Flaky />
      </ErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();

    shouldThrow = false;
    await user.click(screen.getByRole("button", { name: "Повторить" }));

    expect(screen.getByText("восстановлено")).toBeInTheDocument();
    spy.mockRestore();
  });
});
