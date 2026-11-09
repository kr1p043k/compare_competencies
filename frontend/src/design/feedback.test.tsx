/**
 * feedback.tsx: Progress и Callout - a11y-атрибуты, тональность, действия.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Callout, Progress } from "./feedback";

describe("Progress", () => {
  it("обрезает значение до 0..100 и отдаёт aria-valuenow", () => {
    render(<Progress value={137} label="Обработка РПД" />);
    const bar = screen.getByRole("progressbar", { name: "Обработка РПД" });
    expect(bar).toHaveAttribute("aria-valuenow", "100");
    expect(bar).toHaveAttribute("aria-valuemin", "0");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
    expect(bar).toHaveAttribute("aria-valuetext", "100%");
  });

  it("не уходит в минус", () => {
    render(<Progress value={-5} label="Прогресс" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
  });

  it("округляет дробные проценты", () => {
    render(<Progress value={41.6} label="Прогресс" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
  });
});

describe("Callout", () => {
  it("по умолчанию - role=status и тон info", () => {
    render(<Callout>Сервис отвечает</Callout>);
    expect(screen.getByRole("status")).toHaveTextContent("Сервис отвечает");
  });

  it("role=alert для ошибок, текст и действие доходят до DOM", () => {
    render(
      <Callout role="alert" tone="danger" action={<button type="button">Повторить</button>}>
        Не удалось загрузить журнал
      </Callout>,
    );
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Не удалось загрузить журнал");
    expect(screen.getByRole("button", { name: "Повторить" })).toBeInTheDocument();
  });

  it("иконка с aria-hidden не меняет текст баннера", () => {
    render(
      <Callout tone="warning" icon={<span aria-hidden="true">⚠</span>}>
        состояние не читается
      </Callout>,
    );
    expect(screen.getByText("состояние не читается")).toBeInTheDocument();
  });
});
